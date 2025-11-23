# pipeline/refiner.py
# Domain-agnostic refinement layer
# - Facet routing via facet description embeddings
# - Heterogeneity check (silhouette) and optional local sub-clustering (KMeans)
# - Stable refined ids; non-destructive to original cluster labels
# - All logs are ASCII only to avoid console encoding issues on Windows

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None

from pathlib import Path
import io
import os

logger = logging.getLogger(__name__)

__all__ = [
    "Facet",
    "load_facets_yml",
    "load_facets_for_category",
    "normalize_category_facet_config",
    "assign_category_facets_keyword",
    "compute_facet_buckets",
    "apply_facet_routing",
    "refine_clusters",
]

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

@dataclass
class Facet:
    id: str
    name: str
    desc: str
    emb: np.ndarray  # normalized vector


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expected 2-D array for normalization")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _is_other(v: Any, other_label_value: Any) -> bool:
    try:
        if isinstance(v, (int, np.integer)) and int(v) < 0:
            return True
    except Exception:
        pass
    s = str(v).strip().lower()
    return s in {"other", "-1", "nan", "none", ""} or v == other_label_value


def _coerce_to_int_or_other(v: Any) -> int:
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().lower()
    if s in {"other", "-1", "nan", "none", ""}:
        return -1
    try:
        return int(s)
    except Exception:
        return -1


# ---------------------------------------------------------------------
# Facet loader (robust, short, single point of truth)
# ---------------------------------------------------------------------

def load_facets_yml(path: str | Path, facet_embedder, *args, **kwargs) -> List[Facet] | None:
    """
    Read a facets YAML in several accepted shapes and return List[Facet] with
    normalized embeddings computed from description text.

    Accepted inputs:
      - {"buckets": [ {id,name,desc|description|keywords}, ... ]}
      - {"facets":  [ {id,name,desc|description|keywords}, ... ]}
      - {"facets":  { name: {"description"| "desc"| "keywords": [...]}, ... }}
      - { name: {"description"| "desc"| "keywords": [...]}, ... }
      - [ {id,name,desc|description|keywords}, ... ]

    Returns:
      List[Facet] or None if nothing usable.
    """
    path = str(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        y = yaml.safe_load(io.StringIO(raw)) or {}
    except Exception as e:
        logger.exception("[FACETS] yaml.safe_load failed: %s", e)
        return None

    def _as_list(obj) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("buckets"), list):
                return obj["buckets"]
            if isinstance(obj.get("facets"), list):
                return obj["facets"]
            if isinstance(obj.get("facets"), dict):
                out = []
                for name, node in obj["facets"].items():
                    node = node or {}
                    out.append({"name": name, **node})
                return out
            # top-level dict of dicts
            if all(isinstance(v, (dict, type(None))) for v in obj.values()):
                out = []
                for name, node in obj.items():
                    node = node or {}
                    out.append({"name": name, **node})
                return out
        return []

    items = _as_list(y)
    if not items:
        logger.error("[FACETS] no items found in %s (root=%s)", os.path.abspath(path), type(y).__name__)
        return None

    names: List[str] = []
    ids: List[str] = []
    descs: List[str] = []
    skipped = 0

    for i, it in enumerate(items):
        name = str(it.get("name") or it.get("id") or f"F{i}")
        fid = str(it.get("id") or name).lower().replace(" ", "_")
        # desc priority: desc > description > keywords(list) > fallback skip
        desc = it.get("desc") or it.get("description")
        if not desc:
            kws = it.get("keywords")
            if isinstance(kws, (list, tuple)) and kws:
                desc = ", ".join(map(str, kws))
        desc = (str(desc).strip() if desc else "")
        if not desc:
            skipped += 1
            continue
        # normalize slashes into commas to stabilize embedding cues
        desc = desc.replace(" / ", ", ").replace("/", ", ")
        names.append(name)
        ids.append(fid)
        descs.append(desc)

    if not names:
        logger.error("[FACETS] all items were empty after normalization (skipped=%d) in %s", skipped, os.path.abspath(path))
        return None

    if facet_embedder is None:
        logger.error("[FACETS] facet_embedder is None")
        return None

    try:
        vecs = facet_embedder.encode(
            descs,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs = _normalize_rows(vecs)
    except Exception as e:
        logger.exception("[FACETS] embedding failed: %s", e)
        return None

    facets: List[Facet] = [Facet(id=ids[i], name=names[i], desc=descs[i], emb=vecs[i]) for i in range(len(names))]
    logger.info(
        "[FACETS] loaded=%d usable (skipped=%d) | emb.shape=(%d,%d) | head=%s",
        len(facets),
        skipped,
        vecs.shape[0],
        vecs.shape[1],
        [f.name for f in facets[:3]],
    )
    return facets


def load_facets_for_category(
    category: str,
    sku: Optional[str] = None,
    *,
    path: str | Path = Path("rules/facets_by_category.yml"),
    fallback: Any = None,
) -> Any:
    """Load category-specific facet config from ``rules/facets_by_category.yml``.

    Returns the category node when present, otherwise falls back to ``fallback``
    (typically the global facet config). Missing/parse errors are logged but do
    not raise so downstream routing can continue unchanged.
    """

    normalized_category = (category or "").strip().lower() or "generic"
    cfg_path = Path(path)

    if yaml is None:  # pragma: no cover - defensive
        logger.warning(
            "[FACETS] PyYAML unavailable; skipping category facets and using fallback"
        )
        return fallback

    if not cfg_path.exists():
        logger.warning(
            "[FACETS] category facets file missing at %s; using fallback facets", cfg_path
        )
        return fallback

    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning(
            "[FACETS] failed to parse %s; using fallback facets", cfg_path, exc_info=True
        )
        return fallback

    categories = payload.get("categories") if isinstance(payload, dict) else {}
    selected = None
    if isinstance(categories, dict):
        selected = categories.get(normalized_category)
        used_category = normalized_category
        if selected is None and "generic" in categories:
            selected = categories.get("generic")
            used_category = "generic"
    else:
        used_category = normalized_category

    if selected is None:
        logger.info(
            "[FACETS] category '%s' not found in %s; using fallback facets (sku=%s)",
            normalized_category,
            cfg_path,
            sku or "-",
        )
        return fallback

    logger.info(
        "[FACETS] category facets selected: cat=%s sku=%s source=%s",
        used_category,
        sku or "-",
        cfg_path,
    )
    return selected


def normalize_category_facet_config(facet_config: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a category facet config node into ``facet_id -> metadata`` mapping.

    Each value contains ``label_ko`` and keyword lists for positive/negative cues.
    Missing fields are filled with safe defaults so callers can assume presence.
    """

    if not isinstance(facet_config, dict):
        return {}

    # The expected shape is {"facets": {facet_id: {label_ko, positive_keywords, negative_keywords}}}
    facets_node = facet_config.get("facets") if isinstance(facet_config, dict) else None
    if not isinstance(facets_node, dict):
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for facet_id, node in facets_node.items():
        if node is None:
            node = {}
        if not isinstance(node, dict):
            continue

        pos_kws_raw = node.get("positive_keywords") or []
        neg_kws_raw = node.get("negative_keywords") or []

        def _normalize_keywords(seq: Any) -> List[str]:
            if not isinstance(seq, (list, tuple)):
                return []
            out: List[str] = []
            for kw in seq:
                try:
                    s = str(kw).strip()
                except Exception:
                    continue
                if not s:
                    continue
                out.append(s.lower())
            return out

        normalized[facet_id] = {
            "label_ko": str(node.get("label_ko") or ""),
            "positive_keywords": _normalize_keywords(pos_kws_raw),
            "negative_keywords": _normalize_keywords(neg_kws_raw),
        }

    return normalized


def assign_category_facets_keyword(
    df: pd.DataFrame,
    facet_config: Any,
    *,
    text_column: str = "clause",
    polarity_column: str = "polarity",
    output_column: str = "facet_ids",
) -> pd.DataFrame:
    """
    Assign category-specific facet ids to each row using simple keyword matching.

    Returns a DataFrame with ``output_column`` populated as a list of facet ids
    (or empty list) based on polarity-aware keyword cues.
    """

    if df is None or df.empty:
        return df

    if text_column not in df.columns or polarity_column not in df.columns:
        return df

    normalized = normalize_category_facet_config(facet_config)
    if not normalized:
        return df

    out = df.copy()

    def _match_row(text: str, polarity: str) -> List[str]:
        txt = str(text).lower()
        pol = str(polarity).strip().lower()
        matched: List[str] = []

        for facet_id, meta in normalized.items():
            pos_kws = meta.get("positive_keywords") or []
            neg_kws = meta.get("negative_keywords") or []

            if pol == "neg":
                candidates = neg_kws
            elif pol == "pos":
                candidates = pos_kws
            else:
                # For neutral, allow either set to surface facet cues.
                candidates = (pos_kws or []) + (neg_kws or [])

            if any(kw in txt for kw in candidates):
                matched.append(facet_id)

        # Deterministic order
        return sorted(set(matched))

    out[output_column] = [
        _match_row(text, pol)
        for text, pol in zip(out[text_column].astype(str), out[polarity_column])
    ]

    return out


def compute_facet_buckets(
    df: pd.DataFrame,
    *,
    facet_ids_column: str = "facet_ids",
    polarity_column: str = "polarity",
    bucket_column: str = "facet_bucket",
    unmatched_facet_name: str = "unmatched",
) -> pd.DataFrame:
    """Compute a primary facet bucket label per row based on facet ids and polarity."""

    if df is None or df.empty or facet_ids_column not in df.columns:
        return df

    out = df.copy()

    def _coerce_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, tuple):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            cleaned = v.strip().strip("[]")
            parts = [p.strip(" ' \"") for p in cleaned.split(",") if p.strip(" ' \"")]
            return [p for p in parts if p]
        return []

    def _pick_bucket(ids: List[str], pol: str) -> str:
        facet = sorted(ids)[0] if ids else unmatched_facet_name
        polarity = str(pol).strip().lower() or "neu"
        return f"{facet}_{polarity}"

    out[bucket_column] = [
        _pick_bucket(_coerce_list(ids), pol)
        for ids, pol in zip(out[facet_ids_column], out[polarity_column])
    ]

    return out


# ---------------------------------------------------------------------
# Facet routing
# ---------------------------------------------------------------------

def route_to_facets(
    clause_embs: np.ndarray,
    facets: List[Facet],
    *,
    top_k: int = 2,
    score_threshold: float = 0.32,
) -> Tuple[List[Optional[str]], List[List[Tuple[str, float]]]]:
    """
    Compute cosine similarity (dot on normalized vectors) between each clause and facet.
    Returns:
      top1 list of facet ids (or None) and topk list of (facet_id, score).
    """
    if not facets:
        return [None] * len(clause_embs), [[] for _ in range(len(clause_embs))]
    F = np.stack([f.emb for f in facets], axis=0)  # (F, D), already normalized
    X = _normalize_rows(clause_embs)               # (N, D)
    sims = X @ F.T                                 # (N, F)

    k = max(1, min(top_k, sims.shape[1]))
    # partial top-k then stable sort
    idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    sims_part = np.take_along_axis(sims, idx_part, axis=1)
    order = np.argsort(-sims_part, axis=1)
    idx_sorted = np.take_along_axis(idx_part, order, axis=1)
    sims_sorted = np.take_along_axis(sims_part, order, axis=1)

    top1: List[Optional[str]] = []
    topk: List[List[Tuple[str, float]]] = []

    for i in range(sims.shape[0]):
        pairs: List[Tuple[str, float]] = []
        for j in range(k):
            score = float(sims_sorted[i, j])
            if score >= score_threshold:
                pairs.append((facets[idx_sorted[i, j]].id, score))
        topk.append(pairs)
        top1.append(pairs[0][0] if pairs else None)

    return top1, topk


def apply_facet_routing(
    df: pd.DataFrame,
    facets: List[Facet],
    *,
    clause_embs: Optional[np.ndarray] = None,
    embedder: Optional[Any] = None,
    text_column: str = "clause",
    top_k: int = 2,
    threshold: float = 0.32,
    category: Optional[str] = None,
    facet_config: Any = None,
    sku: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fill facet_top1 / facet_topk using cosine similarity to facets.

    Inputs:
      - facets: List[Facet] with normalized emb vectors
      - clause_embs: if provided, use directly; else, encode df[text_column] with embedder
      - embedder: SentenceTransformer-like (encode with normalize_embeddings=True)
    Notes:
      - Existing values are preserved; only NaN/blank cells are filled.
      - facet_topk is a JSON-like string "[(id, score), ...]" for readability.
    """
    if df is None or df.empty or not facets:
        return df

    # get clause vectors
    if clause_embs is None:
        if embedder is None:
            logger.warning("[FACETS] no clause_embs and no embedder; routing skipped")
            return df
        texts = df[text_column].astype(str).tolist()
        clause_embs = embedder.encode(
            texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    clause_embs = _normalize_rows(np.asarray(clause_embs, dtype=np.float32))

    # compute routing
    top1_vals, topk_vals = route_to_facets(
        clause_embs, facets, top_k=top_k, score_threshold=float(threshold)
    )

    top1_series = pd.Series([v if v else None for v in top1_vals], index=df.index, dtype=object)
    # store as simple str to keep xlsx friendly and avoid quoting issues
    topk_series = pd.Series(
        [str(pairs) if pairs else None for pairs in topk_vals],
        index=df.index,
        dtype=object,
    )

    out = df.copy()

    # facet_top1
    if "facet_top1" in out.columns:
        exists = out["facet_top1"]
        needs = exists.isna() | exists.astype(str).str.strip().isin(["", "nan", "none", "None"])
        out.loc[needs, "facet_top1"] = top1_series.loc[needs]
    else:
        out["facet_top1"] = top1_series

    # facet_topk
    if "facet_topk" in out.columns:
        exists = out["facet_topk"]
        needs = exists.isna() | exists.astype(str).str.strip().isin(["", "nan", "none", "None", "[]", "{}"])
        out.loc[needs, "facet_topk"] = topk_series.loc[needs]
    else:
        out["facet_topk"] = topk_series

    # clean blanks to NaN
    mask_blank1 = out["facet_top1"].astype(str).str.strip().isin(["", "nan", "none", "None"])
    out.loc[mask_blank1, "facet_top1"] = None
    if "facet_topk" in out.columns:
        mask_blankk = out["facet_topk"].astype(str).str.strip().isin(["", "nan", "none", "None"])
        out.loc[mask_blankk, "facet_topk"] = None

    # ---------------------------------------------------------------
    # Category-aware keyword facet assignment (non-destructive)
    # ---------------------------------------------------------------
    normalized_cfg = normalize_category_facet_config(facet_config)
    if normalized_cfg:
        try:
            out = assign_category_facets_keyword(
                out,
                {"facets": normalized_cfg},
                text_column=text_column,
                polarity_column="polarity",
                output_column="facet_ids",
            )
            out = compute_facet_buckets(
                out,
                facet_ids_column="facet_ids",
                polarity_column="polarity",
                bucket_column="facet_bucket",
                unmatched_facet_name="unmatched",
            )

            def _has_facets(v: Any) -> bool:
                if isinstance(v, (list, tuple, set)):
                    return len(v) > 0
                if isinstance(v, str):
                    cleaned = v.strip().strip("[]")
                    return bool(cleaned)
                return False

            non_empty = out["facet_ids"].apply(_has_facets).sum()
            bucket_counts = (
                out.get("facet_bucket")
                .value_counts(dropna=True)
                .head(5)
                .to_dict()
                if "facet_bucket" in out.columns
                else {}
            )
            logger.info(
                "[FACETS] keyword assignment done | sku=%s cat=%s | matched_rows=%d/%d | top_buckets=%s",
                sku or "-",
                category or "-",
                int(non_empty),
                len(out),
                bucket_counts,
            )
        except Exception:
            logger.warning(
                "[FACETS] keyword-based facet assignment skipped due to error; sku=%s cat=%s",
                sku or "-",
                category or "-",
                exc_info=True,
            )

    return out


# ---------------------------------------------------------------------
# Heterogeneity and local sub-clustering
# ---------------------------------------------------------------------

def heterogeneity_score(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 4,
    random_state: int = 42,
) -> Tuple[float, Optional[int]]:
    """
    Try KMeans k in [min_k, max_k], return (best_silhouette, best_k).
    If best_k is None, do not split.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < max(8, 2 * min_k):
        return 0.0, None
    best_s, best_k = 0.0, None
    for k in range(min_k, min(max_k, n - 1) + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(X)
            s = silhouette_score(X, labels)
            if s > best_s:
                best_s, best_k = float(s), k
        except Exception:
            continue
    return float(best_s), best_k


def local_subcluster_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    return km.fit_predict(X)


# ---------------------------------------------------------------------
# Main refine per polarity
# ---------------------------------------------------------------------

def refine_clusters(
    df_clauses: pd.DataFrame,
    clause_embs: np.ndarray,
    polarity: str,
    facets: List[Facet],
    *,
    top_k_facets: int = 2,
    facet_threshold: float = 0.32,
    hetero_sil_threshold: float = 0.18,
    min_cluster_size_for_split: int = 40,
    max_local_k: int = 4,
    other_label_value: str | int = "other",
    stable_id_prefix: int = 0,  # negative=0, neutral=1, positive=2
) -> pd.DataFrame:
    """
    Refine within one polarity.
    Non-destructive: keeps original cluster_label and adds refined_label / refined_cluster_id.
    Also fills facet_top1 / facet_topk from provided facets.
    Assumes df_clauses contains this polarity only and aligns with clause_embs.
    """
    if df_clauses is None or df_clauses.empty:
        return df_clauses

    df = df_clauses.copy()

    if "cluster_label" not in df.columns:
        raise ValueError("cluster_label column is required")

    # normalize labels up-front to avoid int('other') errors
    df["cluster_label"] = df["cluster_label"].apply(_coerce_to_int_or_other)

    # normalize embeddings
    clause_embs = _normalize_rows(np.asarray(clause_embs, dtype=np.float32))

    # facet routing into new columns, preserving existing annotations
    top1_vals, topk_vals = route_to_facets(
        clause_embs, facets, top_k=top_k_facets, score_threshold=facet_threshold
    )
    top1_series = pd.Series([v if v else None for v in top1_vals], index=df.index, dtype=object)
    topk_series = pd.Series([str(p) if p else None for p in topk_vals], index=df.index, dtype=object)

    if "facet_top1" in df.columns:
        needs = df["facet_top1"].isna() | df["facet_top1"].astype(str).str.strip().isin(["", "nan", "none", "None"])
        df.loc[needs, "facet_top1"] = top1_series.loc[needs]
    else:
        df["facet_top1"] = top1_series

    if "facet_topk" in df.columns:
        needs = df["facet_topk"].isna() | df["facet_topk"].astype(str).str.strip().isin(["", "nan", "none", "None", "[]", "{}"])
        df.loc[needs, "facet_topk"] = topk_series.loc[needs]
    else:
        df["facet_topk"] = topk_series

    # clean blanks
    blank1 = df["facet_top1"].astype(str).str.strip().isin(["", "nan", "none", "None"])
    df.loc[blank1, "facet_top1"] = None
    blankk = df["facet_topk"].astype(str).str.strip().isin(["", "nan", "none", "None"])
    df.loc[blankk, "facet_topk"] = None

    # default refined_label = original cluster_label
    df["refined_label"] = df["cluster_label"].values

    # split clusters if heterogeneous
    for cl, sub in df.groupby("cluster_label", sort=False):
        if _is_other(cl, other_label_value):
            continue
        idx = sub.index.to_numpy()
        if idx.size < min_cluster_size_for_split:
            continue
        X = clause_embs[idx]
        sil, best_k = heterogeneity_score(X, min_k=2, max_k=max_local_k)
        if best_k and sil >= hetero_sil_threshold:
            sub_labels = local_subcluster_kmeans(X, k=best_k)
            base = _safe_int(cl)
            base = base if (base is not None and base >= 0) else 0
            df.loc[idx, "refined_label"] = [base * 10 + int(s) for s in sub_labels]

    # stable refined id within polarity namespace
    prefix_map = {"negative": 0, "neutral": 1, "positive": 2}
    try:
        prefix = int(stable_id_prefix)
    except (TypeError, ValueError):
        prefix = prefix_map.get(polarity, 0)

    def _mk_id(v: Any) -> int:
        if _is_other(v, other_label_value):
            return prefix * 1000 + 999
        vi = _safe_int(v)
        return prefix * 1000 + (vi if vi is not None else 999)

    df["refined_cluster_id"] = df["refined_label"].map(_mk_id)
    df["polarity"] = polarity
    return df

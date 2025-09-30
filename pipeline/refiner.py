# pipeline/refiner.py
# General refinement layer (domain-agnostic):
# - facet routing via label description embeddings
# - cluster heterogeneity check (silhouette)
# - conditional local sub-clustering (KMeans)
# - stable refined ids (non-destructive: preserves original cluster_label)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
try:
    import yaml  # pyyaml
except Exception:
    yaml = None
from pathlib import Path

# ---- Data models ----

@dataclass
class Facet:
    id: str
    desc: str
    emb: np.ndarray

# ---- Facet utilities ----

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_facets_yml(path: str, embedder) -> List[Facet]:
    """Load facets and embed their descriptions using the same sentence model.
    `embedder` must expose `.encode(list[str], convert_to_numpy=True, normalize_embeddings=True)`.
    """
    if yaml is None:
        raise ImportError("pyyaml is required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    items = y.get("facets", [])
    texts = [f"{it['id']}: {it.get('desc','')}" for it in items]
    embs = embedder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    facets = [Facet(id=it['id'], desc=it.get('desc',''), emb=e) for it, e in zip(items, embs)]
    return facets

def route_to_facets(
    clause_embs: np.ndarray,
    facets: List[Facet],
    top_k: int = 2,
    score_threshold: float = 0.32,
) -> Tuple[List[str], List[List[Tuple[str, float]]]]:
    """Return (top1_facet_id, topk list) per row using cosine on normalized embeddings.
    - clause_embs: (N, D) assumed normalized
    - facets: list of Facet with normalized embeddings
    """
    F = np.stack([f.emb for f in facets], axis=0)  # (F, D)
    sims = clause_embs @ F.T  # cosine if normalized
    # top-k indices
    kth = max(0, min(top_k, sims.shape[1]) - 1)
    idx = np.argpartition(-sims, kth=kth, axis=1)[:, :top_k]
    topk_sorted = np.take_along_axis(sims, idx, axis=1)
    # sort within topk
    order = np.argsort(-topk_sorted, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    sims_sorted = np.take_along_axis(topk_sorted, order, axis=1)

    top1 = []
    topk = []
    for i in range(sims.shape[0]):
        pairs = []
        for j in range(min(top_k, sims.shape[1])):
            if sims_sorted[i, j] >= score_threshold:
                pairs.append((facets[idx_sorted[i, j]].id, float(sims_sorted[i, j])))
        topk.append(pairs)
        top1.append(pairs[0][0] if pairs else "")
    return top1, topk

def load_facets_yml(path: str | Path, embedder):
    import yaml
    y = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = y.get("facets") or []

    # ← NEW: dict/list 모두 허용
    if isinstance(items, dict):
        items = [
            {
                "id": fid,
                "desc": (v.get("desc") or v.get("description") or ""),
                "keywords": list(v.get("keywords") or []),
            }
            for fid, v in items.items()
        ]
    elif isinstance(items, list):
        normed = []
        for it in items:
            fid = it.get("id") or it.get("name")
            desc = it.get("desc") or it.get("description") or ""
            kws  = list(it.get("keywords") or [])
            if not fid:
                raise ValueError("facet item missing 'id'")
            normed.append({"id": fid, "desc": desc, "keywords": kws})
        items = normed
    else:
        raise ValueError("facets must be a list or a dict")

# ---- Robust outlier helpers ----

def _is_other(v, other_label_value):
    """Return True if value represents 'other'/outlier."""
    try:
        if isinstance(v, (int, np.integer)) and int(v) < 0:
            return True
    except Exception:
        pass
    s = str(v).strip().lower()
    return s in {"other", "-1", "nan", "none", ""} or v == other_label_value

def _safe_int(v):
    try:
        return int(v)
    except Exception:
        return None

def _coerce_to_int_or_other(v):
    """Coerce any 'other'ish label to -1, else cast to int.
    Accepts int/str/None/NaN; safe for mixed inputs.
    """
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().lower()
    if s in {"other", "-1", "nan", "none", ""}:
        return -1
    try:
        return int(s)
    except Exception:
        return -1

# ---- Heterogeneity & local sub-clustering ----

def heterogeneity_score(
    X: np.ndarray,
    min_k: int = 2,
    max_k: int = 4,
    random_state: int = 42,
) -> Tuple[float, Optional[int]]:
    """Return (best_silhouette, best_k). If best_k is None, do not split.
    Heuristic: try KMeans k in [min_k, max_k], keep the highest silhouette.
    """
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
                best_s, best_k = s, k
        except Exception:
            continue
    return float(best_s), best_k

def local_subcluster_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    return km.fit_predict(X)

# ---- Main refine ----

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
    """Refine within one polarity.
    - Non-destructive: keeps original `cluster_label` and adds `refined_label` & `refined_cluster_id`.
    - Adds `facet_top1` and `facet_topk` (JSON-able string) columns.
    Assumes `df_clauses` contains this polarity only and row order aligns with clause_embs.
    """
    df = df_clauses.copy()

    # --- normalize labels up-front (critical: avoid int('other') errors) ---
    if "cluster_label" not in df.columns:
        raise ValueError("'cluster_label' column is required in df_clauses")
    df["cluster_label"] = df["cluster_label"].apply(_coerce_to_int_or_other)

    # normalize to cosine space if not already
    clause_embs = _normalize_rows(clause_embs.astype(np.float32))

    # Facet routing
    facet_top1, facet_topk = route_to_facets(
        clause_embs, facets, top_k=top_k_facets, score_threshold=facet_threshold
    )
    df["facet_top1"] = facet_top1
    df["facet_topk"] = [json.dumps(pairs, ensure_ascii=False) for pairs in facet_topk]

    # Prepare refined labels default = original
    df["refined_label"] = df["cluster_label"].values

    # Split per cluster if heterogeneous
    for cl, sub in df.groupby("cluster_label", sort=False):
        if _is_other(cl, other_label_value):
            # keep outliers untouched
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

    # Stable refined cluster id within polarity namespace: prefix*1000 + ...
    def _mk_id(v):
        if _is_other(v, other_label_value):
            return stable_id_prefix * 1000 + 999
        vi = _safe_int(v)
        return stable_id_prefix * 1000 + (vi if vi is not None else 999)

    df["refined_cluster_id"] = df["refined_label"].map(_mk_id)
    df["polarity"] = polarity
    return df

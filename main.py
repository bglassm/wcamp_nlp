from __future__ import annotations

import argparse
import warnings
import logging
import sys
import os
import hashlib
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time
import tempfile

import pandas as pd
import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    from sentence_transformers import util as _st_util
    _st_util.set_progress_bar_enabled(False)
except Exception:
    pass

try:
    import tqdm
    tqdm.tqdm = lambda *a, **k: iter(a[0]) if a else iter([])
except Exception:
    pass

import config
from pipeline.loader import load_reviews
from pipeline.preprocess import preprocess_reviews, clean_review_text
from pipeline.clause_splitter import split_clauses
from pipeline.absa import classify_clauses
from pipeline.embedder import get_embedder, CachingEmbedder, _get_model
from pipeline.reducer import reduce_embeddings
from pipeline.clusterer import cluster_embeddings, evaluate_clusters
from pipeline.tuner import get_cluster_params
from pipeline.summarizer import extract_representatives
from pipeline.mergy import merge_similar_clusters
from pipeline.keywords import extract_keywords
from pipeline.exporter import (
    save_clustered_clauses,
    save_clauses_summary_json,
)
from pipeline.idmap import assign_stable_ids
from pipeline.report import save_client_report
from pipeline.refiner import (
    load_facets_for_category,
    load_facets_yml,
    refine_clusters,
    _normalize_rows,
    apply_facet_routing,
)
from utils.runmeta import write_run_manifest, write_meta_json
from pipeline.visualizer import generate_run_report

try:
    import yaml
except ImportError as _e:
    raise SystemExit("PyYAML is required. Run: pip install pyyaml") from _e

logging.getLogger("pyabsa").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("kss").setLevel(logging.ERROR)
logging.getLogger("weasel").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

logger = logging.getLogger(__name__)


# --- label offset helpers ---

def _offset_labels(labels: np.ndarray, base: int) -> np.ndarray:
    """Shift HDBSCAN labels by polarity base (neg=0, neu=1000, pos=2000).
    Noise label -1 is mapped to base+999."""
    out = []
    for l in labels:
        if isinstance(l, str):
            l = -1 if l.lower() == "other" else int(l)
        out.append(base + 999 if int(l) == -1 else int(l) + base)
    return np.array(out, dtype=int)


def _relabel_dict(d: Dict[int, list], base: int) -> Dict[int, list]:
    """Shift cluster ID keys in a reps/keywords dict by base."""
    new_d: Dict[int, list] = {}
    for k, v in d.items():
        try:
            new_d[int(k) + base] = v
        except Exception:
            new_d[base + 999] = v
    return new_d


def _ensure_facet_top1(df: pd.DataFrame, *, default: str | None = None) -> pd.DataFrame:
    """Ensure facet_top1 column exists, copying from facet_bucket if available."""
    if "facet_top1" in df.columns:
        return df
    if "facet_bucket" in df.columns:
        out = df.assign(facet_top1=df["facet_bucket"])
        mask_blank = out["facet_top1"].astype(str).str.strip() == ""
        out.loc[mask_blank, "facet_top1"] = None
        return out
    return df.assign(facet_top1=default)


def _normalize_review_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


def _has_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


# --- facets YAML loader with schema normalization ---

def _load_facets_forgiving(facets_path: Path, facet_embedder):
    """Load a facets YAML and normalize it to the standard list-of-buckets schema.
    Supports four schema variants; writes a temp YAML and delegates to load_facets_yml."""
    with facets_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
        _abs_path = os.path.abspath(facets_path)
        _root_type = type(y).__name__
        _root_keys = list(y.keys())[:10] if isinstance(y, dict) else []
        logger.info("[FACETS] loaded: %s | root=%s keys=%s", _abs_path, _root_type, _root_keys)

    def to_list(obj):
        """Convert any supported schema to a list of bucket dicts."""
        def _normalize_keywords(raw):
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, (list, tuple)):
                return []
            return [str(kw).strip() for kw in raw if str(kw).strip()]

        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "buckets" in obj and isinstance(obj["buckets"], list):
                return obj["buckets"]
            if "facets" in obj and isinstance(obj["facets"], dict) and obj["facets"]:
                buckets = []
                for name, node in obj["facets"].items():
                    node = node or {}
                    kw_list = _normalize_keywords(node.get("keywords"))
                    desc = None
                    for k in ("desc", "description"):
                        v = node.get(k)
                        if v and str(v).strip():
                            desc = str(v).strip()
                            break
                    if not desc:
                        desc = (
                            f"{name} 관련 표현: " + " ".join(map(str, kw_list))
                            if kw_list
                            else f"{name}이/가 좋다 나쁘다 만족 불만"
                        )
                    bucket = {"id": str(name).lower().replace(" ", ""), "name": str(name), "desc": desc}
                    if kw_list:
                        bucket["keywords"] = kw_list
                    buckets.append(bucket)
                return buckets
        raise RuntimeError(f"Unsupported facets YAML schema: {type(obj).__name__}")

    facets_list = to_list(y)
    logger.info("[FACETS] normalized buckets: %d | head=%s",
                len(facets_list),
                [str(d.get("name") or d.get("id")) for d in facets_list[:3]])

    def _try_build_facets(payload, tag):
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yml", encoding="utf-8")
        yaml.safe_dump(payload, tmp, allow_unicode=True, sort_keys=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            obj = load_facets_yml(str(tmp_path), facet_embedder)
            if obj:
                logger.info("[FACETS] schema=%s OK", tag)
                return obj, tmp_path
            else:
                logger.warning("[FACETS] schema=%s returned empty, trying fallback", tag)
                return None, tmp_path
        except Exception as e:
            logger.warning("[FACETS] schema=%s raised %s, trying fallback", tag, e)
            return None, tmp_path

    facets_obj, tmp_path = _try_build_facets({"buckets": facets_list}, tag="buckets")
    if not facets_obj:
        facets_obj, tmp_path = _try_build_facets({"facets": facets_list}, tag="facets")

    if not facets_obj:
        logger.error("[FACETS] load_facets_yml returned empty for %s", _abs_path)
        raise ValueError(f"No facets loaded after normalization: {facets_path}")

    bucket_names = [d.get("name", f"F{i}") for i, d in enumerate(facets_list)]
    logger.info("[FACETS] bucket_names(head)=%s", bucket_names[:5])
    return facets_obj, bucket_names, tmp_path


# --- embedding-only pipeline ---

def run_embed_pipeline(
    input_files: List[Path],
    output_dir: Path,
    *,
    save_stem: str,
) -> None:
    """Run embedding and cache to disk without clustering."""
    total = len(input_files)
    logging.info("[PIPELINE] embed-only start: %d files", total)
    embedder: CachingEmbedder = get_embedder(config)

    for i, input_file in enumerate(input_files):
        product_id = save_stem or input_file.stem
        logging.info("[PIPELINE] file %d/%d: %s (product=%s)", i + 1, total, input_file.name, product_id)

        df_reviews = load_reviews(input_file)
        df_reviews = preprocess_reviews(df_reviews)
        df_clauses = split_clauses(df_reviews)
        df_clauses = classify_clauses(df_clauses)

        rid_col = getattr(config, "REVIEW_ID_COL", "review_id")
        df_clauses["clause_idx"] = df_clauses.groupby(rid_col).cumcount()
        df_clauses["clause_id"] = df_clauses.apply(
            lambda row: f"{product_id}_{row[rid_col]}_{row['clause_idx']}", axis=1
        )
        embedder.embed(
            texts=df_clauses["clause"].tolist(),
            clause_ids=df_clauses["clause_id"].tolist(),
            product_id=product_id,
        )
        logging.info("[PIPELINE] embed cached: %s", product_id)

    logging.info("[PIPELINE] embed-only done")


# --- full pipeline ---

def run_full_pipeline(
    input_files: List[Path],
    output_dir: Path,
    *,
    resume: bool = False,
    facets_path_override: str | None = None,
    thresholds_path_override: str | None = None,
    alias_terms: List[str] | None = None,
    save_stem: str | None = None,
    audit_root: Path | None = None,
) -> None:
    total = len(input_files)
    logging.info("[PIPELINE] full start: %d files", total)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    main_embedder: CachingEmbedder = get_embedder(config)
    last_embed_dim: int | None = None

    # polarity -> label offset base
    base_map = {"negative": 0, "neutral": 1000, "positive": 2000}

    # --- refinement setup ---
    refine_enabled_cfg = getattr(config, "REFINEMENT_ENABLED", True)
    refine_enabled = refine_enabled_cfg
    facets_obj = None
    refine_th = {
        "top_k_facets": 2,
        "facet_threshold": 0.32,
        "hetero_sil_threshold": 0.18,
        "min_cluster_size_for_split": 40,
        "max_local_k": 4,
        "other_label_value": -1,
    }
    stable_id_prefix_map = getattr(
        config, "REFINEMENT_STABLE_ID", {"negative": 0, "neutral": 1, "positive": 2}
    )

    norm_tmp_path = None
    semantic_helper_model = None
    semantic_helper_name = None
    semantic_helper_device = None
    try:
        facets_path = Path(
            facets_path_override or getattr(config, "REFINEMENT_FACETS_PATH", "rules/facets.yml")
        )
        thresholds_path = Path(
            thresholds_path_override or getattr(config, "REFINEMENT_THRESHOLDS_PATH", "rules/thresholds.yml")
        )
        logging.info("[REFINE] facets=%s | thresholds=%s", facets_path, thresholds_path)

        semantic_cfg = getattr(config, "semantic", None)
        semantic_helper_name = (
            getattr(semantic_cfg, "model", None) or getattr(config, "MODEL_NAME", "jhgan/ko-sbert-sts")
        )
        semantic_helper_device = getattr(semantic_cfg, "device", None) or getattr(config, "DEVICE", None)
        logger.info("[REFINE] semantic helper: %s on %s", semantic_helper_name, semantic_helper_device or "auto")

        facet_embedder = _get_model(semantic_helper_name, semantic_helper_device)
        semantic_helper_model = facet_embedder

        facets_obj, facet_names, norm_tmp_path = _load_facets_forgiving(facets_path, facet_embedder)

        with thresholds_path.open("r", encoding="utf-8") as f:
            config_payload = yaml.safe_load(f) or {}
            refine_th.update((config_payload.get("refinement") or {}))

        head = ", ".join(facet_names[:8]) + ("..." if len(facet_names) > 8 else "")
        logging.info("[REFINE] assets loaded: %d buckets -> %s", len(facet_names), head)

    except Exception:
        logging.exception("[REFINE] init failed, refinement disabled")
        facets_obj = None
        refine_enabled = False
        semantic_helper_model = None
        semantic_helper_name = None
        semantic_helper_device = None
    finally:
        if norm_tmp_path and norm_tmp_path.exists():
            try:
                norm_tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not refine_enabled_cfg:
        logging.info("[REFINE] disabled by config")

    refine_enabled = refine_enabled_cfg and refine_enabled and (facets_obj is not None)

    audit_dir = audit_root or (output_dir / "audit")
    audit_dir.mkdir(parents=True, exist_ok=True)

    # --- per-file loop ---
    for idx, file_path in enumerate(input_files, start=1):
        stem_effective = save_stem or file_path.stem
        out_dir = output_dir / stem_effective
        out_dir.mkdir(parents=True, exist_ok=True)

        category = config.infer_category(file_path, stem_effective)
        logging.info("[FILE %d/%d] %s | category=%s", idx, total, stem_effective, category)
        facet_config = load_facets_for_category(
            category, sku=stem_effective, fallback=facets_obj
        )

        # step 1: load and preprocess
        t0 = time.time()
        logging.info("[1] loading & preprocessing")

        df = load_reviews(file_path)

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        if "review" not in df.columns:
            _cands = ["text", "body", "content", "contents", "summary"]
            _hit = next((c for c in _cands if c in df.columns), None)
            if _hit:
                df = df.rename(columns={_hit: "review"})
            else:
                raise SystemExit("[load] column 'review' not found and no fallback candidate present")

        rid_col = getattr(config, "REVIEW_ID_COL", "review_id")
        if rid_col not in df.columns:
            df = df.reset_index(drop=False).rename(columns={"index": rid_col})
        df[rid_col] = df[rid_col].astype(str)

        raw_reviews = df["review"].astype(str).fillna("")
        clean_reviews = raw_reviews.map(clean_review_text)
        raw_len = raw_reviews.str.len()
        clean_len = clean_reviews.str.len()
        norm_text = clean_reviews.map(_normalize_review_text)
        norm_hash = norm_text.map(_hash_text)
        dup_count = norm_text.map(norm_text.value_counts())

        audit_base = pd.DataFrame({
            rid_col: df[rid_col],
            "__raw_review": raw_reviews,
            "__clean_review": clean_reviews,
            "__raw_len": raw_len,
            "__clean_len": clean_len,
            "__norm_hash": norm_hash,
            "__dup_count": dup_count,
        })
        audit_lookup = audit_base.set_index(rid_col, drop=False)
        drop_records: Dict[str, dict] = {}

        def _record_drop(mask: pd.Series, stage: str, reason: str) -> None:
            for i in audit_base.index[mask]:
                rid = str(audit_base.at[i, rid_col])
                if rid in drop_records:
                    continue
                drop_records[rid] = {
                    "review_id": rid,
                    "raw_len": int(audit_base.at[i, "__raw_len"]),
                    "cleaned_len": int(audit_base.at[i, "__clean_len"]),
                    "drop_stage": stage,
                    "drop_reason": reason,
                    "norm_hash": audit_base.at[i, "__norm_hash"],
                    "raw_preview": str(audit_base.at[i, "__raw_review"])[:120],
                }

        def _record_drop_by_ids(ids: set[str], stage: str, reason: str) -> None:
            for rid in ids:
                rid_str = str(rid)
                if rid_str in drop_records or rid_str not in audit_lookup.index:
                    continue
                row = audit_lookup.loc[rid_str]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                drop_records[rid_str] = {
                    "review_id": rid_str,
                    "raw_len": int(row["__raw_len"]),
                    "cleaned_len": int(row["__clean_len"]),
                    "drop_stage": stage,
                    "drop_reason": reason,
                    "norm_hash": row["__norm_hash"],
                    "raw_preview": str(row["__raw_review"])[:120],
                }

        min_len = 0
        require_korean = bool(getattr(config, "REVIEW_REQUIRE_KOREAN", False))
        min_len_mask = pd.Series(True, index=df.index)
        korean_mask = raw_reviews.map(_has_korean) if require_korean else pd.Series(True, index=df.index)
        load_keep_mask = min_len_mask & korean_mask

        if min_len > 0:
            _record_drop(~min_len_mask, "load", "load_filtered_min_len")
        if require_korean:
            _record_drop(min_len_mask & ~korean_mask, "load", "load_filtered_non_korean")

        preprocess_mask = load_keep_mask.copy()
        empty_mask = preprocess_mask & clean_len.eq(0)
        _record_drop(empty_mask, "preprocess", "empty_after_clean")

        keep_mask = preprocess_mask & ~empty_mask
        df = df.loc[keep_mask].copy()
        df["review"] = clean_reviews.loc[keep_mask].values
        df["dup_count"] = dup_count.loc[keep_mask].astype(int).values
        df["n_reviews_total"] = int(keep_mask.sum())
        df = df.reset_index(drop=True)
        input_review_ids = audit_base[rid_col].astype(str).tolist()

        logging.info("[1] done: %d reviews (%.1fs)", len(df), time.time() - t0)

        # step 1.5: clause splitting
        t0 = time.time()
        logging.info("[1.5] splitting into clauses")
        clause_df = split_clauses(
            df, text_col="review",
            connectives=config.CLAUSE_CONNECTIVES,
            id_col=config.REVIEW_ID_COL,
        )
        fallback_clause_count = 0
        if "clause_source" in clause_df.columns:
            fallback_clause_count = int((clause_df["clause_source"] == "fallback_review").sum())
        logging.info("[1.5] fallback clauses: %d", fallback_clause_count)
        logging.info("[1.5] done: %d clauses (%.1fs)", len(clause_df), time.time() - t0)

        # step 1.6: ABSA
        t0 = time.time()
        logging.info("[1.6] ABSA (batch_size=%d)", config.ABSA_BATCH_SIZE)
        cache_root = Path(getattr(config, "OUTPUT_DIR", "output")) / "cache"
        absa_cache = cache_root / f"{stem_effective}_absa.csv.gz"
        absa_cache.parent.mkdir(parents=True, exist_ok=True)
        if resume and absa_cache.exists():
            logging.info("[1.6] using ABSA cache: %s", absa_cache)
            absa_df = pd.read_csv(absa_cache)
        else:
            raw_absa = classify_clauses(
                clause_df,
                model_name=config.ABSA_MODEL_NAME,
                batch_size=config.ABSA_BATCH_SIZE,
                device=config.DEVICE,
            )
            absa_df = pd.DataFrame(raw_absa, columns=["review_id", "clause", "polarity", "confidence"])
            try:
                absa_df.to_csv(absa_cache, index=False)
            except Exception:
                pass

        logging.info("[1.6] polarity counts: %s", absa_df["polarity"].value_counts().to_dict())
        if not absa_df.empty:
            pol_series = absa_df["polarity"].astype(str).str.lower().str.strip()
            conf_series = pd.to_numeric(absa_df["confidence"], errors="coerce")
            valid_pols = {"negative", "neutral", "positive"}
            missing_pol = pol_series.isna() | pol_series.eq("") | ~pol_series.isin(valid_pols)
            low_conf = conf_series.isna() | (conf_series < config.ABSA_CONFIDENCE_THRESHOLD)
            sentiment_fallback = missing_pol | low_conf
            absa_df["polarity"] = np.where(sentiment_fallback, "neutral", pol_series)
            absa_df["confidence"] = conf_series
            absa_df["sentiment_fallback"] = sentiment_fallback
            logging.info("[1.6] neutral fallback (missing/low-conf): %d", int(sentiment_fallback.sum()))
        else:
            absa_df["sentiment_fallback"] = []

        absa_rid_col = rid_col if rid_col in absa_df.columns else "review_id"
        absa_ids = set(absa_df[absa_rid_col].astype(str)) if not absa_df.empty else set()
        post_preprocess_ids = set(df[rid_col].astype(str))
        missing_after_absa = post_preprocess_ids - absa_ids
        _record_drop_by_ids(missing_after_absa, "absa_join", "join_missing_after_absa")

        def _write_audit_files(kept_ids: set[str]) -> None:
            missing_ids = set(input_review_ids) - kept_ids - set(drop_records.keys())
            _record_drop_by_ids(missing_ids, "exception", "other_exception")
            dropped_ids = [rid for rid in input_review_ids if rid in drop_records and rid not in kept_ids]
            dropped_rows = [drop_records[rid] for rid in dropped_ids]
            dropped_cols = ["review_id", "raw_len", "cleaned_len", "drop_stage", "drop_reason", "norm_hash", "raw_preview"]
            pd.DataFrame({rid_col: input_review_ids}).to_csv(
                audit_dir / f"{stem_effective}_input_review_ids.csv", index=False)
            pd.DataFrame({rid_col: [rid for rid in input_review_ids if rid in kept_ids]}).to_csv(
                audit_dir / f"{stem_effective}_kept_review_ids.csv", index=False)
            pd.DataFrame(dropped_rows, columns=dropped_cols).to_csv(
                audit_dir / f"{stem_effective}_dropped_reviews.csv", index=False)

        # accumulation buffers for polarity loop
        combined_clause_df_list: List[pd.DataFrame] = []
        combined_reps: Dict[int, list] = {}
        combined_kw: Dict[int, list] = {}
        combined_coords_list: List[np.ndarray] = []

        # step 1.7: per-polarity loop
        for pol in ("negative", "neutral", "positive"):
            acc_rows = sum(d.shape[0] for d in combined_clause_df_list)
            logging.info("[POL] accumulated clauses so far: %d", acc_rows)

            if pol == "neutral":
                sub_df = absa_df[absa_df["polarity"] == pol].reset_index(drop=True)
            else:
                conf_ok = absa_df["confidence"].fillna(-1) >= config.ABSA_CONFIDENCE_THRESHOLD
                sub_df = absa_df[(absa_df["polarity"] == pol) & conf_ok].reset_index(drop=True)
            logging.info("[POL:%s] %d/%d clauses selected (thr=%.2f)",
                         pol, len(sub_df), len(absa_df), config.ABSA_CONFIDENCE_THRESHOLD)
            if sub_df.empty:
                logging.info("[POL:%s] no clauses, skip", pol)
                continue

            texts = sub_df["clause"].tolist()

            # step 2: auto-tune UMAP/HDBSCAN params
            tuner_params = get_cluster_params(len(texts), dataset=f"{stem_effective}_{pol}")
            umap_p, hdbscan_p = tuner_params["umap"], tuner_params["hdbscan"]

            # step 3: embedding
            emb_t0 = time.time()
            rid_col = getattr(config, "REVIEW_ID_COL", "review_id")
            sub_df["clause_idx"] = sub_df.groupby(rid_col).cumcount()
            sub_df["clause_id"] = sub_df.apply(
                lambda row: f"{stem_effective}_{row[rid_col]}_{row['clause_idx']}", axis=1
            )
            embeddings = main_embedder.embed(
                texts=sub_df["clause"].tolist(),
                clause_ids=sub_df["clause_id"].tolist(),
                product_id=stem_effective,
            )
            if embeddings.size:
                last_embed_dim = int(embeddings.shape[1])
            logging.info("[POL:%s] embeddings: %s (%.1fs)", pol, embeddings.shape, time.time() - emb_t0)

            # step 4: UMAP reduction
            red_t0 = time.time()
            coords = reduce_embeddings(
                embeddings,
                n_components=umap_p["n_components"],
                n_neighbors=umap_p["n_neighbors"],
                min_dist=umap_p["min_dist"],
                metric=umap_p["metric"],
                random_state=umap_p["random_state"],
            )
            logging.info("[POL:%s] coords: %s (%.1fs)", pol, coords.shape, time.time() - red_t0)

            # step 5: HDBSCAN clustering
            clu_t0 = time.time()
            labels_raw, _ = cluster_embeddings(
                coords,
                min_cluster_size=hdbscan_p["min_cluster_size"],
                min_samples=hdbscan_p["min_samples"],
                metric=hdbscan_p["metric"],
                cluster_selection_epsilon=hdbscan_p["cluster_selection_epsilon"],
            )
            logging.info("[POL:%s] clustered: %d labels (%.1fs)", pol, len(labels_raw), time.time() - clu_t0)

            # step 6: evaluate and save diagnostics
            evaluate_clusters(
                labels_raw.copy(), coords, raw_embeddings=embeddings,
                output_dir=out_dir, timestamp=timestamp, tag=pol,
            )
            if coords.ndim == 2 and coords.shape[1] >= 2:
                combined_coords_list.append(coords[:, :2])

            # step 7: extract representative sentences
            reps = extract_representatives(
                texts=texts, embeddings=embeddings,
                labels=labels_raw, top_k=config.TOP_K_REPRESENTATIVES,
            )
            if alias_terms:
                try:
                    reps = {
                        cid: sorted(lst, key=lambda s: any(a in s for a in alias_terms), reverse=True)
                        for cid, lst in reps.items()
                    }
                except Exception:
                    pass

            # step 8: optional cluster merge
            if getattr(config, "ENABLE_CLUSTER_MERGE", False) and len(reps) >= 2:
                merge_map, merged_reps, _ = merge_similar_clusters(
                    reps,
                    threshold=getattr(config, "CLUSTER_MERGE_THRESHOLD", 0.90),
                )
                lbls_series = pd.to_numeric(pd.Series(labels_raw), errors="coerce").fillna(-1).astype(int)
                labels_raw = np.array([
                    merge_map.get(str(int(x)), int(x)) if int(x) >= 0 else -1
                    for x in lbls_series
                ], dtype=int)
                reps = merged_reps

            # step 9: keyword extraction
            keyword_model = getattr(getattr(config, "semantic", None), "model", None)
            kw = extract_keywords(reps, model_name=keyword_model)

            semantic_clause_embs = None
            if facets_obj is not None and semantic_helper_model is not None:
                try:
                    semantic_clause_embs = semantic_helper_model.encode(
                        texts,
                        batch_size=getattr(config, "BATCH_SIZE", 64),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                except Exception:
                    logging.exception("[FACETS] clause semantic embedding failed, using main embeddings")
                    semantic_clause_embs = None

            # step 10: refinement
            refined_df = None
            if refine_enabled and (facets_obj is not None):
                try:
                    work_df = sub_df.copy()
                    lbl_ser = pd.to_numeric(pd.Series(labels_raw), errors="coerce")
                    n_nan = int(lbl_ser.isna().sum())
                    if n_nan:
                        logging.warning("[REFINE] non-numeric labels: %d -> coercing to -1", n_nan)
                    labels_int = lbl_ser.fillna(-1).astype(int)
                    work_df["cluster_label"] = labels_int

                    clause_emb_source = semantic_clause_embs if semantic_clause_embs is not None else embeddings
                    clause_embs = _normalize_rows(np.asarray(clause_emb_source, dtype=np.float32))

                    logging.info("[REFINE] pol=%s | facets=%d | th=%.2f",
                                 pol, int(len(facets_obj)), float(refine_th.get("facet_threshold", 0.32)))

                    pre_cols = set(work_df.columns)
                    refined_df = refine_clusters(
                        work_df,
                        clause_embs=clause_embs,
                        polarity=pol,
                        facets=facets_obj,
                        top_k_facets=int(refine_th.get("top_k_facets", 2)),
                        facet_threshold=float(refine_th.get("facet_threshold", 0.32)),
                        hetero_sil_threshold=float(refine_th.get("hetero_sil_threshold", 0.18)),
                        min_cluster_size_for_split=int(refine_th.get("min_cluster_size_for_split", 40)),
                        max_local_k=int(refine_th.get("max_local_k", 4)),
                        other_label_value=-1,
                        stable_id_prefix=stable_id_prefix_map.get(pol, 0),
                    )

                    added = sorted(set(refined_df.columns) - pre_cols)
                    logging.info("[REFINE] added_cols=%s", added if added else [])

                    cov_col = (
                        "facet_top1" if "facet_top1" in refined_df.columns
                        else ("facet_bucket" if "facet_bucket" in refined_df.columns else None)
                    )
                    if cov_col is None:
                        raise RuntimeError("Refinement returned no facet columns (facet_top1/facet_bucket missing)")

                    cov_cnt = int(refined_df[cov_col].notna().sum())
                    tot_cnt = int(refined_df.shape[0])
                    logging.info("[REFINE] %s coverage: %d/%d (%.1f%%)",
                                 cov_col, cov_cnt, tot_cnt, 100.0 * (cov_cnt / (tot_cnt or 1)))

                    if cov_cnt == 0:
                        raise RuntimeError("Refinement produced zero facet assignments")

                except Exception:
                    logging.exception("[REFINE] failed, falling back to non-refined path")
                    refined_df = None
            else:
                logging.info("[REFINE] skipped (enabled=%s, facets_obj=%s)",
                             refine_enabled, type(facets_obj).__name__ if facets_obj is not None else None)

            # apply label offsets and accumulate
            base = base_map[pol]
            labels_off = _offset_labels(labels_raw, base)
            reps_off = _relabel_dict(reps, base)
            kw_off = _relabel_dict(kw, base)

            if refined_df is not None:
                clause_frame = refined_df.copy()
                clause_frame["cluster_label"] = labels_off
                if "polarity" not in clause_frame.columns:
                    clause_frame["polarity"] = pol
            else:
                clause_frame = sub_df.assign(cluster_label=labels_off, polarity=pol)

            clause_frame = apply_facet_routing(
                clause_frame,
                clause_embs=semantic_clause_embs if semantic_clause_embs is not None else embeddings,
                facets=facets_obj,
                top_k=int(refine_th.get("top_k_facets", 2)),
                threshold=float(refine_th.get("facet_threshold", 0.32)),
                category=category,
                facet_config=facet_config,
                sku=stem_effective,
            )

            combined_clause_df_list.append(clause_frame)
            combined_reps.update(reps_off)
            combined_kw.update(kw_off)

        if not combined_clause_df_list:
            missing_before_export = absa_ids
            _record_drop_by_ids(missing_before_export, "export_join", "join_missing_before_export")
            _write_audit_files(set())
            logging.info("[PIPELINE] no clauses passed threshold for any polarity, skipping: %s", stem_effective)
            logging.info("[PIPELINE] done: %s (%d/%d)", stem_effective, idx, total)
            continue

        # post-loop: facet column validation
        combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)
        has_f1 = "facet_top1" in combined_clause_df.columns
        has_fb = "facet_bucket" in combined_clause_df.columns

        if not has_f1:
            if has_fb:
                logging.warning("[CHECK] facet_top1 missing, copying from facet_bucket")
            else:
                logging.warning("[CHECK] no facet columns detected, facet_top1 will be null")
            combined_clause_df_list = [_ensure_facet_top1(df) for df in combined_clause_df_list]
            combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)
            has_f1 = "facet_top1" in combined_clause_df.columns
            has_fb = "facet_bucket" in combined_clause_df.columns

        logging.info("[CHECK] combined cols=%s", sorted(list(combined_clause_df.columns)))
        total_rows = len(combined_clause_df)
        if "facet_top1" in combined_clause_df.columns:
            unrouted_mask = combined_clause_df["facet_top1"].astype(str).str.strip().str.lower() == "unrouted"
            logging.info("[CHECK] unrouted facet clauses: %d", int(unrouted_mask.sum()))
        if has_f1:
            non_null = int(combined_clause_df["facet_top1"].notna().sum())
            non_blank = int(
                combined_clause_df["facet_top1"].dropna().astype(str).str.strip()
                .replace({"nan": "", "None": ""}).ne("").sum()
            )
            logging.info("[CHECK] facet_top1 non_null=%d non_blank=%d / total=%d", non_null, non_blank, total_rows)
        else:
            logging.warning("[CHECK] facet_top1 column not found")
        if has_fb:
            logging.info("[CHECK] facet_bucket nnz=%d / total=%d",
                         int(combined_clause_df["facet_bucket"].notna().sum()), total_rows)

        # debug sample CSV
        keep_cols = [c for c in ["review_id", "polarity", "cluster_label", "refined_cluster_id",
                                  "facet_top1", "confidence", "clause"] if c in combined_clause_df.columns]
        combined_clause_df.head(200)[keep_cols].to_csv(
            out_dir / f"debug_combined_head_{stem_effective}.csv", index=False, encoding="utf-8-sig")

        # stable IDs
        if getattr(config, "ENABLE_STABLE_IDS", True):
            combined_clause_df, _stable_map = assign_stable_ids(
                combined_clause_df, combined_reps,
                state_path=out_dir / "_stable_ids.json",
                prefer_col="refined_cluster_id",
            )

        final_ids = set(combined_clause_df[rid_col].astype(str))
        missing_before_export = absa_ids - final_ids
        _record_drop_by_ids(missing_before_export, "export_join", "join_missing_before_export")
        _write_audit_files(final_ids)

        save_clustered_clauses(
            clause_df=combined_clause_df,
            raw_df=df,
            keywords=combined_kw,
            output_path=out_dir / f"{stem_effective}_clauses_clustered_{timestamp}.xlsx"
        )
        report_path = out_dir / f"{stem_effective}_client_report_{timestamp}.xlsx"
        save_client_report(
            clause_df=combined_clause_df,
            raw_df=df,
            reps=combined_reps,
            kw=combined_kw,
            output_path=report_path,
        )
        logging.info("[EXPORT] client report saved: %s", report_path.name)

        save_clauses_summary_json(
            combined_clause_df,
            reps=combined_reps,
            kw=combined_kw,
            output_path=out_dir / f"{stem_effective}_clauses_summary_{timestamp}.json"
        )

        dim = last_embed_dim if last_embed_dim is not None else -1
        write_meta_json(out_dir / "meta.json", model_name=config.embed.model, embed_dim=dim)
        logging.info("[EXPORT] outputs saved")

        # visualization dashboard
        try:
            combined_2d = np.vstack(combined_coords_list) if combined_coords_list else None
            dashboard_path = generate_run_report(
                stem=stem_effective,
                timestamp=timestamp,
                clause_df=combined_clause_df,
                raw_df=df,
                reps=combined_reps,
                kw=combined_kw,
                out_dir=out_dir,
                embeddings_2d=combined_2d,
            )
            logging.info("[VIS] dashboard saved: %s", dashboard_path.name)
        except Exception:
            logging.exception("[VIS] dashboard generation failed (non-fatal)")

        logging.info("[PIPELINE] done: %s (%d/%d)", stem_effective, idx, total)


# --- CLI entry point ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Clause-level clustering pipeline runner")

    parser.add_argument("--files", nargs="*", type=Path, default=getattr(config, "INPUT_FILES", []),
                        help="Input Excel files")
    parser.add_argument("--output_dir", type=Path, default=Path(getattr(config, "OUTPUT_DIR", "output")),
                        help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Reuse caches if present (ABSA, embeddings)")
    parser.add_argument("--facets", type=str, default=None, help="Path to facets YAML (overrides config)")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to thresholds YAML (overrides config)")
    parser.add_argument("--mode", choices=["default", "community_filtered"], default="default",
                        help="community_filtered: summarize + relevance filter then run standard pipeline")
    parser.add_argument("--all", action="store_true",
                        help="Run all .xlsx under data/{review,community}")
    parser.add_argument("--product", default=None, help="Product key for community_filtered mode")
    parser.add_argument("--community_rules",
                        default=getattr(config, "COMMUNITY_RULES_PATH", "rules/community_rules.yml"))
    parser.add_argument("--rel_tau", type=float,
                        default=getattr(config, "COMMUNITY_REL_TAU", 0.40))
    parser.add_argument("--alias_tau", type=float,
                        default=getattr(config, "COMMUNITY_ALIAS_TAU", 0.40))
    parser.add_argument("--ban_mode", choices=["soft", "strict", "off"],
                        default=getattr(config, "COMMUNITY_BAN_MODE", "strict"))
    parser.add_argument("--save_filter_debug", action="store_true",
                        default=getattr(config, "COMMUNITY_SAVE_FILTER_DEBUG", False))
    parser.add_argument("--summary_max_sentences", type=int,
                        default=getattr(config, "COMMUNITY_SUMMARY_MAX_SENTENCES", 10))

    args = parser.parse_args()

    def _product_key(stem: str) -> str:
        return "".join(ch for ch in stem.lower() if ch.isalnum() or ch == "_")

    def _discover_dataset_files():
        base = Path(getattr(config, "DATA_DIR", Path("data")))
        review_dir = getattr(config, "REVIEW_DATA_DIR", base / "review")
        comm_dir = getattr(config, "COMMUNITY_DATA_DIR", base / "community")
        review_files = sorted(p for p in Path(review_dir).glob("*.xlsx") if not p.name.startswith("~$"))
        community_files = sorted(p for p in Path(comm_dir).glob("*.xlsx") if not p.name.startswith("~$"))
        return review_files, community_files

    def _pick_rules_for(product: str, facets_arg: str | None, thr_arg: str | None):
        """Use product-specific rules files if they exist, otherwise fall back to CLI args."""
        f_auto = Path(f"rules/facets_{product}.yml")
        t_auto = Path(f"rules/thresholds_{product}.yml")
        return (
            str(f_auto) if f_auto.exists() else facets_arg,
            str(t_auto) if t_auto.exists() else thr_arg,
        )

    def _build_relevance_embedder():
        base_embedder = get_embedder(config).embedder

        def _encode_norm(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.empty((0, 0), dtype=np.float32)
            arr = base_embedder.embed(list(texts))
            return _normalize_rows(np.asarray(arr, dtype=np.float32))

        class _Wrapper:
            def __init__(self):
                self._dim = 0

            def encode(self, texts, *_, **__):
                if not texts:
                    return np.empty((0, self._dim), dtype=np.float32)
                vecs = _encode_norm(list(texts))
                self._dim = vecs.shape[1]
                return vecs

        return _Wrapper()

    run_date = datetime.now().strftime("%Y%m%d")
    output_root = Path(args.output_dir)
    run_output_root = output_root / run_date

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = run_output_root / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"clause_pipeline_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.captureWarnings(True)
    logging.info("[RUN] output root: %s", run_output_root.as_posix())

    write_run_manifest(run_output_root / "run_manifest.json", config_obj=config)

    files_to_run: List[Path] = list(args.files)
    aliases: List[str] | None = None
    final_output_dir = run_output_root

    # community_filtered mode
    if args.mode == "community_filtered":
        try:
            from pipeline.summarizer_comm import summarize_row
            from pipeline.community_loader import load_posts
            from pipeline.relevance_filter import (
                build_alias_queries, build_facet_queries,
                dual_score_relevance, keep_mask_gated
            )
        except Exception as e:
            raise SystemExit(f"[community_filtered] required module missing: {e}")

        rules_path = Path(args.community_rules)
        if not rules_path.exists():
            raise SystemExit(f"[community_filtered] rules not found: {rules_path}")
        rules: Dict = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}

        product = (args.product or "").strip().lower()
        if not product:
            raise SystemExit("[community_filtered] --product is required")

        aliases = (rules.get("products", {}).get(product, {}) or {}).get("aliases", []) or []
        facets: Dict[str, List[str]] = (rules.get("facets", {}) or {})
        facet_terms_flat: List[str] = [w for lst in facets.values() for w in (lst or [])]
        ban_terms: List[str] = list(rules.get("ban_terms", []) or [])

        posts = [load_posts(p, product=product) for p in files_to_run]
        dfp = pd.concat(posts, ignore_index=True) if posts else pd.DataFrame()
        if dfp.empty:
            raise SystemExit("[community_filtered] no input posts found")

        rows = []
        for _, r in dfp.iterrows():
            sents = summarize_row(
                r.get("title", ""), r.get("body", ""), r.get("summary", ""),
                max_sentences=args.summary_max_sentences
            )
            if not sents:
                continue
            rows.append({
                "post_id": r["post_id"], "platform": r["platform"], "link": r["link"],
                "date": r.get("date"), "product": r["product"], "sentences": [str(s) for s in sents]
            })
        if not rows:
            raise SystemExit("[community_filtered] no summarized rows")

        embedder = _build_relevance_embedder()
        alias_q = build_alias_queries(aliases)
        facet_q = build_facet_queries(facet_terms_flat)

        flat_sents, owners = [], []
        for row in rows:
            for s in row["sentences"]:
                flat_sents.append(s)
                owners.append(row)
        if not flat_sents:
            raise SystemExit("[community_filtered] no summarized sentences")

        alias_sim, facet_sim, total = dual_score_relevance(flat_sents, alias_q, facet_q, embedder, facet_terms_flat)
        mask = keep_mask_gated(
            flat_sents, alias_sim, facet_sim, total,
            tau=args.rel_tau, alias_tau=args.alias_tau,
            lexical_aliases=aliases, ban_terms=ban_terms, ban_mode=args.ban_mode
        )

        kept, serial = [], 0
        for sent, ok, ow in zip(flat_sents, mask, owners):
            serial += 1
            if not ok:
                continue
            kept.append({
                "platform": ow["platform"], "product": ow["product"], "date": ow.get("date"),
                "review": sent, "review_id": f"{ow['post_id']}-s{serial}",
                "link": ow["link"], "source_type": "community"
            })
        kept_df = pd.DataFrame(kept)
        if kept_df.empty:
            raise SystemExit(
                f"[community_filtered] no sentences passed filter (tau={args.rel_tau}, alias_tau={args.alias_tau})"
            )

        out_root = run_output_root / f"{product}_community"
        out_root.mkdir(parents=True, exist_ok=True)
        tmp_input = out_root / "community_kept_input.xlsx"
        kept_df.to_excel(tmp_input, index=False)

        stats_path = out_root / "community_filter_stats.csv"
        base_df = pd.DataFrame({
            "review": flat_sents,
            "alias_sim": alias_sim,
            "facet_sim": facet_sim,
            "total": total,
            "kept": mask.astype(int),
            "lex_alias_hit": [int(any(a in s for a in aliases)) for s in flat_sents],
            "banned_hit": [int(any(b in s for b in ban_terms)) for s in flat_sents],
        })
        summ = {
            "total_sentences": len(base_df),
            "kept_sentences": int(base_df["kept"].sum()),
            "keep_rate": float(base_df["kept"].mean()) if len(base_df) else 0.0,
            "alias_hit_rate_all": float(base_df["lex_alias_hit"].mean()) if len(base_df) else 0.0,
            "alias_hit_rate_kept": (
                float(base_df.loc[base_df["kept"] == 1, "lex_alias_hit"].mean())
                if (base_df["kept"] == 1).any() else 0.0
            ),
            "banned_excluded": int(((base_df["banned_hit"] == 1) & (base_df["kept"] == 0)).sum()),
            "rel_tau": float(args.rel_tau),
            "alias_tau": float(args.alias_tau),
            "ban_mode": str(args.ban_mode),
        }
        pd.DataFrame([summ]).to_csv(stats_path, index=False)
        if args.save_filter_debug:
            base_df.to_csv(out_root / "community_filter_debug.csv", index=False, encoding="utf-8-sig")

        files_to_run = [tmp_input]
        final_output_dir = out_root

    # auto mode: run all xlsx under data/review and data/community
    if args.all or (not args.files and args.mode == "default" and args.product is None):
        review_files, community_files = _discover_dataset_files()
        if not review_files and not community_files:
            raise SystemExit("No .xlsx files found under data/review or data/community")

        for f in review_files:
            name = _product_key(f.stem)
            facets_path, thres_path = _pick_rules_for(name, args.facets, args.thresholds)
            logging.info("[AUTO] review run: %s", name)
            run_full_pipeline(
                [f], run_output_root,
                resume=args.resume,
                facets_path_override=facets_path,
                thresholds_path_override=thres_path,
                alias_terms=None,
                save_stem=None,
                audit_root=run_output_root / "audit",
            )

        if community_files:
            try:
                from pipeline.summarizer_comm import summarize_row
                from pipeline.community_loader import load_posts
                from pipeline.relevance_filter import (
                    build_alias_queries, build_facet_queries,
                    dual_score_relevance, keep_mask_gated,
                )
            except Exception as e:
                raise SystemExit(f"[auto] community module import failed: {e}")

            rules_path = Path(args.community_rules)
            if not rules_path.exists():
                raise SystemExit(f"[auto] community rules not found: {rules_path}")
            rules: Dict = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}

            for f in community_files:
                product = _product_key(f.stem)
                logging.info("[AUTO] community run: %s", product)

                aliases = (rules.get("products", {}).get(product, {}) or {}).get("aliases", []) or []
                facets: Dict[str, List[str]] = (rules.get("facets", {}) or {})
                facet_terms_flat: List[str] = [w for lst in facets.values() for w in (lst or [])]
                ban_terms: List[str] = list(rules.get("ban_terms", []) or [])

                dfp = load_posts(f, product=product)
                if dfp.empty:
                    logging.warning("[AUTO] empty file, skip: %s", f.name)
                    continue

                rows = []
                for _, r in dfp.iterrows():
                    sents = summarize_row(
                        r.get("title", ""), r.get("body", ""), r.get("summary", ""),
                        max_sentences=args.summary_max_sentences
                    )
                    if not sents:
                        continue
                    rows.append({
                        "post_id": r["post_id"], "platform": r["platform"], "link": r["link"],
                        "date": r.get("date"), "product": r["product"],
                        "sentences": [str(s) for s in sents]
                    })
                if not rows:
                    logging.warning("[AUTO] no summarized rows, skip: %s", f.name)
                    continue

                embedder = _build_relevance_embedder()
                alias_q = build_alias_queries(aliases)
                facet_q = build_facet_queries(facet_terms_flat)

                flat_sents, owners = [], []
                for row in rows:
                    for s in row["sentences"]:
                        flat_sents.append(s)
                        owners.append(row)
                if not flat_sents:
                    logging.warning("[AUTO] no summarized sentences, skip: %s", f.name)
                    continue

                alias_sim, facet_sim, total = dual_score_relevance(
                    flat_sents, alias_q, facet_q, embedder, facet_terms_flat
                )
                mask = keep_mask_gated(
                    flat_sents, alias_sim, facet_sim, total,
                    tau=args.rel_tau, alias_tau=args.alias_tau,
                    lexical_aliases=aliases, ban_terms=ban_terms, ban_mode=args.ban_mode
                )

                kept, serial = [], 0
                for sent, ok, ow in zip(flat_sents, mask, owners):
                    serial += 1
                    if not ok:
                        continue
                    kept.append({
                        "platform": ow["platform"], "product": ow["product"], "date": ow.get("date"),
                        "review": sent, "review_id": f"{ow['post_id']}-s{serial}",
                        "link": ow["link"], "source_type": "community"
                    })
                kept_df = pd.DataFrame(kept)
                if kept_df.empty:
                    logging.warning("[AUTO] no sentences passed filter, skip: %s", f.name)
                    continue

                out_root = run_output_root / f"{product}_community"
                out_root.mkdir(parents=True, exist_ok=True)
                tmp_input = out_root / "community_kept_input.xlsx"
                kept_df.to_excel(tmp_input, index=False)

                base_df = pd.DataFrame({
                    "review": flat_sents,
                    "alias_sim": alias_sim,
                    "facet_sim": facet_sim,
                    "total": total,
                    "kept": mask.astype(int),
                    "lex_alias_hit": [int(any(a in s for a in aliases)) for s in flat_sents],
                    "banned_hit": [int(any(b in s for b in ban_terms)) for s in flat_sents],
                })
                summ = {
                    "total_sentences": len(base_df),
                    "kept_sentences": int(base_df["kept"].sum()),
                    "keep_rate": float(base_df["kept"].mean()) if len(base_df) else 0.0,
                    "alias_hit_rate_all": float(base_df["lex_alias_hit"].mean()) if len(base_df) else 0.0,
                    "alias_hit_rate_kept": (
                        float(base_df.loc[base_df["kept"] == 1, "lex_alias_hit"].mean())
                        if (base_df["kept"] == 1).any() else 0.0
                    ),
                    "banned_excluded": int(((base_df["banned_hit"] == 1) & (base_df["kept"] == 0)).sum()),
                    "rel_tau": float(args.rel_tau),
                    "alias_tau": float(args.alias_tau),
                    "ban_mode": str(args.ban_mode),
                }
                pd.DataFrame([summ]).to_csv(out_root / "community_filter_stats.csv", index=False)
                if args.save_filter_debug:
                    base_df.to_csv(out_root / "community_filter_debug.csv", index=False, encoding="utf-8-sig")

                facets_path, thres_path = _pick_rules_for(product, args.facets, args.thresholds)
                run_full_pipeline(
                    [tmp_input], out_root,
                    resume=args.resume,
                    facets_path_override=facets_path,
                    thresholds_path_override=thres_path,
                    alias_terms=aliases,
                    save_stem=f"{product}_community",
                    audit_root=run_output_root / "audit",
                )

        logging.info("[AUTO] all runs complete")
        return

    # standard run
    run_full_pipeline(
        files_to_run,
        final_output_dir if args.mode == "community_filtered" else run_output_root,
        resume=args.resume,
        facets_path_override=args.facets,
        thresholds_path_override=args.thresholds,
        alias_terms=aliases if args.mode == "community_filtered" else None,
        audit_root=run_output_root / "audit",
    )


if __name__ == "__main__":
    main()

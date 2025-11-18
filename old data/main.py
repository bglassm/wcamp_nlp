# main.py — aligned with new refiner.Facet API (ASCII-only logs)

from __future__ import annotations

import argparse
import warnings
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time

import pandas as pd
import numpy as np

# quiet noisy libs
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
from pipeline.preprocess import preprocess_reviews
from pipeline.clause_splitter import split_clauses
from pipeline.absa import classify_clauses
from pipeline.embedder import embed_reviews
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
    load_facets_yml,
    refine_clusters,
    _normalize_rows,
    apply_facet_routing,
    Facet,
)
from utils.runmeta import write_run_manifest, write_meta_json

from sentence_transformers import SentenceTransformer

try:
    import yaml
except ImportError as _e:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from _e

# --- logging / warnings ---
logging.getLogger("pyabsa").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("kss").setLevel(logging.ERROR)
logging.getLogger("weasel").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

logger = logging.getLogger(__name__)

# --- helpers: label offset/merge ---

def _offset_labels(labels: np.ndarray, base: int) -> np.ndarray:
    """
    Shift HDBSCAN labels by a polarity base (neg=0, neu=1000, pos=2000).
    Noise -1 -> base+999.
    """
    out = []
    for l in labels:
        if isinstance(l, str):
            l = -1 if l.lower() == "other" else int(l)
        if int(l) == -1:
            out.append(base + 999)
        else:
            out.append(int(l) + base)
    return np.array(out, dtype=int)

def _relabel_dict(d: Dict[int, list], base: int) -> Dict[int, list]:
    """Shift keys in representatives/keywords dicts by base."""
    new_d: Dict[int, list] = {}
    for k, v in d.items():
        try:
            kk = int(k)
            new_d[kk + base] = v
        except Exception:
            new_d[base + 999] = v
    return new_d

def _ensure_facet_top1(df: pd.DataFrame, *, default: str | None = None) -> pd.DataFrame:
    """Ensure facet_top1 column exists (copy from facet_bucket or fill with default)."""
    if "facet_top1" in df.columns:
        return df
    if "facet_bucket" in df.columns:
        out = df.assign(facet_top1=df["facet_bucket"])
        mask_blank = out["facet_top1"].astype(str).str.strip() == ""
        out.loc[mask_blank, "facet_top1"] = None
        return out
    return df.assign(facet_top1=default)

# --- simple facets loader wrapper (no temp YAML, uses new API) ---

def _load_facets_simple(facets_path: Path, facet_embedder) -> tuple[list[Facet] | None, list[str], None]:
    with facets_path.open("r", encoding="utf-8") as f:
        y_head = f.read(200).replace("\n", " ")
    logger.info("[FACETS] reading %s | head=%s", str(facets_path), y_head[:120])

    facets = load_facets_yml(str(facets_path), facet_embedder)
    if not facets:
        raise ValueError(f"No facets loaded — check {facets_path}")

    names = [f.name for f in facets]
    logger.info("[FACETS] loaded=%d | head=%s", len(names), names[:5])
    return facets, names, None

# --- main pipeline ---

def run_full_pipeline(
    input_files: List[Path],
    output_dir: Path,
    *,
    resume: bool = False,
    facets_path_override: str | None = None,
    thresholds_path_override: str | None = None,
    alias_terms: List[str] | None = None,
    save_stem: str | None = None,
) -> None:
    total = len(input_files)
    logging.info(">> Starting full pipeline for %d files", total)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # polarity -> offset base
    base_map = {"negative": 0, "neutral": 1000, "positive": 2000}

    # refinement setup (optional)
    refine_enabled_cfg = getattr(config, "REFINEMENT_ENABLED", True)
    refine_enabled = refine_enabled_cfg
    facets_obj: list[Facet] | None = None
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

    try:
        facets_path = Path(
            facets_path_override or getattr(config, "REFINEMENT_FACETS_PATH", "rules/facets.yml")
        )
        thresholds_path = Path(
            thresholds_path_override or getattr(config, "REFINEMENT_THRESHOLDS_PATH", "rules/thresholds.yml")
        )
        logging.info("   Refinement config -> facets=%s | thresholds=%s", facets_path, thresholds_path)

        device_arg = getattr(config, "DEVICE", None)
        facet_embedder = SentenceTransformer(config.MODEL_NAME, device=device_arg)

        facets_obj, facet_names, _ = _load_facets_simple(facets_path, facet_embedder)

        with thresholds_path.open("r", encoding="utf-8") as f:
            config_payload = yaml.safe_load(f) or {}
            refine_th.update((config_payload.get("refinement") or {}))

        head = ", ".join(facet_names[:8]) + ("..." if len(facet_names) > 8 else "")
        logging.info("   Refinement assets loaded: %d buckets -> %s", len(facet_names), head)

    except Exception:
        logging.exception("   WARNING Refinement disabled (init failed). Check facets/thresholds YAML.")
        facets_obj = None
        refine_enabled = False

    if not refine_enabled_cfg:
        logging.info("   Refinement is disabled by config.")

    refine_enabled = refine_enabled_cfg and refine_enabled and (facets_obj is not None)

    # --- file loop ---
    for idx, file_path in enumerate(input_files, start=1):
        logging.info(">> [%d/%d] Processing %s", idx, total, file_path.name)
        stem_effective = (save_stem or file_path.stem)
        out_dir = output_dir / stem_effective
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load + preprocess
        t0 = time.time()
        logging.info("   1) Loading and preprocessing reviews...")

        df = load_reviews(file_path)
        df = preprocess_reviews(df)

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        if "review" not in df.columns:
            _cands = ["text", "body", "content", "contents", "summary"]
            _hit = next((c for c in _cands if c in df.columns), None)
            if _hit:
                df = df.rename(columns={_hit: "review"})
            else:
                raise SystemExit("[load] text column 'review' not found and no fallback candidate present.")

        rid_col = getattr(config, "REVIEW_ID_COL", "review_id")
        if rid_col not in df.columns:
            df = df.reset_index(drop=False).rename(columns={"index": rid_col})
        df[rid_col] = df[rid_col].astype(str)

        logging.info("      -> Loaded %d reviews (%.1fs)", len(df), time.time() - t0)

        # 1.5) Clause splitting
        t0 = time.time()
        logging.info("   1.5) Splitting into clauses...")
        clause_df = split_clauses(
            df, text_col="review",
            connectives=config.CLAUSE_CONNECTIVES,
            id_col=config.REVIEW_ID_COL,
        )
        logging.info("      -> split_clauses done (%d clauses, %.1fs)", len(clause_df), time.time() - t0)

        # 1.6) ABSA
        t0 = time.time()
        logging.info("   1.6) Running ABSA on clauses (batch_size=%d)...", config.ABSA_BATCH_SIZE)
        absa_cache = out_dir / "cache" / f"{stem_effective}_absa.csv.gz"
        absa_cache.parent.mkdir(parents=True, exist_ok=True)
        if resume and absa_cache.exists():
            logging.info("   1.6) Using ABSA cache -> %s", absa_cache)
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
        logging.info("      >> ABSA polarity counts: %s", absa_df["polarity"].value_counts().to_dict())

        # --- accumulators ---
        combined_clause_df_list: List[pd.DataFrame] = []
        combined_reps: Dict[int, list] = {}
        combined_kw: Dict[int, list] = {}

        # 1.7) polarity loop
        for pol in ("negative", "neutral", "positive"):
            acc_rows = sum(d.shape[0] for d in combined_clause_df_list)
            logging.info("   [ACC] accumulated clauses so far: %d", acc_rows)

            sub_df = absa_df[
                (absa_df["polarity"] == pol) &
                (absa_df["confidence"] >= config.ABSA_CONFIDENCE_THRESHOLD)
            ].reset_index(drop=True)
            logging.info("   >> [%s] %d/%d clauses selected (thr=%.2f)",
                         pol, len(sub_df), len(absa_df), config.ABSA_CONFIDENCE_THRESHOLD)
            if sub_df.empty:
                logging.info("      skip [%s] no clauses", pol)
                continue

            texts = sub_df["clause"].tolist()

            # 2) Auto tune UMAP/HDBSCAN
            tuner_params = get_cluster_params(len(texts), dataset=f"{stem_effective}_{pol}")
            umap_p, hdbscan_p = tuner_params["umap"], tuner_params["hdbscan"]

            # 3) Embedding
            emb_t0 = time.time()
            emb_cache = out_dir / "cache" / f"{stem_effective}_{pol}_{config.MODEL_NAME.replace('/', '_')}.npy"
            if resume and emb_cache.exists():
                try:
                    embeddings = np.load(emb_cache)
                    if embeddings.shape[0] != len(texts):
                        raise ValueError("shape mismatch — cache invalid")
                    logging.info("      -> [%s] Embeddings cache hit %s", pol, emb_cache.name)
                except Exception:
                    embeddings = embed_reviews(
                        texts, model_name=config.MODEL_NAME,
                        batch_size=config.BATCH_SIZE, device=config.DEVICE,
                    )
                    np.save(emb_cache, embeddings)
            else:
                embeddings = embed_reviews(
                    texts, model_name=config.MODEL_NAME,
                    batch_size=config.BATCH_SIZE, device=config.DEVICE,
                )
                try:
                    np.save(emb_cache, embeddings)
                except Exception:
                    pass
            logging.info("      -> [%s] Embeddings shape: %s (%.1fs)", pol, embeddings.shape, time.time() - emb_t0)

            # 4) UMAP reduction
            red_t0 = time.time()
            coords = reduce_embeddings(
                embeddings,
                n_components=umap_p["n_components"],
                n_neighbors=umap_p["n_neighbors"],
                min_dist=umap_p["min_dist"],
                metric=umap_p["metric"],
                random_state=umap_p["random_state"],
            )
            logging.info("      -> [%s] Reduced coords shape: %s (%.1fs)", pol, coords.shape, time.time() - red_t0)

            # 5) HDBSCAN clustering
            clu_t0 = time.time()
            labels_raw, _ = cluster_embeddings(
                coords,
                min_cluster_size=hdbscan_p["min_cluster_size"],
                min_samples=hdbscan_p["min_samples"],
                metric=hdbscan_p["metric"],
                cluster_selection_epsilon=hdbscan_p["cluster_selection_epsilon"],
            )
            logging.info("      -> [%s] Clustered (%d labels) (%.1fs)", pol, len(labels_raw), time.time() - clu_t0)

            # 6) diagnostics
            evaluate_clusters(
                labels_raw.copy(), coords, raw_embeddings=embeddings,
                output_dir=out_dir, timestamp=timestamp,
            )

            # 7) representatives
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

            # 8) optional merge
            if getattr(config, "ENABLE_CLUSTER_MERGE", False) and len(reps) >= 2:
                merge_map, merged_reps, _ = merge_similar_clusters(
                    reps,
                    model_name=config.MODEL_NAME,
                    threshold=getattr(config, "CLUSTER_MERGE_THRESHOLD", 0.90),
                )
                lbls_series = pd.to_numeric(pd.Series(labels_raw), errors="coerce").fillna(-1).astype(int)
                labels_raw = np.array([
                    merge_map.get(str(int(x)), int(x)) if int(x) >= 0 else -1
                    for x in lbls_series
                ], dtype=int)
                reps = merged_reps

            # 9) keywords
            kw = extract_keywords(reps, model_name=config.MODEL_NAME)

            # 10) refinement
            refined_df = None
            if refine_enabled and (facets_obj is not None):
                try:
                    work_df = sub_df.copy()
                    lbl_ser = pd.to_numeric(pd.Series(labels_raw), errors="coerce")
                    n_nan = int(lbl_ser.isna().sum())
                    if n_nan:
                        logging.warning("   [REFINE] non-numeric labels: %d -> coercing to -1", n_nan)
                    labels_int = lbl_ser.fillna(-1).astype(int)
                    work_df["cluster_label"] = labels_int

                    clause_embs = _normalize_rows(embeddings.astype(np.float32))

                    logging.info(
                        "   [REFINE] start pol=%s | facets=%d | th(facet)=%.2f",
                        pol, int(len(facets_obj)), float(refine_th.get("facet_threshold", 0.32))
                    )

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

                    post_cols = set(refined_df.columns)
                    added = sorted([c for c in post_cols - pre_cols])
                    logging.info("   [REFINE] added_cols=%s", added if added else [])

                    cov_col = "facet_top1" if "facet_top1" in refined_df.columns else (
                        "facet_bucket" if "facet_bucket" in refined_df.columns else None
                    )
                    if cov_col is None:
                        raise RuntimeError("Refinement returned no facet columns (facet_top1/facet_bucket missing)")

                    cov_cnt = int(refined_df[cov_col].notna().sum())
                    tot_cnt = int(refined_df.shape[0])
                    logging.info("   [REFINE] %s coverage: %d / %d (%.1f%%)",
                                cov_col, cov_cnt, tot_cnt, 100.0 * (cov_cnt / (tot_cnt or 1)))

                    if cov_cnt == 0:
                        raise RuntimeError("Refinement produced zero facet assignments")

                except Exception:
                    logging.exception("      [REFINE] failed; fallback to non-refined path")
                    refined_df = None
            else:
                logging.info(
                    "   [REFINE] skipped (enabled=%s, facets_obj=%s)",
                    refine_enabled, type(facets_obj).__name__ if facets_obj is not None else None
                )

            # offset and accumulate
            base = base_map[pol]
            labels_off = _offset_labels(labels_raw, base)
            reps_off = _relabel_dict(reps, base)
            kw_off = _relabel_dict(kw, base)

            if refined_df is not None:
                clause_frame = refined_df.copy()
                clause_frame["cluster_label"] = labels_off  # keep offset space
                if "polarity" not in clause_frame.columns:
                    clause_frame["polarity"] = pol
            else:
                clause_frame = sub_df.assign(cluster_label=labels_off, polarity=pol)

            # facet routing pass (preserves existing annotations)
            clause_frame = apply_facet_routing(
                clause_frame,
                facets_obj,
                clause_embs=embeddings,
                top_k=int(refine_th.get("top_k_facets", 2)),
                threshold=float(refine_th.get("facet_threshold", 0.32)),
            )

            combined_clause_df_list.append(clause_frame)
            combined_reps.update(reps_off)
            combined_kw.update(kw_off)

        # after concatenation, ensure facet columns present
        combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True) if combined_clause_df_list else pd.DataFrame()
        has_f1 = "facet_top1" in combined_clause_df.columns
        has_fb = "facet_bucket" in combined_clause_df.columns

        if not has_f1:
            if has_fb:
                logging.warning("   [WARN] facet_top1 missing but facet_bucket present — copying bucket values")
            else:
                logging.warning("   [WARN] No facet columns detected — leaving facet_top1 as nulls")
            combined_clause_df_list = [_ensure_facet_top1(df) for df in combined_clause_df_list]
            combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True) if combined_clause_df_list else pd.DataFrame()
            has_f1 = "facet_top1" in combined_clause_df.columns
            has_fb = "facet_bucket" in combined_clause_df.columns

        logging.info("   [CHECK] combined_clause_df cols=%s", sorted(list(combined_clause_df.columns)))
        total_rows = len(combined_clause_df)
        if has_f1:
            non_null = int(combined_clause_df["facet_top1"].notna().sum())
            non_blank = int(
                combined_clause_df["facet_top1"].dropna().astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").sum()
            )
            logging.info("   [CHECK] facet_top1 non_null=%d non_blank=%d / total=%d", non_null, non_blank, total_rows)
        else:
            logging.warning("   [CHECK] facet_top1 column not found")
        if has_fb:
            logging.info("   [CHECK] facet_bucket nnz=%d / total=%d", int(combined_clause_df["facet_bucket"].notna().sum()), total_rows)

        # save artifacts
        if combined_clause_df_list:
            combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)

            keep_cols = [c for c in ["review_id","polarity","cluster_label","refined_cluster_id","facet_top1","confidence","clause"] if c in combined_clause_df.columns]
            try:
                combined_clause_df.head(200)[keep_cols].to_csv(out_dir / f"debug_combined_head_{stem_effective}.csv", index=False, encoding="utf-8-sig")
            except Exception:
                pass

            if getattr(config, "ENABLE_STABLE_IDS", True):
                combined_clause_df, _stable_map = assign_stable_ids(
                    combined_clause_df, combined_reps,
                    state_path=out_dir / "_stable_ids.json",
                    prefer_col="refined_cluster_id",
                )

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
                output_path=report_path,
            )
            logging.info("      report saved -> %s", report_path.name)

            save_clauses_summary_json(
                combined_clause_df,
                reps=combined_reps,
                kw=combined_kw,
                output_path=out_dir / f"{stem_effective}_clauses_summary_{timestamp}.json"
            )

            dim = -1
            try:
                if 'embeddings' in locals() and hasattr(embeddings, 'shape'):
                    dim = int(embeddings.shape[1])
                else:
                    dim = SentenceTransformer(config.MODEL_NAME, device=getattr(config, "DEVICE", None))\
                            .get_sentence_embedding_dimension()
            except Exception:
                pass
            write_meta_json(out_dir / "meta.json", model_name=config.MODEL_NAME, embed_dim=dim)
            logging.info("      outputs saved")
        else:
            logging.info("   skip save: no clauses passed threshold for any polarity")

        logging.info("OK Completed %s (%d/%d)", stem_effective, idx, total)

# --- CLI ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Clause-level clustering pipeline runner")

    parser.add_argument("--files", nargs="*", type=Path, default=getattr(config, "INPUT_FILES", []),
                        help="List of input Excel files")
    parser.add_argument("--output_dir", type=Path, default=Path(getattr(config, "OUTPUT_DIR", "output")),
                        help="Directory to save outputs")
    parser.add_argument("--resume", action="store_true", help="Reuse caches if present (ABSA, embeddings)")
    parser.add_argument("--facets", type=str, default=None, help="Path to facets YAML (overrides config)")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to thresholds YAML (overrides config)")
    parser.add_argument("--mode", choices=["default", "community_filtered"], default="default",
                        help="community_filtered: summarize + relevance filter then run standard pipeline")
    parser.add_argument("--all", action="store_true",
                        help="Run all .xlsx under data/{review,community}")

    # community options
    parser.add_argument("--product", default=None, help="e.g., apple, paprika (for community_filtered)")
    parser.add_argument("--community_rules",
                        default=getattr(config, "COMMUNITY_RULES_PATH", "rules/community_rules.yml"))
    parser.add_argument("--rel_tau", type=float,
                        default=getattr(config, "COMMUNITY_REL_TAU", 0.40))
    parser.add_argument("--alias_tau", type=float,
                        default=getattr(config, "COMMUNITY_ALIAS_TAU", 0.40),
                        help="alias similarity threshold")
    parser.add_argument("--ban_mode", choices=["soft", "strict", "off"],
                        default=getattr(config, "COMMUNITY_BAN_MODE", "strict"),
                        help="forbidden terms mode")
    parser.add_argument("--save_filter_debug", action="store_true",
                        default=getattr(config, "COMMUNITY_SAVE_FILTER_DEBUG", False),
                        help="save filter score debug CSV")
    parser.add_argument("--summary_max_sentences", type=int,
                        default=getattr(config, "COMMUNITY_SUMMARY_MAX_SENTENCES", 10))

    args = parser.parse_args()

    # helpers
    def _product_key(stem: str) -> str:
        return "".join(ch for ch in stem.lower() if ch.isalnum() or ch == "_")

    def _discover_dataset_files():
        base = Path(getattr(config, "DATA_DIR", Path("data")))
        review_dir = getattr(config, "REVIEW_DATA_DIR", base / "review")
        comm_dir   = getattr(config, "COMMUNITY_DATA_DIR", base / "community")
        review_files = sorted(p for p in Path(review_dir).glob("*.xlsx") if not p.name.startswith("~$"))
        community_files = sorted(p for p in Path(comm_dir).glob("*.xlsx") if not p.name.startswith("~$"))
        return review_files, community_files

    def _pick_rules_for(product: str, facets_arg: str|None, thr_arg: str|None):
        f_auto = Path(f"rules/facets_{product}.yml")
        t_auto = Path(f"rules/thresholds_{product}.yml")
        facets_path = str(f_auto) if f_auto.exists() else facets_arg
        thr_path    = str(t_auto) if t_auto.exists() else thr_arg
        return facets_path, thr_path

    # logging setup
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = Path(getattr(config, "OUTPUT_DIR", "output")) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"clause_pipeline_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True
    )
    logging.captureWarnings(True)

    # manifest
    write_run_manifest(Path(getattr(config, "OUTPUT_DIR", "output")) / "run_manifest.json", config_obj=config)

    # inputs
    files_to_run: List[Path] = list(args.files)
    aliases: List[str] | None = None
    final_output_dir = args.output_dir

    # default: if --all or no files in default mode, run auto
    if args.all or (not args.files and args.mode == "default" and (args.product is None)):
        review_files, community_files = _discover_dataset_files()
        if not review_files and not community_files:
            raise SystemExit("No .xlsx files under data/review or data/community.")

        # reviews
        for f in review_files:
            name = _product_key(f.stem)
            facets_path, thres_path = _pick_rules_for(name, args.facets, args.thresholds)
            logging.info(">> [AUTO] Review run -> %s", name)
            run_full_pipeline(
                [f],
                args.output_dir,
                resume=args.resume,
                facets_path_override=facets_path,
                thresholds_path_override=thres_path,
                alias_terms=None,
                save_stem=None,
            )

        # communities (delegates to community pipeline; omitted here for brevity)
        logging.info("OK AUTO: review runs completed")
        return

    # standard pipeline
    run_full_pipeline(
        files_to_run,
        (final_output_dir if args.mode == "community_filtered" else args.output_dir),
        resume=args.resume,
        facets_path_override=args.facets,
        thresholds_path_override=args.thresholds,
        alias_terms=(aliases if args.mode == "community_filtered" else None),
    )

if __name__ == "__main__":
    main()

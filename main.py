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
import tempfile

import pandas as pd
import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")  # tqdm 전역 비활성화
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 허공 경고/프로그레스 억제
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# sentence-transformers tqdm 전역 OFF
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
from pipeline.embedder import get_embedder, CachingEmbedder # embed_reviews is now deprecated
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
from pipeline.refiner import load_facets_yml, refine_clusters, _normalize_rows, apply_facet_routing
from utils.runmeta import write_run_manifest, write_meta_json

# from sentence_transformers import SentenceTransformer # Not needed here anymore, moved to embedder.py

try:
    import yaml
except ImportError as _e:
    raise SystemExit("PyYAML가 필요합니다. 가상환경에서: pip install pyyaml") from _e

# --- 로그 & 경고 억제 ---
logging.getLogger("pyabsa").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("kss").setLevel(logging.ERROR)
logging.getLogger("weasel").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

logger = logging.getLogger(__name__)

# --- 유틸: 라벨 오프셋/병합 ---
def _offset_labels(labels: np.ndarray, base: int) -> np.ndarray:
    """
    HDBSCAN 라벨을 폴라리티별 base(neg=0, neu=1000, pos=2000)만큼 이동.
    -1(노이즈) → base+999 숫자화.
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
    """reps/keywords 딕셔너리의 키(클러스터 id)를 base만큼 이동."""
    new_d: Dict[int, list] = {}
    for k, v in d.items():
        try:
            kk = int(k)
            new_d[kk + base] = v
        except Exception:
            new_d[base + 999] = v
    return new_d

def _ensure_facet_top1(df: pd.DataFrame, *, default: str | None = None) -> pd.DataFrame:
    """Ensure a ``facet_top1`` column exists by copying from ``facet_bucket`` or filling."""
    if "facet_top1" in df.columns:
        return df
    if "facet_bucket" in df.columns:
        out = df.assign(facet_top1=df["facet_bucket"])
        mask_blank = out["facet_top1"].astype(str).str.strip() == ""
        out.loc[mask_blank, "facet_top1"] = None
        return out
    return df.assign(facet_top1=default)

# --- 유틸: 다양한 facets YAML 스키마를 표준 스키마로 정규화 ---
def _load_facets_forgiving(facets_path: Path, facet_embedder):
    """
    지원 스키마 예:
      1) {"facets": [{"name": "...", "keywords": [...]}, ...]}
      2) {"facets": {"sweetness": {"keywords":[...]}, ...}}
      3) [{"name": "...", "keywords": [...]}, ...]
      4) {"sweetness": {"keywords":[...]}, ...} (top-level dict-of-dicts)
    위를 list-of-facets 표준으로 변환 후 임시 YAML로 저장 → load_facets_yml 재사용.
    """
    with facets_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

        import os as _os
        _abs_path = _os.path.abspath(facets_path)
        _root_type = type(y).__name__
        _root_keys = list(y.keys())[:10] if isinstance(y, dict) else []
        logger.info("[FACETS] loaded: %s | root=%s keys=%s", _abs_path, _root_type, _root_keys)

    def to_list(obj):
        """
        Accept:
        - list schema: [{"id":..,"name":..,"desc":..}, ...] or {"buckets":[...]}
        - dict schema: {"facets": {"freshness": {"description": "...", "keywords":[...]}, ...}}
        """
        # case 1: already a list
        if isinstance(obj, list):
            return obj
        # case 2: dict wrapper
        if isinstance(obj, dict):
            # standard: buckets list
            if "buckets" in obj and isinstance(obj["buckets"], list):
                return obj["buckets"]
            # legacy: facets dict
            if "facets" in obj and isinstance(obj["facets"], dict) and obj["facets"]:
                buckets = []
                for name, node in obj["facets"].items():
                    node = node or {}
                    # desc: desc > description > keywords → fallback
                    desc = None
                    for k in ("desc", "description"):
                        v = node.get(k)
                        if v and str(v).strip():
                            desc = str(v).strip()
                            break
                    if not desc:
                        kws = node.get("keywords") or []
                        if isinstance(kws, (list, tuple)) and kws:
                            desc = f"{name} 관련 표현: " + " ".join(map(str, kws))
                        else:
                            desc = f"{name}이/가 좋다 나쁘다 만족 불만"
                    buckets.append({
                        "id": str(name).lower().replace(" ", ""),
                        "name": str(name),
                        "desc": desc
                    })
                return buckets
        # fallback
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
                logger.info("[FACETS] load_facets_yml OK via schema=%s", tag)
                return obj, tmp_path
            else:
                logger.warning("[FACETS] schema=%s yielded empty from load_facets_yml; will retry with fallback", tag)
                return None, tmp_path
        except Exception as e:
            logger.warning("[FACETS] schema=%s raised %s; will retry with fallback", tag, e)
            return None, tmp_path

    facets_obj, tmp_path = _try_build_facets({"buckets": facets_list}, tag="buckets")
    if not facets_obj:
        facets_obj, tmp_path2 = _try_build_facets({"facets": facets_list}, tag="facets")
        tmp_path = tmp_path2

    if not facets_obj:
        logger.error("[FACETS] load_facets_yml returned empty/None for %s (after both schemas)", _abs_path)
        raise ValueError(f"No facets loaded after normalization — check path/schema: {facets_path}")

    bucket_names = [d.get("name", f"F{i}") for i, d in enumerate(facets_list)]
    logger.info("[FACETS] bucket_names(head)=%s", bucket_names[:5])
    return facets_obj, bucket_names, tmp_path

# --- 메인 파이프라인 ---
def run_embed_pipeline(
    input_files: List[Path],
    output_dir: Path,
    *,
    save_stem: str,
) -> None:
    """
    임베딩만 생성하고 캐시 파일로 저장하는 파이프라인.
    """
    total = len(input_files)
    logging.info("▶ Starting embedding-only pipeline for %d files", total)
    
    embedder: CachingEmbedder = get_embedder(config)

    for i, input_file in enumerate(input_files):
        product_id = save_stem or input_file.stem
        logging.info("--- File %d/%d: %s (Product: %s) ---", i + 1, total, input_file.name, product_id)
        
        # 1. 데이터 로드 및 전처리
        df_reviews = load_reviews(input_file)
        df_reviews = preprocess_reviews(df_reviews)
        
        # 2. 절 분할 및 ABSA
        df_clauses = split_clauses(df_reviews)
        df_clauses = classify_clauses(df_clauses)
        
        # 3. 임베딩
        logging.info("--- 3. Embedding ---")
        
        # 3-1. 임베더 초기화 (캐싱 포함)
        # 3-2. 임베딩 실행 (캐싱 로직 내장)
        # clause_id 생성
        rid_col = getattr(config, "REVIEW_ID_COL", "review_id")
        df_clauses["clause_idx"] = df_clauses.groupby(rid_col).cumcount()
        
        # clause_id 생성: {product_id}_{review_id}_{clause_idx}
        df_clauses["clause_id"] = df_clauses.apply(
            lambda row: f"{product_id}_{row[rid_col]}_{row['clause_idx']}", axis=1
        )
        
        embedder.embed(
            texts=df_clauses["clause"].tolist(),
            clause_ids=df_clauses["clause_id"].tolist(),
            product_id=product_id,
        )
        
        # 임베딩 결과는 CachingEmbedder 내부에서 캐시 파일로 저장됨.
        # 여기서는 임베딩 벡터를 데이터프레임에 추가할 필요 없이 종료.
        logging.info("✅ Embedding complete and cached for product: %s", product_id)
        
    logging.info("▶ Embedding-only pipeline finished.")


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
    logging.info("▶ Starting full pipeline for %d files", total)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    main_embedder: CachingEmbedder = get_embedder(config)
    last_embed_dim: int | None = None

    # 폴라리티 → 오프셋 베이스
    base_map = {"negative": 0, "neutral": 1000, "positive": 2000}

    # --- Refinement 준비(옵션) ---
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
    try:
        facets_path = Path(
            facets_path_override or getattr(config, "REFINEMENT_FACETS_PATH", "rules/facets.yml")
        )
        thresholds_path = Path(
            thresholds_path_override or getattr(config, "REFINEMENT_THRESHOLDS_PATH", "rules/thresholds.yml")
        )
        logging.info("   Refinement config -> facets=%s | thresholds=%s", facets_path, thresholds_path)

        # Refinement/Facet embedder uses the same configuration as the main embedder
        # It must be a LocalEmbedder for SentenceTransformer compatibility in load_facets_yml
        # We assume the facet embedding is always local SBERT for compatibility with existing code.
        
        # Initialize the facet embedder (must be LocalEmbedder for SentenceTransformer)
        # If the main embedder is OpenAIEmbedder, we need to initialize a separate LocalEmbedder for facets.
        if config.embed.backend != "local":
            from pipeline.embedder import LocalEmbedder
            # Use the main config for local embedder init, but override backend to 'local'
            class TempConfig:
                class Embed:
                    backend = "local"
                    model = "jhgan/ko-sbert-sts" # Default SBERT model
                    device = config.embed.device
                    batch_size = config.embed.batch_size
                    api_base = config.embed.api_base
                    max_retries = config.embed.max_retries
                    timeout_sec = config.embed.timeout_sec
                    cache_dir = config.embed.cache_dir
                embed = Embed()

            # Initialize a temporary LocalEmbedder to get the SentenceTransformer model
            facet_embedder = LocalEmbedder(TempConfig()).model
            logger.info("✅ Initialized separate LocalEmbedder for facet embedding.")
        else:
            # If main embedder is local, initialize it once and reuse the model
            # We need to call get_embedder to ensure the model is loaded and cached
            facet_embedder = main_embedder.embedder.model
            logger.info("✅ Reusing main LocalEmbedder model for facet embedding.")
            
        # Now facet_embedder is a SentenceTransformer instance, as required by _load_facets_forgiving
        # device_arg = getattr(config, "DEVICE", None)
        # facet_embedder = SentenceTransformer(config.MODEL_NAME, device=device_arg) # Old code removed
        
        # --- Refinement/Facet embedder initialization end ---

        facets_obj, facet_names, norm_tmp_path = _load_facets_forgiving(facets_path, facet_embedder)

        with thresholds_path.open("r", encoding="utf-8") as f:
            config_payload = yaml.safe_load(f) or {}
            refine_th.update((config_payload.get("refinement") or {}))

        head = ", ".join(facet_names[:8]) + ("..." if len(facet_names) > 8 else "")
        logging.info("   Refinement assets loaded: %d buckets -> %s", len(facet_names), head)

    except Exception:
        logging.exception("   WARNING Refinement disabled (init failed). Check facets/thresholds YAML & schema.")
        facets_obj = None
        refine_enabled = False
    finally:
        if norm_tmp_path and norm_tmp_path.exists():
            try:
                norm_tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not refine_enabled_cfg:
        logging.info("   Refinement is disabled by config.")

    refine_enabled = refine_enabled_cfg and refine_enabled and (facets_obj is not None)

    # --- 파일 루프 ---
    for idx, file_path in enumerate(input_files, start=1):
        logging.info("🔄 [%d/%d] Processing %s", idx, total, file_path.name)
        stem_effective = (save_stem or file_path.stem)
        out_dir = output_dir / stem_effective
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load + preprocess
        t0 = time.time()
        logging.info("   1) Loading & preprocessing reviews…")

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

        logging.info("      → Loaded %d reviews (%.1fs)", len(df), time.time() - t0)

        # 1.5) Clause splitting
        t0 = time.time()
        logging.info("   1.5) Splitting into clauses…")
        clause_df = split_clauses(
            df, text_col="review",
            connectives=config.CLAUSE_CONNECTIVES,
            id_col=config.REVIEW_ID_COL,
        )
        logging.info("      → split_clauses done (%d clauses, %.1fs)", len(clause_df), time.time() - t0)

        # 1.6) ABSA
        t0 = time.time()
        logging.info("   1.6) Running ABSA on clauses (batch_size=%d)…", config.ABSA_BATCH_SIZE)
        absa_cache = out_dir / "cache" / f"{stem_effective}_absa.csv.gz"
        absa_cache.parent.mkdir(parents=True, exist_ok=True)
        if resume and absa_cache.exists():
            logging.info(" 1.6) Using ABSA cache → %s", absa_cache)
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
        logging.info("      ▶ [DEBUG] ABSA polarity counts: %s", absa_df["polarity"].value_counts().to_dict())

        # --- 누적 버퍼 ---
        combined_clause_df_list: List[pd.DataFrame] = []
        combined_reps: Dict[int, list] = {}
        combined_kw: Dict[int, list] = {}

        # 1.7) 폴라리티 루프
        for pol in ("negative", "neutral", "positive"):
            acc_rows = sum(d.shape[0] for d in combined_clause_df_list)
            logging.info("   [ACC] accumulated clauses so far: %d", acc_rows)

            sub_df = absa_df[
                (absa_df["polarity"] == pol) &
                (absa_df["confidence"] >= config.ABSA_CONFIDENCE_THRESHOLD)
            ].reset_index(drop=True)
            logging.info("   ▶ [%s] %d/%d clauses selected (thr=%.2f)",
                         pol, len(sub_df), len(absa_df), config.ABSA_CONFIDENCE_THRESHOLD)
            if sub_df.empty:
                logging.info("      ⏭️ [%s] no clauses, skip", pol)
                continue

            texts = sub_df["clause"].tolist()

            # 2) Auto‐tune UMAP/HDBSCAN params
            tuner_params = get_cluster_params(len(texts), dataset=f"{stem_effective}_{pol}")
            umap_p, hdbscan_p = tuner_params["umap"], tuner_params["hdbscan"]

            # 3) Embedding
            emb_t0 = time.time()
            
            # 3-2. 임베딩 실행 (캐싱 로직 내장)
            # clause_id 생성
            rid_col = getattr(config, "REVIEW_ID_COL", "review_id")
            sub_df["clause_idx"] = sub_df.groupby(rid_col).cumcount()
            
            # clause_id 생성: {product_id}_{review_id}_{clause_idx}
            sub_df["clause_id"] = sub_df.apply(
                lambda row: f"{stem_effective}_{row[rid_col]}_{row['clause_idx']}", axis=1
            )
            
            # 기존 numpy cache 로직은 제거하고 CachingEmbedder를 사용
            embeddings = main_embedder.embed(
                texts=sub_df["clause"].tolist(),
                clause_ids=sub_df["clause_id"].tolist(),
                product_id=stem_effective,
            )
            if embeddings.size:
                last_embed_dim = int(embeddings.shape[1])
            
            # embeddings는 numpy array로 반환됨
            logging.info("      → [%s] Embeddings shape: %s (%.1fs)", pol, embeddings.shape, time.time() - emb_t0)

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
            logging.info("      → [%s] Reduced coords shape: %s (%.1fs)", pol, coords.shape, time.time() - red_t0)

            # 5) HDBSCAN clustering
            clu_t0 = time.time()
            labels_raw, _ = cluster_embeddings(
                coords,
                min_cluster_size=hdbscan_p["min_cluster_size"],
                min_samples=hdbscan_p["min_samples"],
                metric=hdbscan_p["metric"],
                cluster_selection_epsilon=hdbscan_p["cluster_selection_epsilon"],
            )
            logging.info("      → [%s] Clustered (%d labels) (%.1fs)", pol, len(labels_raw), time.time() - clu_t0)

            # 6) 진단 저장
            evaluate_clusters(
                labels_raw.copy(), coords, raw_embeddings=embeddings,
                output_dir=out_dir, timestamp=timestamp,
            )

            # 7) 대표 문장
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

            # 8) 병합(옵션)
            if getattr(config, "ENABLE_CLUSTER_MERGE", False) and len(reps) >= 2:
                merge_map, merged_reps, _ = merge_similar_clusters(
                    reps,
                    model_name=config.embed.model,
                    threshold=getattr(config, "CLUSTER_MERGE_THRESHOLD", 0.90),
                )
                lbls_series = pd.to_numeric(pd.Series(labels_raw), errors="coerce").fillna(-1).astype(int)
                labels_raw = np.array([
                    merge_map.get(str(int(x)), int(x)) if int(x) >= 0 else -1
                    for x in lbls_series
                ], dtype=int)
                reps = merged_reps

            # 9) 키워드
            kw = extract_keywords(reps, model_name=config.embed.model)

            # 10) Refinement
            refined_df = None
            if refine_enabled and (facets_obj is not None):
                try:
                    # before snapshot
                    work_df = sub_df.copy()
                    lbl_ser = pd.to_numeric(pd.Series(labels_raw), errors="coerce")
                    n_nan = int(lbl_ser.isna().sum())
                    if n_nan:
                        logging.warning("   [REFINE] non-numeric labels: %d → coercing to -1", n_nan)
                    labels_int = lbl_ser.fillna(-1).astype(int)
                    work_df["cluster_label"] = labels_int

                    # normalize embeddings (cosine)
                    clause_embs = _normalize_rows(embeddings.astype(np.float32))

                    # run refinement (NOTE: other_label_value MUST be int -1)
                    logging.info("   [REFINE] start pol=%s | facets=%d | th(facet)=%.2f",
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
                        other_label_value=-1,  # ← 정수 -1로 고정 (중요)
                        stable_id_prefix=stable_id_prefix_map.get(pol, 0),
                    )

                    post_cols = set(refined_df.columns)
                    added = sorted([c for c in post_cols - pre_cols])
                    logging.info("   [REFINE] added_cols=%s", added if added else [])

                    # coverage check
                    cov_col = "facet_top1" if "facet_top1" in refined_df.columns else (
                            "facet_bucket" if "facet_bucket" in refined_df.columns else None)
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
                logging.info("   [REFINE] skipped (enabled=%s, facets_obj=%s)", refine_enabled, type(facets_obj).__name__ if facets_obj is not None else None)


            # --- 오프셋 적용 후 통합 ---
            base = base_map[pol]
            labels_off = _offset_labels(labels_raw, base)
            reps_off = _relabel_dict(reps, base)
            kw_off = _relabel_dict(kw, base)

            if refined_df is not None:
                clause_frame = refined_df.copy()
                clause_frame["cluster_label"] = labels_off  # offset 적용
                if "polarity" not in clause_frame.columns:
                    clause_frame["polarity"] = pol
            else:
                clause_frame = sub_df.assign(cluster_label=labels_off, polarity=pol)

            clause_frame = apply_facet_routing(
                clause_frame,
                clause_embs=embeddings,
                facets=facets_obj,
                top_k=int(refine_th.get("top_k_facets", 2)),
                threshold=float(refine_th.get("facet_threshold", 0.32)),
            )

            combined_clause_df_list.append(clause_frame)

            combined_reps.update(reps_off)
            combined_kw.update(kw_off)

        if not combined_clause_df_list:
            logging.info("   ⏭️ No clauses passed threshold for any polarity — nothing to save.")
            logging.info("✅ Completed %s (%d/%d)\n", stem_effective, idx, total)
            continue

        # -- after concatenation, fail-fast if facet columns missing ------------------
        combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)
        has_f1 = "facet_top1" in combined_clause_df.columns
        has_fb = "facet_bucket" in combined_clause_df.columns

        if not has_f1:
            if has_fb:
                logging.warning(
                    "   [WARN] facet_top1 missing but facet_bucket present — copying bucket values"
                )
            else:
                logging.warning(
                    "   [WARN] No facet columns detected — leaving facet_top1 as nulls"
                )
            combined_clause_df_list = [_ensure_facet_top1(df) for df in combined_clause_df_list]
            combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)
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

        # 작은 샘플 CSV (보고서 전에 눈으로 바로 봄)
        keep_cols = [c for c in ["review_id","polarity","cluster_label","refined_cluster_id","facet_top1","confidence","clause"] if c in combined_clause_df.columns]
        combined_clause_df.head(200)[keep_cols].to_csv(out_dir / f"debug_combined_head_{stem_effective}.csv", index=False, encoding="utf-8-sig")

        # Stable IDs
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
        logging.info("      💾 client report saved → %s", report_path.name)

        save_clauses_summary_json(
            combined_clause_df,
            reps=combined_reps,
            kw=combined_kw,
            output_path=out_dir / f"{stem_effective}_clauses_summary_{timestamp}.json"
        )

        dim = last_embed_dim if last_embed_dim is not None else -1
        write_meta_json(out_dir / "meta.json", model_name=config.embed.model, embed_dim=dim)
        logging.info("      💾 merged outputs saved")

        logging.info("✅ Completed %s (%d/%d)\n", stem_effective, idx, total)

# --- CLI 진입점 ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Clause-level clustering pipeline runner")

    # 공통 파이프라인 인자
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

    # 커뮤니티 전처리 인자
    parser.add_argument("--product", default=None, help="e.g., apple, paprika (for community_filtered)")
    parser.add_argument("--community_rules",
                        default=getattr(config, "COMMUNITY_RULES_PATH", "rules/community_rules.yml"))
    parser.add_argument("--rel_tau", type=float,
                        default=getattr(config, "COMMUNITY_REL_TAU", 0.40))
    parser.add_argument("--alias_tau", type=float,
                        default=getattr(config, "COMMUNITY_ALIAS_TAU", 0.40),
                        help="별칭 임베딩 유사도 임계값")
    parser.add_argument("--ban_mode", choices=["soft", "strict", "off"],
                        default=getattr(config, "COMMUNITY_BAN_MODE", "strict"),
                        help="금칙어 적용 강도")
    parser.add_argument("--save_filter_debug", action="store_true",
                        default=getattr(config, "COMMUNITY_SAVE_FILTER_DEBUG", False),
                        help="필터링 스코어 디버그 CSV 저장")
    parser.add_argument("--summary_max_sentences", type=int,
                        default=getattr(config, "COMMUNITY_SUMMARY_MAX_SENTENCES", 10))

    args = parser.parse_args()

    # 규칙/경로 헬퍼
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
        """
        rules/facets_{product}.yml, rules/thresholds_{product}.yml 이 있으면 우선 적용.
        없으면 CLI 인자 → 없으면 글로벌 기본으로 폴백.
        """
        f_auto = Path(f"rules/facets_{product}.yml")
        t_auto = Path(f"rules/thresholds_{product}.yml")
        facets_path = str(f_auto) if f_auto.exists() else facets_arg
        thr_path    = str(t_auto) if t_auto.exists() else thr_arg
        return facets_path, thr_path

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

    # 로깅
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

    # 매니페스트
    write_run_manifest(Path(getattr(config, "OUTPUT_DIR", "output")) / "run_manifest.json", config_obj=config)

    # 입력 준비
    files_to_run: List[Path] = list(args.files)
    aliases: List[str] | None = None
    final_output_dir = args.output_dir

    # community_filtered 모드(요약→관련성→임시 입력)
    if args.mode == "community_filtered":
        try:
            from pipeline.summarizer_comm import summarize_row
            from pipeline.community_loader import load_posts
            from pipeline.relevance_filter import (
                build_alias_queries, build_facet_queries,
                dual_score_relevance, keep_mask_gated
            )
        except Exception as e:
            raise SystemExit(f"[community_filtered] 필요한 모듈이 없습니다: {e}")

        rules_path = Path(args.community_rules)
        if not rules_path.exists():
            raise SystemExit(f"[community_filtered] rules not found: {rules_path}")
        rules: Dict = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}

        product = (args.product or "").strip().lower()
        if not product:
            raise SystemExit("[community_filtered] --product 가 필요합니다. 예: --product apple")

        aliases = (rules.get("products", {}).get(product, {}) or {}).get("aliases", []) or []
        facets: Dict[str, List[str]] = (rules.get("facets", {}) or {})
        facet_terms_flat: List[str] = [w for lst in facets.values() for w in (lst or [])]
        ban_terms: List[str] = list(rules.get("ban_terms", []) or [])

        posts = [load_posts(p, product=product) for p in files_to_run]
        dfp = pd.concat(posts, ignore_index=True) if posts else pd.DataFrame()
        if dfp.empty:
            raise SystemExit("[community_filtered] 입력 게시글이 비어 있습니다.")

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
            raise SystemExit("[community_filtered] 요약 결과가 없습니다.")

        embedder = _build_relevance_embedder()
        alias_q = build_alias_queries(aliases)
        facet_q = build_facet_queries(facet_terms_flat)

        flat_sents, owners = [], []
        for row in rows:
            for s in row["sentences"]:
                flat_sents.append(s)
                owners.append(row)
        if not flat_sents:
            raise SystemExit("[community_filtered] 요약 문장이 없습니다.")

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
                "platform": ow["platform"],
                "product": ow["product"],
                "date": ow.get("date"),
                "review": sent,
                "review_id": f"{ow['post_id']}-s{serial}",
                "link": ow["link"],
                "source_type": "community"
            })
        kept_df = pd.DataFrame(kept)
        if kept_df.empty:
            raise SystemExit(f"[community_filtered] 관련성 임계 통과 문장이 없습니다. (tau={args.rel_tau}, alias_tau={args.alias_tau})")

        out_root = Path(args.output_dir) / f"{product}_community"
        out_root.mkdir(parents=True, exist_ok=True)
        tmp_input = out_root / "community_kept_input.xlsx"
        kept_df.to_excel(tmp_input, index=False)

        stats_path = out_root / "community_filter_stats.csv"
        base_df = pd.DataFrame({
            "review":  flat_sents,
            "alias_sim": alias_sim,
            "facet_sim": facet_sim,
            "total":     total,
            "kept":      mask.astype(int),
            "lex_alias_hit": [int(any(a in s for a in aliases)) for s in flat_sents],
            "banned_hit":    [int(any(b in s for b in ban_terms)) for s in flat_sents],
        })
        summ = {
            "total_sentences": len(base_df),
            "kept_sentences":  int(base_df["kept"].sum()),
            "keep_rate":       float(base_df["kept"].mean()) if len(base_df) else 0.0,
            "alias_hit_rate_all": float(base_df["lex_alias_hit"].mean()) if len(base_df) else 0.0,
            "alias_hit_rate_kept": float(base_df.loc[base_df["kept"]==1, "lex_alias_hit"].mean())
                                   if (base_df["kept"]==1).any() else 0.0,
            "banned_excluded": int(((base_df["banned_hit"]==1) & (base_df["kept"]==0)).sum()),
            "rel_tau": float(args.rel_tau),
            "alias_tau": float(args.alias_tau),
            "ban_mode": str(args.ban_mode),
        }
        pd.DataFrame([summ]).to_csv(stats_path, index=False)
        if args.save_filter_debug:
            base_df.to_csv(out_root / "community_filter_debug.csv", index=False, encoding="utf-8-sig")

        files_to_run = [tmp_input]
        final_output_dir = out_root

    # 자동 실행: 인자를 안 주면 리뷰+커뮤니티 전량 실행
    if args.all or (not args.files and args.mode == "default" and (args.product is None)):
        review_files, community_files = _discover_dataset_files()
        if not review_files and not community_files:
            raise SystemExit("data/review, data/community 하위에 .xlsx가 없습니다.")

        # 리뷰 전량
        for f in review_files:
            name = _product_key(f.stem)
            facets_path, thres_path = _pick_rules_for(name, args.facets, args.thresholds)
            logging.info(f"▶ [AUTO] Review run → {name}")
            run_full_pipeline(
                [f],
                args.output_dir,
                resume=args.resume,
                facets_path_override=facets_path,
                thresholds_path_override=thres_path,
                alias_terms=None,
                save_stem=None,
            )

        # 커뮤니티 전량
        if community_files:
            try:
                from pipeline.summarizer_comm import summarize_row
                from pipeline.community_loader import load_posts
                from pipeline.relevance_filter import (
                    build_alias_queries, build_facet_queries,
                    dual_score_relevance, keep_mask_gated,
                )
            except Exception as e:
                raise SystemExit(f"[auto] 커뮤니티 모듈 임포트 실패: {e}")

            rules_path = Path(args.community_rules)
            if not rules_path.exists():
                raise SystemExit(f"[auto] community rules not found: {rules_path}")
            rules: Dict = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}

            for f in community_files:
                product = _product_key(f.stem)
                logging.info(f"▶ [AUTO] Community run → {product}")

                aliases = (rules.get("products", {}).get(product, {}) or {}).get("aliases", []) or []
                facets: Dict[str, List[str]] = (rules.get("facets", {}) or {})
                facet_terms_flat: List[str] = [w for lst in facets.values() for w in (lst or [])]
                ban_terms: List[str] = list(rules.get("ban_terms", []) or [])

                dfp = load_posts(f, product=product)
                if dfp.empty:
                    logging.warning(f"[auto] 빈 파일 → {f.name}, skip")
                    continue

                rows = []
                for _, r in dfp.iterrows():
                    sents = summarize_row(r.get("title",""), r.get("body",""), r.get("summary",""),
                                          max_sentences=args.summary_max_sentences)
                    if not sents:
                        continue
                    rows.append({
                        "post_id": r["post_id"],
                        "platform": r["platform"],
                        "link": r["link"],
                        "date": r.get("date"),
                        "product": r["product"],
                        "sentences": [str(s) for s in sents]
                    })
                if not rows:
                    logging.warning(f"[auto] 요약 결과 없음 → {f.name}, skip")
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
                    logging.warning(f"[auto] 요약 문장 없음 → {f.name}, skip")
                    continue

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
                        "platform": ow["platform"],
                        "product": ow["product"],
                        "date": ow.get("date"),
                        "review": sent,
                        "review_id": f"{ow['post_id']}-s{serial}",
                        "link": ow["link"],
                        "source_type": "community"
                    })
                kept_df = pd.DataFrame(kept)
                if kept_df.empty:
                    logging.warning(f"[auto] 필터 통과 문장 없음 → {f.name}, skip")
                    continue

                out_root = Path(args.output_dir) / f"{product}_community"
                out_root.mkdir(parents=True, exist_ok=True)
                tmp_input = out_root / "community_kept_input.xlsx"
                kept_df.to_excel(tmp_input, index=False)

                stats_path = out_root / "community_filter_stats.csv"
                base_df = pd.DataFrame({
                    "review":  flat_sents,
                    "alias_sim": alias_sim,
                    "facet_sim": facet_sim,
                    "total":     total,
                    "kept":      mask.astype(int),
                    "lex_alias_hit": [int(any(a in s for a in aliases)) for s in flat_sents],
                    "banned_hit":    [int(any(b in s for b in ban_terms)) for s in flat_sents],
                })
                summ = {
                    "total_sentences": len(base_df),
                    "kept_sentences":  int(base_df["kept"].sum()),
                    "keep_rate":       float(base_df["kept"].mean()) if len(base_df) else 0.0,
                    "alias_hit_rate_all": float(base_df["lex_alias_hit"].mean()) if len(base_df) else 0.0,
                    "alias_hit_rate_kept": float(base_df.loc[base_df["kept"]==1, "lex_alias_hit"].mean())
                                           if (base_df["kept"]==1).any() else 0.0,
                    "banned_excluded": int(((base_df["banned_hit"]==1) & (base_df["kept"]==0)).sum()),
                    "rel_tau": float(args.rel_tau),
                    "alias_tau": float(args.alias_tau),
                    "ban_mode": str(args.ban_mode),
                }
                pd.DataFrame([summ]).to_csv(stats_path, index=False)
                if args.save_filter_debug:
                    base_df.to_csv(out_root / "community_filter_debug.csv", index=False, encoding="utf-8-sig")

                facets_path, thres_path = _pick_rules_for(product, args.facets, args.thresholds)

                run_full_pipeline(
                    [tmp_input],
                    out_root,
                    resume=args.resume,
                    facets_path_override=facets_path,
                    thresholds_path_override=thres_path,
                    alias_terms=aliases,
                    save_stem=f"{product}_community",
                )

        logging.info("🎉 AUTO: review + community 전체 실행 완료")
        return

    # 표준 절-단위 파이프라인 실행
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

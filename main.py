from __future__ import annotations

import argparse
import warnings
import logging
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time

import pandas as pd
import numpy as np

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
from pipeline.refiner import load_facets_yml, refine_clusters
from pipeline.idmap import assign_stable_ids
from utils.runmeta import write_run_manifest, write_meta_json

# ────────────────────────────────────────────────────────────────────────────
# 로깅 & 경고 억제
# ────────────────────────────────────────────────────────────────────────────
logging.getLogger("pyabsa").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("kss").setLevel(logging.ERROR)
logging.getLogger("weasel").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# ────────────────────────────────────────────────────────────────────────────
# 유틸: 라벨 오프셋/병합
# ────────────────────────────────────────────────────────────────────────────
def _offset_labels(labels: np.ndarray, base: int) -> np.ndarray:
    """
    HDBSCAN 라벨을 폴라리티별 base(neg=0, neu=1000, pos=2000)만큼 이동.
    -1(노이즈) → base+999 로 숫자화.
    """
    out = []
    for l in labels:
        # 문자열 'other' 같은 케이스 방지
        if isinstance(l, str):
            l = -1 if l.lower() == "other" else int(l)
        if int(l) == -1:
            out.append(base + 999)
        else:
            out.append(int(l) + base)
    return np.array(out, dtype=int)

def _relabel_dict(d: Dict[int, list], base: int) -> Dict[int, list]:
    """
    reps/keywords 딕셔너리의 키(클러스터 id)를 base만큼 이동.
    """
    new_d: Dict[int, list] = {}
    for k, v in d.items():
        try:
            kk = int(k)
            new_d[kk + base] = v
        except Exception:
            # 혹시 문자열 'other' 등 들어오면 노이즈로 처리
            new_d[base + 999] = v
    return new_d

# ────────────────────────────────────────────────────────────────────────────
def run_full_pipeline(input_files: List[Path], output_dir: Path, *, resume: bool = False) -> None:
    total = len(input_files)
    logging.info("▶ Starting full pipeline for %d files", total)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # 폴라리티 → 오프셋 베이스 (기존 규칙 유지)
    base_map = {"negative": 0, "neutral": 1000, "positive": 2000}

    # ── Refinement 준비 (옵션) ──
    refine_enabled = getattr(config, "REFINEMENT_ENABLED", True)
    facets_obj = None
    refine_th = {
        "top_k_facets": 2,
        "facet_threshold": 0.32,
        "hetero_sil_threshold": 0.18,
        "min_cluster_size_for_split": 40,
        "max_local_k": 4,
        "other_label_value": "other",
    }
    stable_id_prefix_map = getattr(
        config, "REFINEMENT_STABLE_ID", {"negative": 0, "neutral": 1, "positive": 2}
    )

    if refine_enabled:
        try:
            facets_path = getattr(config, "REFINEMENT_FACETS_PATH", "rules/facets.yml")
            thresholds_path = getattr(config, "REFINEMENT_THRESHOLDS_PATH", "rules/thresholds.yml")
            # SentenceTransformer는 facets 임베딩용으로만 사용 (20개 내외 → 오버헤드 미미)
            from sentence_transformers import SentenceTransformer
            device_arg = getattr(config, "DEVICE", None)
            facet_embedder = SentenceTransformer(config.MODEL_NAME, device=device_arg)
            facets_obj = load_facets_yml(facets_path, facet_embedder)
            import yaml
            with open(thresholds_path, "r", encoding="utf-8") as f:
                _y = yaml.safe_load(f) or {}
                refine_th.update(_y.get("refinement", {}))
            logging.info("   ✓ Refinement loaded (facets=%d)", len(facets_obj))
        except Exception as e:
            logging.warning("   ⚠️ Refinement disabled due to init error: %s", e)
            refine_enabled = False

    for idx, file_path in enumerate(input_files, start=1):
        logging.info("🔄 [%d/%d] Processing %s", idx, total, file_path.name)
        stem = file_path.stem
        out_dir = output_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Load + preprocess
        t0 = time.time()
        logging.info("   1) Loading & preprocessing reviews…")
        df = load_reviews(file_path)
        df = preprocess_reviews(df)
        df = df.reset_index(drop=False).rename(columns={"index": config.REVIEW_ID_COL})
        logging.info("      → Loaded %d reviews (%.1fs)", len(df), time.time() - t0)

        # 1.5) Clause splitting
        t0 = time.time()
        logging.info("   1.5) Splitting into clauses…")
        clause_df = split_clauses(
            df, text_col="review",
            connectives=config.CLAUSE_CONNECTIVES,
            id_col=config.REVIEW_ID_COL,
        )
        logging.info("      → split_clauses done (%d clauses, %.1fs)",
                     len(clause_df), time.time() - t0)

        # 1.6) ABSA
        t0 = time.time()
        logging.info("   1.6) Running ABSA on clauses (batch_size=%d)…",
                     config.ABSA_BATCH_SIZE)
        # ABSA cache (per file)
        absa_cache = out_dir/"cache"/f"{stem}_absa.csv.gz"
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
        logging.info("      ▶ [DEBUG] ABSA polarity counts: %s",
                     absa_df["polarity"].value_counts().to_dict())

        # ── 통합 저장을 위한 누적 버퍼 ──
        combined_clause_df_list = []     # 폴라리티별 절(라벨 적용 후) 이어붙이기
        combined_reps: Dict[int, list] = {}
        combined_kw: Dict[int, list] = {}

        # 1.7) 폴라리티 루프
        for pol in ("negative", "neutral", "positive"):
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
            tuner_params = get_cluster_params(len(texts), dataset=f"{stem}_{pol}")
            umap_p, hdbscan_p = tuner_params["umap"], tuner_params["hdbscan"]

            # 3) Embedding
            emb_t0 = time.time()
            # Embedding cache (per file/polarity and model)
            emb_cache = out_dir/"cache"/f"{stem}_{pol}_{config.MODEL_NAME.replace('/', '_')}.npy"
            if resume and emb_cache.exists():
                try:
                    embeddings = np.load(emb_cache)
                    if embeddings.shape[0] != len(texts):
                        raise ValueError("shape mismatch — cache invalid")
                    logging.info(" → [%s] Embeddings cache hit %s", pol, emb_cache.name)
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
            logging.info("      → [%s] Embeddings shape: %s (%.1fs)",
                         pol, embeddings.shape, time.time() - emb_t0)

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
            logging.info("      → [%s] Reduced coords shape: %s (%.1fs)",
                         pol, coords.shape, time.time() - red_t0)

            # 5) HDBSCAN clustering
            clu_t0 = time.time()
            labels_raw, _ = cluster_embeddings(
                coords,
                min_cluster_size=hdbscan_p["min_cluster_size"],
                min_samples=hdbscan_p["min_samples"],
                metric=hdbscan_p["metric"],
                cluster_selection_epsilon=hdbscan_p["cluster_selection_epsilon"],
            )
            logging.info("      → [%s] Clustered (%d labels) (%.1fs)",
                         pol, len(labels_raw), time.time() - clu_t0)

            # 6) 진단 (폴라리티별 분포/실루엣 CSV는 그대로 저장)
            evaluate_clusters(
                labels_raw.copy(), coords, raw_embeddings=embeddings,
                output_dir=out_dir, timestamp=timestamp,
            )

            # 7) 대표 문장
            reps = extract_representatives(
                texts=texts, embeddings=embeddings,
                labels=labels_raw, top_k=config.TOP_K_REPRESENTATIVES,
            )

            # 8) (옵션) 병합
            if getattr(config, "ENABLE_CLUSTER_MERGE", False):
                merge_map, reps, _ = merge_similar_clusters(
                    reps, model_name=config.MODEL_NAME,
                    threshold=getattr(config, "CLUSTER_MERGE_THRESHOLD", 0.90),
                )
                labels_raw = np.array([merge_map.get(str(lbl), lbl) for lbl in labels_raw])

            # 9) 키워드
            kw = extract_keywords(reps, model_name=config.MODEL_NAME)

            # ── Refinement (비파괴적) : 원본 라벨 유지 + 보조 컬럼 추가 ──
            if refine_enabled and facets_obj is not None:
                # [DEBUG] pre-refine probe (labels_raw 상태 확인)
                lbl_ser = pd.Series(labels_raw)
                n_other_str = int((lbl_ser.astype(str).str.lower() == "other").sum())
                logging.info(
                    "   [DEBUG] labels_raw pre-refine: dtype=%s | uniq(head)=%s | n_other_str=%d",
                    lbl_ser.dtype,
                    lbl_ser.astype(str).unique()[:10].tolist(),
                    n_other_str,
                )
                try:
                    work_df = sub_df.copy()

                    # 안전 캐스팅: 숫자 외 값(NaN/문자열) → -1로 강제
                    lbl_ser = pd.to_numeric(pd.Series(labels_raw), errors="coerce")
                    n_nan = int(lbl_ser.isna().sum())
                    if n_nan:
                        logging.warning(
                            "   [DEBUG] labels_raw had %d non-numeric; coercing to -1", n_nan
                        )
                    labels_int = lbl_ser.fillna(-1).astype(int)
                    work_df["cluster_label"] = labels_int

                    logging.info(
                        "   [DEBUG] cluster_label before refine (head uniq)=%s",
                        work_df["cluster_label"].astype(str).unique()[:10].tolist(),
                    )

                    refined_df = refine_clusters(
                        work_df,
                        clause_embs=embeddings,
                        polarity=pol,
                        facets=facets_obj,
                        top_k_facets=int(refine_th.get("top_k_facets", 2)),
                        facet_threshold=float(refine_th.get("facet_threshold", 0.32)),
                        hetero_sil_threshold=float(refine_th.get("hetero_sil_threshold", 0.18)),
                        min_cluster_size_for_split=int(refine_th.get("min_cluster_size_for_split", 40)),
                        max_local_k=int(refine_th.get("max_local_k", 4)),
                        other_label_value=-1,  # 고정
                        stable_id_prefix=stable_id_prefix_map.get(pol, 0),
                    )
                except Exception:
                    # 전체 스택트레이스 출력 (원인 파악 쉬움)
                    logging.exception("      ⚠️ [%s] Refinement failed with exception", pol)
                    refined_df = None
            else:
                refined_df = None


            # ── 오프셋 적용 후 통합 버퍼에 축적 ──
            base = base_map[pol]
            labels_off = _offset_labels(labels_raw, base)        # np.ndarray(int)
            reps_off = _relabel_dict(reps, base)                 # Dict[int, list]
            kw_off = _relabel_dict(kw, base)                     # Dict[int, list]

            if refined_df is not None:
                # 기존 파이프라인과의 호환을 위해 cluster_label 컬럼은 오프셋 숫자로 유지
                refined_out = refined_df.copy()
                refined_out["cluster_label"] = labels_off
                # polarity, review_id, clause 등은 refined_df가 이미 보존
                combined_clause_df_list.append(refined_out)
            else:
                # 리파인이 비활성/실패한 경우 기존 방식으로 적재
                combined_clause_df_list.append(
                    sub_df.assign(cluster_label=labels_off, polarity=pol)
                )

            # reps/kw 누적(키 충돌 없음: 폴라리티별 베이스가 다름)
            combined_reps.update(reps_off)
            combined_kw.update(kw_off)

        # ── 폴라리티 루프 끝: 한 번만 저장 ──
        if combined_clause_df_list:
            combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)

            # Stable IDs (optional)
            if getattr(config, "ENABLE_STABLE_IDS", True):
                combined_clause_df, stable_map = assign_stable_ids(
                    combined_clause_df, combined_reps,
                    state_path=out_dir/"_stable_ids.json",
                    prefer_col="refined_cluster_id",
                    )
            
            save_clustered_clauses(
                clause_df=combined_clause_df,
                raw_df=df,
                keywords=combined_kw,
                output_path=out_dir / f"{stem}_clauses_clustered_{timestamp}.xlsx"
            )

            # Summary JSON (representatives + keywords)
            # JSON 키는 문자열이므로 그대로 덤프 (정렬은 파일 뷰어에서 편함)
            save_clauses_summary_json(
                combined_clause_df,
                reps=combined_reps,
                kw=combined_kw,
                output_path=out_dir / f"{stem}_clauses_summary_{timestamp}.json"
            )
            
            # run meta for this file
            try:
                from sentence_transformers import SentenceTransformer
                dim = SentenceTransformer(config.MODEL_NAME).get_sentence_embedding_dimension()
            except Exception:
                dim = -1
            write_meta_json(out_dir/"meta.json", model_name=config.MODEL_NAME, embed_dim=dim)
            logging.info("      💾 merged outputs saved")
        else:
            logging.info("   ⏭️ No clauses passed threshold for any polarity — nothing to save.")

        logging.info("✅ Completed %s (%d/%d)\n", stem, idx, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clause-level clustering pipeline runner")
    parser.add_argument("--files", nargs="*", type=Path, default=config.INPUT_FILES,
                        help="List of input Excel files")
    parser.add_argument("--output_dir", type=Path, default=Path(config.OUTPUT_DIR),
                        help="Directory to save outputs")
    parser.add_argument("--resume", action="store_true", help="Reuse caches if present (ABSA, embeddings)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = Path(config.OUTPUT_DIR) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"clause_pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)]
    )
    # run manifest (top level)
    write_run_manifest(Path(config.OUTPUT_DIR)/"run_manifest.json", config_obj=config)    
    run_full_pipeline(args.files, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()

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
def run_full_pipeline(input_files: List[Path], output_dir: Path) -> None:
    total = len(input_files)
    logging.info("▶ Starting full pipeline for %d files", total)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # 폴라리티 → 오프셋 베이스
    base_map = {"negative": 0, "neutral": 1000, "positive": 2000}

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
        raw_absa = classify_clauses(
            clause_df,
            model_name=config.ABSA_MODEL_NAME,
            batch_size=config.ABSA_BATCH_SIZE,
            device=config.DEVICE,
        )
        logging.info("      → ABSA complete (%d results, %.1fs)",
                     len(raw_absa), time.time() - t0)

        absa_df = pd.DataFrame(
            raw_absa, columns=["review_id", "clause", "polarity", "confidence"]
        )
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
            embeddings = embed_reviews(
                texts, model_name=config.MODEL_NAME,
                batch_size=config.BATCH_SIZE, device=config.DEVICE,
            )
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
                labels_raw, coords, raw_embeddings=embeddings,
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

            # ── 오프셋 적용 후 통합 버퍼에 축적 ──
            base = base_map[pol]
            labels_off = _offset_labels(labels_raw, base)        # np.ndarray(int)
            reps_off = _relabel_dict(reps, base)                 # Dict[int, list]
            kw_off = _relabel_dict(kw, base)                     # Dict[int, list]

            # 절 데이터 누적 (원래 sub_df 컬럼 + 오프셋 라벨 + 폴라리티)
            combined_clause_df_list.append(
                sub_df.assign(cluster_label=labels_off, polarity=pol)
            )

            # reps/kw 누적(키 충돌 없음: 폴라리티별 베이스가 다름)
            combined_reps.update(reps_off)
            combined_kw.update(kw_off)

        # ── 폴라리티 루프 끝: 한 번만 저장 ──
        if combined_clause_df_list:
            combined_clause_df = pd.concat(combined_clause_df_list, ignore_index=True)

            # Excel (clauses/mapping/reviews)
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
    run_full_pipeline(args.files, args.output_dir)


if __name__ == "__main__":
    main()

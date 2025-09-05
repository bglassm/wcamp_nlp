import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import config

from sklearn.metrics import silhouette_score

try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "🛑 `hdbscan`가 설치되어 있지 않습니다. ``pip install hdbscan``로 설치 후 사용하세요."
    ) from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def cluster_embeddings(
    embeddings_2d: np.ndarray,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: Optional[int] = None,
    metric: str = config.HDBSCAN_METRIC,
    cluster_selection_epsilon: float = config.HDBSCAN_SELECTION_EPS,
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Cluster reduced embeddings using HDBSCAN with parameters from config by default.
    - config.HANDLE_OUTLIERS=True 이면, noise(-1) 레이블을 config.OUTLIER_LABEL로 치환합니다.
    """
    if embeddings_2d.ndim != 2:
        raise ValueError("`embeddings_2d` must be 2-D array (n_samples × n_dim).")

    if min_samples is None:
        min_samples = (
            config.HDBSCAN_MIN_SAMPLES
            if hasattr(config, "HDBSCAN_MIN_SAMPLES")
            else max(2, min_cluster_size // 2)
        )

    logger.info(
        "🧩 HDBSCAN clustering: %s • min_cluster_size=%d • min_samples=%d • metric=%s • epsilon=%.3f",
        embeddings_2d.shape,
        min_cluster_size,
        min_samples,
        metric,
        cluster_selection_epsilon,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=False,
    ).fit(embeddings_2d)

    labels = clusterer.labels_
    # ─────── outlier 처리 ───────
    if getattr(config, "HANDLE_OUTLIERS", False):
        # labels 배열의 dtype을 object로 바꿔야 문자열과 숫자를 섞을 수 있습니다
        labels = np.array(
            [t if t != -1 else config.OUTLIER_LABEL for t in labels],
            dtype=object,
        )
    # ──────────────────────────────

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == config.OUTLIER_LABEL).sum()) if config.HANDLE_OUTLIERS else int((clusterer.labels_ == -1).sum())

    logger.info(
        "✅ Clustering complete — %d clusters • %d noise/%s points",
        n_clusters,
        n_noise,
        config.OUTLIER_LABEL if config.HANDLE_OUTLIERS else "-1",
    )
    return labels, clusterer

# pipeline/clusterer.py 안의 evaluate_clusters를 다음으로 교체

def evaluate_clusters(
    labels: np.ndarray,
    embeddings_2d: np.ndarray,
    raw_embeddings: np.ndarray | None = None,
    logger: logging.Logger | None = None,
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> None:
    """
    클러스터 분포와 실루엣 점수를 안전하게 기록/저장합니다.
    - 라벨에 int와 str('other')가 섞여 있어도 동작
    - 실루엣 계산은 유효 라벨이 2개 이상일 때만 수행
    - 분포/통계 CSV 저장
    """
    lg = logger or logging.getLogger(__name__)

    # --- 1) 분포: 혼합형 라벨 안전 처리 (문자열로 변환 후 집계) ---
    labels_str = np.array([str(x) for x in np.asarray(labels, dtype=object)])
    uniq, counts = np.unique(labels_str, return_counts=True)
    dist_info = ", ".join(f"{c}@{u}" for u, c in zip(uniq, counts))
    lg.info("🔢 Cluster distribution: %s", dist_info)

    if output_dir and timestamp:
        try:
            df_dist = pd.DataFrame({"cluster_label": uniq, "count": counts})
            dist_path = Path(output_dir) / f"cluster_distribution_{timestamp}.csv"
            df_dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
            lg.info("💾 Cluster distribution saved → %s", dist_path)
        except Exception:
            lg.exception("⚠️ failed to save cluster_distribution CSV")

    # --- 2) 실루엣: noise/other 제외 후 유효성 점검 ---
    # 제외 대상 라벨 집합: -1, 'other' (대소문자 무시)
    exclude = {"-1", "other"}
    mask = np.array([s.lower() not in exclude for s in labels_str], dtype=bool)

    # 유효 표본 수와 유효 라벨 개수 점검
    valid_n = int(mask.sum())
    valid_labels = np.unique(labels_str[mask]) if valid_n > 0 else np.array([])
    n_valid_labels = len(valid_labels)

    if valid_n >= 2 and n_valid_labels >= 2:
        from sklearn.metrics import silhouette_score

        try:
            sil_umap = float(silhouette_score(embeddings_2d[mask], labels_str[mask], metric="euclidean"))
        except Exception:
            lg.exception("⚠️ Silhouette (UMAP) failed")
            sil_umap = None

        sil_raw = None
        if raw_embeddings is not None:
            try:
                sil_raw = float(silhouette_score(raw_embeddings[mask], labels_str[mask], metric="cosine"))
            except Exception:
                lg.exception("⚠️ Silhouette (raw) failed")

        msg = "🔍 Silhouette"
        if sil_umap is not None:
            msg += f" (UMAP): {sil_umap:.3f}"
        if sil_raw is not None:
            msg += f" | (raw): {sil_raw:.3f}"
        lg.info(msg)

        if output_dir and timestamp:
            try:
                df_stats = pd.DataFrame([{
                    "silhouette_umap": sil_umap,
                    "silhouette_raw": sil_raw,
                    "n_clusters": int(n_valid_labels),
                    "n_noise_or_other": int((~mask).sum()),
                    "n_samples_used": valid_n,
                }])
                stats_path = Path(output_dir) / f"cluster_stats_{timestamp}.csv"
                df_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
                lg.info("💾 Cluster stats saved → %s", stats_path)
            except Exception:
                lg.exception("⚠️ failed to save cluster_stats CSV")
    else:
        # 유효 라벨이 1개거나 표본이 부족할 때
        if valid_n < 2:
            lg.info("🔍 Silhouette: insufficient non-noise points (<2)")
        else:
            lg.info("🔍 Silhouette: only one valid label (need ≥2 distinct labels)")


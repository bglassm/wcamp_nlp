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

def evaluate_clusters(
    labels: np.ndarray,
    embeddings_2d: np.ndarray,
    raw_embeddings: np.ndarray | None = None,
    logger: logging.Logger | None = None,
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> None:
    """
    Log cluster size distribution and silhouette scores.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels (-1 for noise or other outlier label).
    embeddings_2d : np.ndarray
        UMAP‐reduced embeddings used for clustering.
    raw_embeddings : np.ndarray, optional
        Original high‐dimensional embeddings. If provided, will also compute
        silhouette on raw space (cosine).
    logger : logging.Logger, optional
        Logger to use; defaults to module logger.
    output_dir : Path, optional
        Directory to save CSVs.
    timestamp : str, optional
        Timestamp suffix for filenames.
    """
    lg = logger or logging.getLogger(__name__)

    # --- 1) 문자열 레이블 변환 후 분포 계산 ---
    # int 와 str 이 섞여 있을 수 있으니 모두 str 로 바꿔 처리
    labels_str = labels.astype(str)
    unique, counts = np.unique(labels_str, return_counts=True)
    dist_info = ", ".join(f"{c}@{u}" for u, c in zip(unique, counts))
    lg.info("🔢 Cluster distribution: %s", dist_info)

    # CSV 로 저장
    if output_dir and timestamp:
        df_dist = pd.DataFrame({
            "cluster_label": unique,
            "count": counts,
        })
        dist_path = Path(output_dir) / f"cluster_distribution_{timestamp}.csv"
        df_dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
        lg.info("💾 Cluster distribution saved → %s", dist_path)

    # --- 2) Silhouette 점수 계산 ---
    # noise(-1) 또는 other 처리된 레이블 제외
    mask = labels_str != str(config.OUTLIER_LABEL if getattr(config, "OUTLIER_LABEL", "-1") else "-1")
    if mask.sum() >= 2:
        from sklearn.metrics import silhouette_score

        sil_umap = silhouette_score(
            embeddings_2d[mask], labels_str[mask], metric="euclidean"
        )
        msg = f"🔍 Silhouette (UMAP): {sil_umap:.3f}"

        if raw_embeddings is not None:
            sil_raw = silhouette_score(
                raw_embeddings[mask], labels_str[mask], metric="cosine"
            )
            msg += f" | (raw): {sil_raw:.3f}"

        lg.info(msg)

        if output_dir and timestamp:
            df_stats = pd.DataFrame([{
                "silhouette_umap": sil_umap,
                "silhouette_raw": sil_raw if raw_embeddings is not None else None,
                "n_clusters": int(len(set(labels_str[mask])) - (1 if str(config.OUTLIER_LABEL) in set(labels_str) else 0)),
                "n_noise": int((~mask).sum()),
            }])
            stats_path = Path(output_dir) / f"cluster_stats_{timestamp}.csv"
            df_stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
            lg.info("💾 Cluster stats saved → %s", stats_path)
    else:
        lg.info("🔍 Silhouette: insufficient non‐noise points (<2)")

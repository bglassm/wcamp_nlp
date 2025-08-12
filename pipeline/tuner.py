# pipeline/tuner.py

from typing import Dict, Optional
import numpy as np
import config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dataset-specific fixed overrides (optional)
# 예: {"abalone": {"pct": 0.01}}
_OVERRIDES: Dict[str, Dict[str, float]] = {}


def get_cluster_params(
    n_docs: int,
    dataset: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    n_docs 및 dataset에 따라 UMAP/HDBSCAN 파라미터를 계산해 반환.
    Returns a dict with keys:
      - "umap": dict of UMAP kwargs
      - "hdbscan": dict of HDBSCAN kwargs
      - "pct": the proportion used to compute min_cluster_size
    """
    # n_docs가 1 미만이면 기본 파라미터 반환 (클러스터링 최소화)
    if n_docs < 1:
        logger.info(f"[TUNER] n_docs={n_docs} (<1), using default parameters")
        return {
            "umap": {
                "n_neighbors": 15,
                "n_components": 2,
                "min_dist": config.UMAP_MIN_DIST,
                "metric": config.UMAP_METRIC,
                "random_state": config.UMAP_RANDOM_STATE,
            },
            "hdbscan": {
                "min_cluster_size": 2,
                "min_samples": 1,
                "metric": config.HDBSCAN_METRIC,
                "cluster_selection_epsilon": getattr(config, "HDBSCAN_SELECTION_EPS", 0.0),
            },
            "pct": 0.01,
        }

    # 1) Dataset override 우선 적용
    if dataset and dataset in _OVERRIDES:
        ov = _OVERRIDES[dataset]
        pct = ov.get("pct", 0.01)
        logger.info(f"[TUNER-OVERRIDE] dataset={dataset}, pct={pct:.3f}")
    else:
        # 2) 리뷰 수 규모별 비례 계수 설정
        if n_docs > 10000:
            pct = 0.005
        elif n_docs > 5000:
            pct = 0.01
        else:
            pct = 0.005
        logger.info(f"[TUNER] n_docs={n_docs}, pct={pct:.3f}")

    # 3) UMAP 파라미터 계산
    n_neighbors  = min(100, max(5, int(np.sqrt(n_docs))))
    n_components = max(2, int(np.log2(n_docs)))  # 최소 2차원
    umap_params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_dist": config.UMAP_MIN_DIST,
        "metric": config.UMAP_METRIC,
        "random_state": config.UMAP_RANDOM_STATE,
    }

    # 4) HDBSCAN 파라미터 계산
    min_cluster_size = max(5, int(n_docs * pct))
    min_samples      = max(2, int(min_cluster_size * 0.5))
    hdbscan_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": config.HDBSCAN_METRIC,
        "cluster_selection_epsilon": getattr(config, "HDBSCAN_SELECTION_EPS", 0.0),
    }

    logger.info(
        "[TUNER] using n_neighbors=%d, n_components=%d, min_cluster_size=%d, min_samples=%d",
        n_neighbors, n_components, min_cluster_size, min_samples
    )

    return {
        "umap": umap_params,
        "hdbscan": hdbscan_params,
        "pct": pct,
    }
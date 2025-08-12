import logging

import numpy as np
import config
try:
    from umap import UMAP
except ImportError as e:
    raise ImportError(
        "🛑 `umap-learn`가 설치되어 있지 않습니다. ``pip install umap-learn``로 설치 후 사용하세요."
    ) from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = config.UMAP_DIMS_CLUSTER,
    n_neighbors: int = config.UMAP_N_NEIGHBORS,
    min_dist: float = config.UMAP_MIN_DIST,
    metric: str = config.UMAP_METRIC,
    random_state: int = config.UMAP_RANDOM_STATE,
) -> np.ndarray:
    """
    Reduce SBERT embeddings to lower-dimensional coordinates using UMAP.

    Parameters are loaded from config by default but can be overridden.
    """
    if embeddings.ndim != 2:
        raise ValueError("`embeddings` must be 2-D array (n_samples × n_features).")

    logger.info(
        "📉 UMAP reducing: %s → %dD (n_neighbors=%d, min_dist=%.2f, metric=%s)",
        embeddings.shape,
        n_components,
        n_neighbors,
        min_dist,
        metric
    )

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    reduced = reducer.fit_transform(embeddings)

    logger.info("✅ Reduction complete — new shape: %s", reduced.shape)
    return reduced

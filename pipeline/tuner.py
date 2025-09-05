# pipeline/tuner.py
from __future__ import annotations
import math
import logging
from typing import Dict, Optional

import numpy as np

import config

log = logging.getLogger("Kss")  # 기존 로그와 동일 prefix 사용

def _pick_pct(n_docs: int) -> float:
    """데이터 크기에 따라 베이스 pct를 선택."""
    return config.TUNER_BASE_PCT_SMALL if n_docs < 6000 else config.TUNER_BASE_PCT_LARGE

def _pick_umap_dims(n_docs: int) -> int:
    """데이터 크기에 따라 n_components를 완만히 조정."""
    if n_docs < 500:
        return max(config.TUNER_UMAP_MIN_DIMS, 8)
    if n_docs < 2000:
        return 10
    if n_docs < 8000:
        return 12
    if n_docs < 16000:
        return 13
    return config.TUNER_UMAP_MAX_DIMS

def _as_bucket(dataset: Optional[str]) -> bool:
    """
    버킷 컨텍스트 판정:
    main에서 dataset=f"{stem}_{pol}_{facetId}" 형태로 넘기므로
    언더스코어가 2개 이상이면 버킷으로 간주.
    """
    if not dataset:
        return False
    return dataset.count("_") >= 2

def get_cluster_params(n_rows: int, dataset: Optional[str] = None) -> Dict:
    """
    UMAP/HDBSCAN 파라미터 자동 산출.
    - 전역(폴라리티 단위)과 버킷(파셋 단위)을 컨텍스트로 구분
    - 버킷 컨텍스트에서는 '합치는' 방향으로 완만히 조정
    반환 형식:
    {
      "umap": {"n_components", "n_neighbors", "min_dist", "metric", "random_state"},
      "hdbscan": {"min_cluster_size", "min_samples", "metric", "cluster_selection_epsilon"},
    }
    """
    n = max(1, int(n_rows))
    pct = _pick_pct(n)
    is_bucket = _as_bucket(dataset)

    # --- UMAP ---
    # 이웃수: 대략 n * pct를 베이스로, 범위를 5~100로 제한
    base_neighbors = max(5, min(config.TUNER_UMAP_MAX_NEIGHBORS, int(round(n * pct))))
    if is_bucket:
        n_neighbors = int(round(base_neighbors * config.BUCKET_UMAP_NEIGHBORS_MULT))
    else:
        n_neighbors = base_neighbors

    # 차원 수: 규모에 따라 완만 조정
    n_components = _pick_umap_dims(n)

    # min_dist: 버킷이면 조금 키워서 과분할 억제
    min_dist = config.UMAP_MIN_DIST
    if is_bucket:
        min_dist = max(min_dist, float(config.BUCKET_UMAP_MIN_DIST))

    umap_params = {
        "n_components": int(n_components),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": config.UMAP_METRIC,
        "random_state": config.UMAP_RANDOM_STATE,
    }

    # --- HDBSCAN ---
    base_min_cluster_size = max(5, int(round(n * pct)))
    base_min_samples = max(2, int(round(base_min_cluster_size * 0.5)))

    if is_bucket:
        min_cluster_size = int(round(base_min_cluster_size * config.BUCKET_MIN_CLUSTER_SIZE_MULT))
        min_samples      = int(round(base_min_samples      * config.BUCKET_MIN_SAMPLES_MULT))
        eps = float(config.HDBSCAN_SELECTION_EPS) + float(config.BUCKET_SELECTION_EPS_ADD)
    else:
        min_cluster_size = base_min_cluster_size
        min_samples      = base_min_samples
        eps = float(config.HDBSCAN_SELECTION_EPS)

    hdbscan_params = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "metric": config.HDBSCAN_METRIC,
        "cluster_selection_epsilon": float(eps),
    }

    # ---- 로그 (기존 형식과 유사) ----
    log.info("[TUNER] n_docs=%d, pct=%.3f%s",
             n, pct, " [bucket]" if is_bucket else "")
    log.info("[TUNER] using n_neighbors=%d, n_components=%d, "
             "min_cluster_size=%d, min_samples=%d",
             n_neighbors, n_components, min_cluster_size, min_samples)

    return {"umap": umap_params, "hdbscan": hdbscan_params}

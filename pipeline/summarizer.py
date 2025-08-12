import logging
import numpy as np
from typing import Dict, List
import config

def extract_representatives(
    texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: int = config.TOP_K_REPRESENTATIVES
) -> Dict[int, List[str]]:
    """
    각 클러스터별로 중심에 가장 가까운 top_k개의 representative sentences를 추출합니다.

    Parameters
    ----------
    texts : list[str]
        원본 텍스트 리스트.
    embeddings : np.ndarray
        텍스트 임베딩 배열 (n_samples, dim).
    labels : np.ndarray
        각 텍스트의 클러스터 레이블 (정수 또는 문자열 혼합 가능).
    top_k : int
        클러스터당 대표 문장 수.

    Returns
    -------
    reps : Dict[int, List[str]]
        {cluster_id: [rep1, rep2, ...]} 형식의 대표 문장 딕셔너리.
    """
    logger = logging.getLogger(__name__)
    # 수치형 레이블만 대상으로 cluster_ids 구성 (문자열 레이블은 제외)
    cluster_ids = set()
    for lbl in labels:
        try:
            li = int(lbl)
            if li != -1:
                cluster_ids.add(li)
        except (ValueError, TypeError):
            # e.g., config.OUTLIER_LABEL 같은 문자열 레이블은 건너뜀
            continue
    cluster_ids = sorted(cluster_ids)
    logger.info("🔎 Extracting representatives for %d clusters …", len(cluster_ids))

    reps: Dict[int, List[str]] = {}
    # 각 클러스터에서 거리 계산하여 대표 문장 선택
    for cid in cluster_ids:
        idxs = [i for i, lbl in enumerate(labels) if str(lbl) == str(cid)]
        if not idxs:
            continue
        cluster_embeds = embeddings[idxs]
        center = cluster_embeds.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(cluster_embeds - center, axis=1)
        nearest = np.argsort(dists)[:top_k]
        reps[cid] = [texts[idxs[i]] for i in nearest]
    return reps

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from pipeline.embedder import _get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# SBERT 모델 캐시
_MODEL_CACHE: Dict[str, object] = { }


def merge_similar_clusters(
    reps: Dict[Union[int, str], List[str]],
    model_name: str = config.MODEL_NAME,
    threshold: float = config.CLUSTER_MERGE_THRESHOLD,
    save_candidates_path: Union[str, Path, None] = None,
) -> Tuple[Dict[str, int], Dict[int, List[str]], List[Dict]]:
    """
    Merge clusters whose representative sentences exceed a similarity threshold.

    Parameters
    ----------
    reps : dict[cluster_id, list[str]]
        Cluster → 대표 문장 리스트 (문자열 키)
    model_name : str
        SBERT 모델 이름 (config.MODEL_NAME 기본)
    threshold : float
        Cosine similarity 이상일 때 병합
    save_candidates_path : str | Path | None
        JSON로 후보 쌍 저장 경로 (예: merge_candidates.json)

    Returns
    -------
    merge_map : dict[str, int]
        원본 cluster_id(str) → 병합 후 cluster 인덱스(int)
    merged_reps : dict[int, list[str]]
        병합된 cluster 인덱스 → 모든 대표 문장 리스트
    candidates : list[dict]
        병합 후보 리스트(cosine desc)
    """
    if not reps:
        raise ValueError("Empty representatives dict")

    # 1) Load SBERT embedding model
    if model_name not in _MODEL_CACHE:
        logger.info("⬇️  Loading SBERT for merging: %s", model_name)
        _MODEL_CACHE[model_name] = _get_model(model_name, config.DEVICE)
    model = _MODEL_CACHE[model_name]

    # 2) Prepare representative sentences (only clusters that have at least one rep)
    pairs = [(cid, reps[cid][0]) for cid in reps if reps[cid]]
    if pairs:
        cluster_ids, head_sents = zip(*pairs)
        cluster_ids = list(cluster_ids)
        head_sents  = list(head_sents)
    else:
        cluster_ids, head_sents = [], []

    # 3) Embed head sentences
    head_vecs = model.encode(
        head_sents,
        batch_size=config.BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=config.DEVICE,
        normalize_embeddings=True,
    )

    # 4) Compute similarity matrix
    sim_mat = cosine_similarity(head_vecs)
    n = len(head_sents)

    # 5) Collect candidate pairs
    candidates: List[Dict] = []
    for i in range(n):
        for j in range(i+1, n):
            sim = float(sim_mat[i, j])
            if sim >= threshold:
                candidates.append({
                    "cid1": cluster_ids[i],
                    "cid2": cluster_ids[j],
                    "sim": round(sim, 4),
                    "rep1": head_sents[i],
                    "rep2": head_sents[j],
                })
    candidates.sort(key=lambda x: x["sim"], reverse=True)
    logger.info("🔗 Found %d merge candidates (thr=%.2f)", len(candidates), threshold)

    # 6) Union-Find merge
    parent = {i: i for i in range(n)}
    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u: int, v: int) -> None:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
    for cand in candidates:
        i = cluster_ids.index(cand["cid1"])
        j = cluster_ids.index(cand["cid2"])
        union(i, j)

    # 7) Build merge_map and merged_reps
    root_to_new: Dict[int, int] = {}
    merge_map: Dict[str, int] = {}
    for idx, cid in enumerate(cluster_ids):
        root = find(idx)
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new)
        merge_map[str(cid)] = root_to_new[root]

    merged_reps: Dict[int, List[str]] = {}
    for orig_cid, new_cid in merge_map.items():
        key = orig_cid
        if key not in reps and isinstance(key, str):
            try:
                i = int(key)
            except ValueError:
                i = key
            if i in reps:
                key = i

        rep_list = reps.get(key, [])
        merged_reps.setdefault(new_cid, []).extend(rep_list)
    for new_cid in merged_reps:
        merged_reps[new_cid] = list(dict.fromkeys(merged_reps[new_cid]))

    logger.info("✅ Merge done: %d → %d clusters", len(reps), len(merged_reps))

    # 8) Save candidates JSON if requested
    if save_candidates_path:
        path = Path(save_candidates_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(candidates, fp, ensure_ascii=False, indent=2)
        logger.info("💾 Merge candidates saved → %s", path)

    return merge_map, merged_reps, candidates

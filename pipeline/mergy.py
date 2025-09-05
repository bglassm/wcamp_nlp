import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from pipeline.embedder import _get_model

# Quieter logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# SBERT model cache
_MODEL_CACHE: Dict[str, object] = {}


def _as_int_if_possible(x: Union[str, int]) -> Union[str, int]:
    try:
        return int(x)
    except Exception:
        return str(x)


def _centroids_from_reps(
    reps: Dict[Union[int, str], List[str]],
    model,
    *,
    topk_per_cluster: int = 3,
) -> Tuple[List[Union[int, str]], np.ndarray, Dict[Union[int, str], List[str]]]:
    """Embed up to top-k reps per cluster and return L2-normalized centroids.

    Returns
    -------
    cluster_ids : list
        Cluster ids in the same order as centroids.
    centroids : np.ndarray (C, D)
        Unit-normalized centroid embedding per cluster.
    pruned_reps : dict
        reps pruned to top-k actually used (for merged output).
    """
    cluster_ids: List[Union[int, str]] = []
    chunks: List[str] = []
    ranges: List[Tuple[int, int]] = []
    pruned: Dict[Union[int, str], List[str]] = {}

    for cid, sents in reps.items():
        if not sents:
            continue
        use = sents[:topk_per_cluster]
        start = len(chunks)
        chunks.extend(use)
        end = len(chunks)
        if end > start:
            cluster_ids.append(cid)
            ranges.append((start, end))
            pruned[cid] = use

    if not cluster_ids:
        return [], np.empty((0, 0), dtype=np.float32), {}

    vecs = model.encode(
        chunks,
        batch_size=getattr(config, "BATCH_SIZE", 64),
        show_progress_bar=False,
        convert_to_numpy=True,
        device=getattr(config, "DEVICE", None),
        normalize_embeddings=True,
    ).astype(np.float32)

    # mean per cluster, then re-normalize
    cents: List[np.ndarray] = []
    for (s, e) in ranges:
        c = vecs[s:e].mean(axis=0)
        n = np.linalg.norm(c) + 1e-12
        cents.append((c / n).astype(np.float32))
    centroids = np.stack(cents, axis=0)
    return cluster_ids, centroids, pruned


def merge_similar_clusters(
    reps: Dict[Union[int, str], List[str]],
    *,
    model_name: str = config.MODEL_NAME,
    threshold: Optional[float] = config.CLUSTER_MERGE_THRESHOLD,
    save_candidates_path: Union[str, Path, None] = None,
    topk_per_cluster: int = 3,
    same_facet_only: bool = False,
    cluster_facet: Optional[Dict[Union[int, str], str]] = None,
    dynamic_percentile: Optional[float] = None,  # e.g., 95.0 → use p95 if threshold is None
) -> Tuple[Dict[str, int], Dict[int, List[str]], List[Dict]]:
    """
    Merge clusters whose representative *centroids* exceed a similarity threshold.

    Differences vs. previous version
    - uses centroid of top-k reps per cluster (more robust than head-only)
    - optional facet gating (prevent cross-facet merges)
    - optional dynamic threshold from similarity distribution
    - quieter logging & progress bars

    Parameters
    ----------
    reps : dict[cluster_id, list[str]]
        Cluster → representative sentences.
    model_name : str
        SentenceTransformer model name (defaults to config.MODEL_NAME).
    threshold : float | None
        Cosine similarity cutoff to merge. If None and `dynamic_percentile` is set,
        a percentile-based threshold will be used.
    save_candidates_path : str | Path | None
        If provided, dump candidate pairs to JSON.
    topk_per_cluster : int
        Number of representative sentences to build the centroid.
    same_facet_only : bool
        If True, allow merging only when both clusters share the same facet.
    cluster_facet : dict[cluster_id, str] | None
        Mapping cluster id → facet id (used when same_facet_only=True).
    dynamic_percentile : float | None
        If set (e.g., 95.0) and `threshold` is None, compute threshold as that
        percentile of the off-diagonal similarities.

    Returns
    -------
    merge_map : dict[str, int]
        Original cluster id (as str) → merged cluster index (0..M-1).
    merged_reps : dict[int, list[str]]
        Merged index → concatenated unique reps.
    candidates : list[dict]
        Candidate merge pairs with similarities (sorted desc).
    """
    if not reps:
        return {}, {}, []

    # 1) Load embedding model (cached)
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = _get_model(model_name, getattr(config, "DEVICE", None))
    model = _MODEL_CACHE[model_name]

    # 2) Build centroids from top-k reps
    cluster_ids, centroids, pruned_reps = _centroids_from_reps(reps, model, topk_per_cluster=topk_per_cluster)
    if len(cluster_ids) <= 1:
        # nothing to merge
        return {str(cluster_ids[0]): 0} if cluster_ids else {}, {0: pruned_reps.get(cluster_ids[0], [])} if cluster_ids else {}, []

    # 3) Similarity matrix (cosine on normalized centroids)
    sim_mat = cosine_similarity(centroids)
    n = len(cluster_ids)

    # 4) Determine threshold (static or dynamic)
    thr = threshold
    if thr is None and dynamic_percentile is not None:
        # take upper-tri off-diagonal similarities
        iu = np.triu_indices(n, k=1)
        sims = sim_mat[iu]
        if sims.size:
            thr = float(np.percentile(sims, float(dynamic_percentile)))
        else:
            thr = 0.999  # degenerate; will produce no merges

    if thr is None:
        thr = 0.90  # safe default if nothing provided

    # 5) Collect candidate pairs
    facet_of: Dict[str, str] = {}
    if cluster_facet:
        # normalize keys to str for stable lookup
        facet_of = {str(_as_int_if_possible(k)): v for k, v in cluster_facet.items()}

    candidates: List[Dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            if same_facet_only and facet_of:
                fi = facet_of.get(str(_as_int_if_possible(cluster_ids[i])), None)
                fj = facet_of.get(str(_as_int_if_possible(cluster_ids[j])), None)
                if (fi is None) or (fj is None) or (fi != fj):
                    continue
            sim = float(sim_mat[i, j])
            if sim >= thr:
                candidates.append({
                    "cid1": cluster_ids[i],
                    "cid2": cluster_ids[j],
                    "sim": round(sim, 4),
                    "rep1": pruned_reps.get(cluster_ids[i], [""])[0],
                    "rep2": pruned_reps.get(cluster_ids[j], [""])[0],
                })
    candidates.sort(key=lambda x: x["sim"], reverse=True)

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

    # apply unions
    idx_of = {cluster_ids[i]: i for i in range(n)}
    for cand in candidates:
        i = idx_of[cand["cid1"]]
        j = idx_of[cand["cid2"]]
        union(i, j)

    # 7) Build merge_map (original → new contiguous index) and merged reps
    root_to_new: Dict[int, int] = {}
    merge_map: Dict[str, int] = {}

    for idx, cid in enumerate(cluster_ids):
        root = find(idx)
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new)
        merge_map[str(_as_int_if_possible(cid))] = root_to_new[root]

    merged_reps: Dict[int, List[str]] = {}
    for orig_cid, new_idx in merge_map.items():
        # recover the original key type to access pruned_reps
        key: Union[int, str]
        try:
            key = int(orig_cid)
        except Exception:
            key = orig_cid
        rep_list = reps.get(key, [])
        if not rep_list and isinstance(key, int):
            # fall back to string key if necessary
            rep_list = reps.get(str(key), [])
        merged_reps.setdefault(new_idx, []).extend(rep_list)

    # deduplicate reps per merged group preserving order
    for k in list(merged_reps.keys()):
        seen = set()
        uniq = []
        for s in merged_reps[k]:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        merged_reps[k] = uniq

    # 8) Optionally save candidates
    if save_candidates_path:
        path = Path(save_candidates_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(candidates, fp, ensure_ascii=False, indent=2)

    return merge_map, merged_reps, candidates

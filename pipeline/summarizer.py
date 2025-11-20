import logging
from typing import Dict, List, Optional

import numpy as np

import config
from pipeline.embedder import _get_model

logger = logging.getLogger(__name__)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length, guarding against zero vectors."""

    if embeddings is None or not embeddings.size:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _load_semantic_helper():
    """Load the configured SBERT helper model for semantic operations."""

    semantic_cfg = getattr(config, "semantic", None)
    model_name = getattr(semantic_cfg, "model", None) or getattr(config, "MODEL_NAME", None)
    device = getattr(semantic_cfg, "device", None)
    if not model_name:
        logger.warning("Semantic helper model name missing; skipping semantic MMR")
        return None

    try:
        return _get_model(model_name, device)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load semantic helper %s (%s); using provided embeddings", model_name, exc)
        return None


def _get_semantic_embeddings(
    texts: List[str],
    provided_embeddings: Optional[np.ndarray],
    semantic_model,
) -> Optional[np.ndarray]:
    """Prefer SBERT helper embeddings; reuse provided ones if dimensions match."""

    if semantic_model is None:
        return None

    target_dim: Optional[int] = None
    try:
        target_dim = semantic_model.get_sentence_embedding_dimension()
    except Exception:  # pragma: no cover - best effort
        target_dim = None

    if (
        provided_embeddings is not None
        and provided_embeddings.size
        and target_dim is not None
        and provided_embeddings.shape[1] == target_dim
    ):
        logger.info("Reusing provided embeddings as semantic helper outputs (dim=%d)", target_dim)
        return _normalize_embeddings(np.asarray(provided_embeddings))

    try:
        return semantic_model.encode(
            texts,
            batch_size=getattr(config, "BATCH_SIZE", 64),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Semantic helper encoding failed; falling back to provided embeddings (%s)", exc)
        return None


def _compute_cluster_centroid(cluster_embeds: np.ndarray) -> Optional[np.ndarray]:
    """Compute and normalize a centroid for the given cluster embeddings."""

    if cluster_embeds is None or not cluster_embeds.size:
        return None
    centroid = cluster_embeds.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if not norm:
        return None
    return centroid / norm


def _fallback_nearest(texts: List[str], embeddings: np.ndarray, top_k: int) -> List[str]:
    """Fallback to nearest-to-centroid selection when MMR cannot proceed."""

    if embeddings is None or not embeddings.size:
        return texts[:top_k]
    centroid = _compute_cluster_centroid(embeddings)
    if centroid is None:
        return texts[:top_k]
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    nearest = np.argsort(dists)[:top_k]
    return [texts[i] for i in nearest]


def _mmr_select(
    candidate_texts: List[str],
    candidate_embs: np.ndarray,
    centroid: np.ndarray,
    top_k: int,
    lambda_mult: float,
    duplicate_threshold: float,
) -> List[str]:
    """Run an MMR loop to select diverse, centroid-relevant sentences."""

    if not candidate_texts or candidate_embs is None or not candidate_embs.size:
        return []

    relevance = candidate_embs @ centroid
    remaining = list(range(len(candidate_texts)))
    selected: List[int] = []

    while remaining and len(selected) < top_k:
        if not selected:
            best_idx = remaining.pop(0)
            selected.append(best_idx)
            continue

        best_choice = None
        best_score = -np.inf

        for idx in remaining:
            sim_to_selected = np.max(candidate_embs[selected] @ candidate_embs[idx])
            mmr_score = lambda_mult * relevance[idx] - (1 - lambda_mult) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_choice = idx

        if best_choice is None:
            break

        max_dup_sim = float(np.max(candidate_embs[selected] @ candidate_embs[best_choice]))
        if max_dup_sim >= duplicate_threshold:
            remaining.remove(best_choice)
            continue

        selected.append(best_choice)
        remaining.remove(best_choice)

    ordered = [candidate_texts[i] for i in selected]
    return ordered[:top_k]

def extract_representatives(
    texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    top_k: int = config.TOP_K_REPRESENTATIVES,
) -> Dict[int, List[str]]:
    """Select diverse representative sentences per cluster using MMR.

    The routine prioritizes SBERT semantic helper embeddings for relevance/diversity
    scoring; if unavailable it falls back to the provided embeddings. Public
    signature remains stable for downstream callers.
    """

    requested_top_k = max(1, top_k or getattr(config, "MAX_REPRESENTATIVE_SENTENCES", 1))
    max_allowed = getattr(config, "MAX_REPRESENTATIVE_SENTENCES", requested_top_k)
    final_top_k = min(requested_top_k, max_allowed)

    candidate_multiplier = max(1, getattr(config, "REPRESENTATIVE_CANDIDATE_MULTIPLIER", 10))
    lambda_mult = float(getattr(config, "REPRESENTATIVE_MMR_LAMBDA", 0.7))
    duplicate_threshold = float(getattr(config, "REPRESENTATIVE_DUPLICATE_SIM_THRESHOLD", 0.95))

    cluster_ids = set()
    for lbl in labels:
        try:
            li = int(lbl)
            if li != -1:
                cluster_ids.add(li)
        except (ValueError, TypeError):
            continue
    cluster_ids = sorted(cluster_ids)
    logger.info("🔎 Extracting representatives for %d clusters (top_k=%d, lambda=%.2f)", len(cluster_ids), final_top_k, lambda_mult)

    semantic_model = _load_semantic_helper()
    semantic_embeddings = _get_semantic_embeddings(texts, embeddings, semantic_model)
    active_embeddings = semantic_embeddings if semantic_embeddings is not None else embeddings
    active_embeddings = _normalize_embeddings(np.asarray(active_embeddings)) if active_embeddings is not None else None

    reps: Dict[int, List[str]] = {}
    for cid in cluster_ids:
        idxs = [i for i, lbl in enumerate(labels) if str(lbl) == str(cid)]
        if not idxs:
            continue

        cluster_texts = [texts[i] for i in idxs]
        cluster_embeds = active_embeddings[idxs] if active_embeddings is not None else None
        centroid = _compute_cluster_centroid(cluster_embeds)

        if centroid is None:
            logger.warning("Centroid unavailable for cluster %s; using fallback", cid)
            reps[cid] = _fallback_nearest(cluster_texts, cluster_embeds, final_top_k)
            continue

        relevance = cluster_embeds @ centroid
        ordered_idx = list(relevance.argsort()[::-1])
        candidate_pool = [
            (i, cluster_texts[i], cluster_embeds[i])
            for i in ordered_idx
            if cluster_texts[i] and len(cluster_texts[i].strip()) >= 3
        ]

        if not candidate_pool:
            reps[cid] = _fallback_nearest(cluster_texts, cluster_embeds, final_top_k)
            continue

        candidate_size = min(len(candidate_pool), max(final_top_k * candidate_multiplier, final_top_k))
        candidate_pool = candidate_pool[:candidate_size]

        candidate_texts = [c[1] for c in candidate_pool]
        candidate_embs = np.vstack([c[2] for c in candidate_pool]) if cluster_embeds is not None else None

        selected = _mmr_select(
            candidate_texts,
            candidate_embs,
            centroid,
            top_k=final_top_k,
            lambda_mult=lambda_mult,
            duplicate_threshold=duplicate_threshold,
        )

        if not selected:
            selected = _fallback_nearest(candidate_texts, candidate_embs, final_top_k)

        reps[cid] = selected

    return reps

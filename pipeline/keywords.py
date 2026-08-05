from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional

import config
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from kiwipiepy import Kiwi
from pipeline.embedder import _get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tokenizer
kiwi = Kiwi()

# Internal KeyBERT cache
_kw_models: Dict[str, KeyBERT] = {}
EXTRA_STOPWORDS = {w.strip() for w in getattr(config, "KEYWORD_EXTRA_STOPWORDS", []) if w}
STOPWORD_SET = set(getattr(config, "KOREAN_STOPWORDS", [])) | EXTRA_STOPWORDS
ALLOWED_POS_TAGS = set(getattr(config, "ALLOWED_POS_TAGS_KO", set()))


def _normalize_token_form(token) -> Optional[str]:
    """Return a normalized surface form for keyword comparison."""

    base = getattr(token, "lemma", None) or getattr(token, "form", "")
    if not base:
        return None

    if base.endswith("하") and len(base) > 1:
        base = base[:-1]
    return base


def is_valid_keyword_token(token, normalized_form: Optional[str] = None, stopwords: Optional[set] = None) -> bool:
    """Check whether a token passes POS/stopword/regex filters for keyword use."""

    stopword_set = stopwords or STOPWORD_SET
    tag = getattr(token, "tag", "")
    if ALLOWED_POS_TAGS and tag not in ALLOWED_POS_TAGS:
        return False

    normalized_form = normalized_form or _normalize_token_form(token)
    if not normalized_form:
        return False

    forms_to_check = {getattr(token, "form", ""), normalized_form}
    lemma = getattr(token, "lemma", None)
    if lemma:
        forms_to_check.add(lemma)

    if any(form and form in stopword_set for form in forms_to_check):
        return False

    return bool(re.fullmatch(config.VALID_KEYWORD_RE, normalized_form))


def normalize_keyword_token(token, stopwords: Optional[set] = None) -> Optional[str]:
    """Normalize and validate a token, returning the cleaned keyword or None."""

    normalized_form = _normalize_token_form(token)
    if not is_valid_keyword_token(token, normalized_form=normalized_form, stopwords=stopwords):
        return None
    return normalized_form


def _tokenize_sentence(text: str, stopwords: Optional[set] = None) -> List[str]:
    """Tokenize a sentence with Kiwi and filter using POS/stopword rules."""

    tokens: List[str] = []
    for tok in kiwi.tokenize(text or ""):
        normalized = normalize_keyword_token(tok, stopwords=stopwords)
        if normalized:
            tokens.append(normalized)
    return tokens


def _resolve_keyword_model_name(model_name: Optional[str]) -> str:
    if model_name:
        return model_name
    semantic_cfg = getattr(config, "semantic", None)
    if semantic_cfg and getattr(semantic_cfg, "model", None):
        return semantic_cfg.model
    return getattr(config, "MODEL_NAME", "jhgan/ko-sbert-sts")


def _get_vectorizer() -> CountVectorizer:
    """
    Create a CountVectorizer using config-defined token pattern, ngram range, and stopwords.
    """
    return CountVectorizer(
        token_pattern=config.TOKEN_PATTERN,
        stop_words=list(STOPWORD_SET),
        ngram_range=config.KEYWORD_NGRAM_RANGE,
    )


def _sample_and_normalize(sents: List[str]) -> str:
    """
    Sample up to config.MAX_KEYWORD_DOCS sentences, normalize nouns/lemmas, and concatenate.
    """
    docs = sents.copy()
    if len(docs) > config.MAX_KEYWORD_DOCS:
        import random
        random.seed(config.RANDOM_SEED if hasattr(config, 'RANDOM_SEED') else 0)
        docs = random.sample(docs, config.MAX_KEYWORD_DOCS)
    tokens: List[str] = []
    weighted_tags = {"VA", "VV"}
    for sent in docs:
        for tok in kiwi.tokenize(sent or ""):
            normalized = normalize_keyword_token(tok, stopwords=STOPWORD_SET)
            if not normalized:
                continue

            weight = 2 if getattr(tok, "tag", "")[:2] in weighted_tags else 1
            tokens.extend([normalized] * weight)
    return " ".join(tokens)


def _get_kw_model(model_name: Optional[str]) -> KeyBERT:
    """Load and cache a KeyBERT model."""
    resolved = _resolve_keyword_model_name(model_name)
    if resolved not in _kw_models:
        logger.info("[KEYWORDS] loading KeyBERT model: %s", resolved)
        _kw_models[resolved] = KeyBERT(resolved)
    return _kw_models[resolved]


def _get_semantic_helper(model_name: Optional[str] = None, device: Optional[str] = None):
    """Load the semantic SBERT helper defined in config.semantic.*."""

    semantic_cfg = getattr(config, "semantic", None)
    resolved_model = model_name or getattr(semantic_cfg, "model", None) or getattr(config, "MODEL_NAME", None)
    resolved_device = device or getattr(semantic_cfg, "device", None)
    if not resolved_model:
        raise ValueError("Semantic helper model name is required for keyword extraction")
    return _get_model(resolved_model, resolved_device)


def _prepare_cluster_documents(cluster_reps: Dict[int, List[str]]) -> tuple[list[str], list[int]]:
    """Tokenize and filter cluster representative sentences for vectorization."""

    docs: list[str] = []
    labels: list[int] = []
    for cid, reps in cluster_reps.items():
        samples = reps
        if len(samples) > getattr(config, "KEYWORD_MAX_SENT", len(samples)):
            import random

            random.seed(getattr(config, "RANDOM_SEED", 0))
            samples = random.sample(samples, getattr(config, "KEYWORD_MAX_SENT", len(samples)))

        for sent in samples:
            tokens = _tokenize_sentence(sent, stopwords=STOPWORD_SET)
            if not tokens:
                continue
            docs.append(" ".join(tokens))
            labels.append(cid)
    return docs, labels


def _compute_cluster_centroid(texts: List[str], semantic_model) -> Optional[np.ndarray]:
    """Compute a normalized centroid embedding for a cluster using the semantic helper."""

    if not texts:
        return None
    try:
        embeddings = semantic_model.encode(
            texts,
            batch_size=getattr(config, "BATCH_SIZE", 64),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Semantic centroid encoding failed; skipping MMR. Error: %s", exc)
        return None

    if embeddings.size == 0:
        return None
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if not norm:
        return None
    return centroid / norm


def _mmr_with_centroid(
    candidate_terms: List[str],
    term_embeddings: np.ndarray,
    centroid: np.ndarray,
    top_k: int,
    diversity: float,
) -> List[str]:
    """Apply a simple MMR routine using centroid relevance and candidate diversity."""

    if not candidate_terms or term_embeddings.size == 0 or centroid is None:
        return []

    relevance = term_embeddings @ centroid
    selected: List[int] = []
    remaining = list(range(len(candidate_terms)))

    if not remaining:
        return []

    first_idx = int(np.argmax(relevance))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while remaining and len(selected) < top_k:
        selected_embs = term_embeddings[selected]
        similarity_to_selected = np.max(term_embeddings[remaining] @ selected_embs.T, axis=1)
        mmr_scores = (1 - diversity) * relevance[remaining] - diversity * similarity_to_selected
        best_pos = int(np.argmax(mmr_scores))
        selected.append(remaining[best_pos])
        remaining.pop(best_pos)

    return [candidate_terms[i] for i in selected][:top_k]


def _filter_global_common_terms(cluster_keywords: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Remove overly common terms that appear across many clusters."""

    total_clusters = len(cluster_keywords)
    if total_clusters <= 1:
        return cluster_keywords

    ratio = getattr(config, "GLOBAL_COMMON_TERMS_MAX_CLUSTER_RATIO", 1.0)
    counter: Counter = Counter()
    for kws in cluster_keywords.values():
        counter.update(set(kws))

    threshold = max(2, math.ceil(ratio * total_clusters))
    common_terms = {term for term, cnt in counter.items() if cnt >= threshold}

    if not common_terms:
        return cluster_keywords

    logger.info("Filtering %d global-common terms across clusters", len(common_terms))
    filtered: Dict[int, List[str]] = {}
    for cid, kws in cluster_keywords.items():
        filtered[cid] = [kw for kw in kws if kw not in common_terms]
    return filtered


def _extract_with_keybert(
    cluster_reps: Dict[int, List[str]],
    kw_model: KeyBERT,
    top_n: int,
) -> Dict[int, List[str]]:
    """Extract keywords using KeyBERT with MMR and POS/stopword filtering."""

    result: Dict[int, List[str]] = {}
    vectorizer = _get_vectorizer()
    multiplier = max(1, getattr(config, "KEYWORD_CANDIDATE_MULTIPLIER", 2))
    diversity = getattr(config, "KEYWORD_MMR_DIVERSITY", 0.5)

    for cid, reps in cluster_reps.items():
        doc = _sample_and_normalize(reps)
        if not doc:
            result[cid] = []
            continue
        try:
            candidates = kw_model.extract_keywords(
                doc,
                vectorizer=vectorizer,
                use_mmr=True,
                diversity=diversity,
                top_n=top_n,
                nr_candidates=max(top_n * multiplier, top_n),
            )
        except ValueError:
            logger.warning("KeyBERT failed for cluster %s; returning empty list", cid)
            result[cid] = []
            continue

        clean: List[str] = []
        for kw, _ in candidates:
            if kw not in clean:
                clean.append(kw)
            if len(clean) >= top_n:
                break
        result[cid] = clean

    return result


def _extract_with_ctfidf(
    cluster_reps: Dict[int, List[str]],
    top_n: int,
    semantic_model_name: Optional[str] = None,
) -> Dict[int, List[str]]:
    """Extract keywords using c-TF-IDF followed by semantic MMR filtering."""

    docs, labels = _prepare_cluster_documents(cluster_reps)
    result: Dict[int, List[str]] = {cid: [] for cid in cluster_reps}
    if not docs:
        logger.warning("No documents available for c-TF-IDF keyword extraction")
        return result

    vectorizer = CountVectorizer(
        token_pattern=config.TOKEN_PATTERN,
        stop_words=list(STOPWORD_SET),
        ngram_range=config.KEYWORD_NGRAM_RANGE,
    )
    X = vectorizer.fit_transform(docs)
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf = TfidfTransformer().fit_transform(X)
    terms = np.array(vectorizer.get_feature_names_out())

    semantic_model = _get_semantic_helper(model_name=semantic_model_name)
    candidate_pool_size = max(top_n * getattr(config, "KEYWORD_CANDIDATE_MULTIPLIER", 2),
                              getattr(config, "KEYWORD_MAX_PER_CLUSTER", top_n) * 3)

    unique_labels = set(labels)
    for cid in unique_labels:
        mask = np.array(labels) == cid
        if not mask.any():
            result[cid] = []
            continue

        cluster_tfidf = tfidf[mask].mean(axis=0).A1
        top_idx = cluster_tfidf.argsort()[::-1]
        candidate_terms = terms[top_idx][:candidate_pool_size].tolist()

        if not candidate_terms:
            result[cid] = []
            continue

        centroid = _compute_cluster_centroid(cluster_reps.get(cid, []), semantic_model)
        if centroid is None:
            logger.info("Skipping semantic MMR for cluster %s due to missing centroid", cid)
            result[cid] = candidate_terms[:top_n]
            continue

        try:
            term_embeddings = semantic_model.encode(
                candidate_terms,
                batch_size=getattr(config, "BATCH_SIZE", 64),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            selected = _mmr_with_centroid(
                candidate_terms,
                term_embeddings,
                centroid,
                top_k=top_n,
                diversity=getattr(config, "KEYWORD_MMR_DIVERSITY", 0.5),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Semantic MMR failed for cluster %s (%s); falling back to TF-IDF order", cid, exc)
            selected = []

        if not selected:
            selected = candidate_terms[:top_n]

        result[cid] = selected

    return result


def extract_keywords(
    cluster_reps: Dict[int, List[str]],
    model_name: Optional[str] = None,
    top_n: int = config.CLUSTER_NAME_TOPK,
) -> Dict[int, List[str]]:
    """
    Extract top_n keywords per cluster using either KeyBERT or c-TF-IDF.

    Branches on config.USE_KEYBERT.
    """

    max_per_cluster = getattr(config, "KEYWORD_MAX_PER_CLUSTER", top_n)
    desired_top_n = min(top_n or max_per_cluster, max_per_cluster)

    if config.USE_KEYBERT:
        kw_model = _get_kw_model(model_name)
        extracted = _extract_with_keybert(cluster_reps, kw_model=kw_model, top_n=desired_top_n)
    else:
        extracted = _extract_with_ctfidf(
            cluster_reps,
            top_n=min(desired_top_n, getattr(config, "KEYWORD_MAX_PER_CLUSTER", desired_top_n)),
            semantic_model_name=model_name,
        )

    return _filter_global_common_terms(extracted)

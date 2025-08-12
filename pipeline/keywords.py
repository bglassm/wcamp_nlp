from __future__ import annotations

import logging
import re
from typing import Dict, List

import config
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tokenizer
kiwi = Kiwi()

# Internal KeyBERT cache
_kw_model: KeyBERT | None = None


def _get_vectorizer() -> CountVectorizer:
    """
    Create a CountVectorizer using config-defined token pattern, ngram range, and stopwords.
    """
    return CountVectorizer(
        token_pattern=config.TOKEN_PATTERN,
        stop_words=config.KOREAN_STOPWORDS,
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
    for sent in docs:
        for tok in kiwi.tokenize(sent):
            if tok.tag in config.JOSA_EOMI_TAGS:
                continue
            form = tok.form
            # adjective stem correction
            if form.endswith('하') and form[:-1]:
                form = form[:-1]
            if re.fullmatch(config.VALID_KEYWORD_RE, form):
                tokens.append(form)
    return " ".join(tokens)


def _get_kw_model(model_name: str) -> KeyBERT:
    """Load and cache a KeyBERT model."""
    global _kw_model
    if _kw_model is None:
        logger.info("⬇️ Loading KeyBERT model: %s", model_name)
        _kw_model = KeyBERT(model_name)
    return _kw_model


def extract_keywords(
    cluster_reps: Dict[int, List[str]],
    model_name: str = config.MODEL_NAME,
    top_n: int = config.CLUSTER_NAME_TOPK,
) -> Dict[int, List[str]]:
    """
    Extract top_n keywords per cluster using either KeyBERT or c-TF-IDF.

    Branches on config.USE_KEYBERT.
    """
    result: Dict[int, List[str]] = {}

    if config.USE_KEYBERT:
        kw_model = _get_kw_model(model_name)
        vectorizer = _get_vectorizer()
        multiplier = config.KEYWORD_CANDIDATE_MULTIPLIER

        for cid, reps in cluster_reps.items():
            doc = _sample_and_normalize(reps)
            if not doc:
                result[cid] = []
                continue
            try:
                candidates = kw_model.extract_keywords(
                    doc,
                    vectorizer=vectorizer,
                    use_maxsum=True,
                    top_n=top_n * multiplier,
                )
            except ValueError:
                result[cid] = []
                continue
            clean: List[str] = []
            for kw, _ in candidates:
                if kw not in clean:
                    clean.append(kw)
                if len(clean) >= top_n:
                    break
            result[cid] = clean
    else:
        # c-TF-IDF fallback
        # Build corpus and labels for c-TF-IDF
        docs = []
        labels = []
        for cid, reps in cluster_reps.items():
            docs.extend(reps)
            labels.extend([cid] * len(reps))
        vectorizer = CountVectorizer(
            token_pattern=config.TOKEN_PATTERN,
            stop_words=config.KOREAN_STOPWORDS,
            ngram_range=config.KEYWORD_NGRAM_RANGE,
        )
        X = vectorizer.fit_transform(docs)
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidf = TfidfTransformer().fit_transform(X)
        terms = np.array(vectorizer.get_feature_names_out())
        for cid in set(labels):
            mask = np.array(labels) == cid
            if not mask.any():
                result[cid] = []
                continue
            # compute mean tf-idf per term for this cluster
            cluster_tfidf = tfidf[mask].mean(axis=0).A1
            top_idx = cluster_tfidf.argsort()[::-1][:top_n]
            kws = terms[top_idx].tolist()
            result[cid] = kws

    return result

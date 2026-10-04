from __future__ import annotations
import logging
import os
import time
from typing import List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# module-level SBERT model cache to avoid reloading across calls
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, device: Optional[str]) -> SentenceTransformer:
    """Load and cache an SBERT model by (model_name, device) key.
    Kept at module scope so pipeline.mergy and other callers can share the cache."""
    from sentence_transformers import SentenceTransformer

    cache_key = f"{model_name}@{device or 'auto'}"
    if cache_key not in _MODEL_CACHE:
        logger.info("[EMBED] loading SBERT model %s on %s", model_name, device or "auto")
        kwargs = {"device": device} if device else {}
        _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, **kwargs)
    return _MODEL_CACHE[cache_key]


class Embedder(ABC):
    """Base interface: convert a list of strings to a numpy embedding matrix."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        ...


class LocalEmbedder(Embedder):
    """Embed texts with a local SBERT model."""

    def __init__(self, cfg):
        import torch

        self.model_name = cfg.embed.model
        self.batch_size = cfg.embed.batch_size
        self.device = cfg.embed.device

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[EMBED] CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.model = _get_model(self.model_name, self.device)
        logger.info("[EMBED] LocalEmbedder ready: %s on %s", self.model_name, self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("embed() received an empty text list")

        logger.info("[EMBED] encoding %d sentences (batch=%d) with %s",
                    len(texts), self.batch_size, self.model_name)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
        logger.info("[EMBED] done: shape=%s", embeddings.shape)
        return embeddings


class OpenAIEmbedder(Embedder):
    """Embed texts via the OpenAI Embeddings API with batching and retry logic."""

    def __init__(self, cfg):
        from dotenv import load_dotenv
        from openai import OpenAI

        self.model_name = cfg.embed.model
        self.batch_size = cfg.embed.batch_size
        self.max_retries = cfg.embed.max_retries
        self.timeout_sec = cfg.embed.timeout_sec
        self.api_base = cfg.embed.api_base

        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=api_key, base_url=self.api_base, timeout=self.timeout_sec)
        logger.info("[EMBED] OpenAIEmbedder ready: %s", self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        from openai import APIError, APIConnectionError, RateLimitError

        all_embeddings = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            batch_idx = i // self.batch_size + 1
            logger.info("[EMBED] batch %d/%d size=%d", batch_idx, n_batches, len(batch))

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(model=self.model_name, input=batch)
                    all_embeddings.extend([d.embedding for d in response.data])
                    break

                except RateLimitError as e:
                    delay = 2 ** attempt + 1
                    logger.warning("[EMBED] rate limit, retry %d/%d in %ds: %s",
                                   attempt + 1, self.max_retries, delay, e)
                    if attempt + 1 == self.max_retries:
                        raise
                    time.sleep(delay)

                except (APIError, APIConnectionError) as e:
                    delay = 2 ** attempt + 1
                    logger.warning("[EMBED] API error, retry %d/%d in %ds: %s",
                                   attempt + 1, self.max_retries, delay, e)
                    if attempt + 1 == self.max_retries:
                        raise
                    time.sleep(delay)

                except Exception:
                    logger.exception("[EMBED] unexpected error during embedding")
                    raise

        result = np.array(all_embeddings, dtype=np.float32)
        logger.info("[EMBED] all batches done: shape=%s", result.shape)
        return result


class CachingEmbedder(Embedder):
    """Wrap any Embedder with parquet-based per-product caching.
    Only clauses not yet in cache are sent to the underlying embedder."""

    def __init__(self, embedder: Embedder, cfg):
        self.embedder = embedder
        self.cfg = cfg
        self.cache_dir = Path(cfg.embed.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # V2 never trusts an old ID-only cache, including same-width stale vectors.
        key_parts = {
            "schema": 2, "backend": cfg.embed.backend, "model": cfg.embed.model,
            "batch_size": cfg.embed.batch_size,
            "endpoint": cfg.embed.device if cfg.embed.backend == "local" else cfg.embed.api_base,
        }
        self.cache_key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()[:16]
        logger.info("[EMBED] cache key: %s", self.cache_key)

    def _get_cache_path(self, product_id: str) -> Path:
        safe_product = hashlib.sha256(str(product_id).encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe_product}_{self.cache_key}.parquet"

    def embed(self, texts: List[str], clause_ids: List[str], product_id: str) -> np.ndarray:
        """Cache by (clause ID, text digest), returning exactly one vector per input row."""
        if len(texts) != len(clause_ids):
            raise ValueError("texts and clause_ids must have equal lengths")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")
        input_df = pd.DataFrame({
            "clause_id": [str(cid) for cid in clause_ids],
            "text_hash": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
        })
        if (input_df.groupby("clause_id")["text_hash"].nunique() > 1).any():
            raise ValueError("One clause ID refers to different texts within the same batch")
        keys = ["clause_id", "text_hash"]
        cache_path = self._get_cache_path(product_id)
        cached_df = pd.DataFrame(columns=keys + ["embedding_vector"])
        if cache_path.exists():
            try:
                candidate = pd.read_parquet(cache_path)
                if not set(cached_df.columns).issubset(candidate.columns):
                    raise ValueError("Unsupported cache schema")
                if candidate.duplicated(keys).any():
                    raise ValueError("Duplicate composite cache keys")
                cached_df = candidate[keys + ["embedding_vector"]]
            except Exception as exc:
                logger.warning("[EMBED] invalid cache, rebuilding: %s", exc)
        cache = {tuple(row[:2]): row[2] for row in cached_df.itertuples(index=False, name=None)}
        input_keys = list(input_df.itertuples(index=False, name=None))
        missing = {}
        for key, text in zip(input_keys, texts):
            if key not in cache:
                missing.setdefault(key, text)
        logger.info("[EMBED] cache: %d hit rows, %d unique misses", sum(k in cache for k in input_keys), len(missing))
        if missing:
            vectors = np.asarray(self.embedder.embed(list(missing.values())), dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(missing) or not np.isfinite(vectors).all():
                raise ValueError("Embedder must return one finite vector per unique clause")
            cache.update(zip(missing, vectors.tolist()))
        result = np.asarray([cache[key] for key in input_keys], dtype=np.float32)
        if result.ndim != 2 or not np.isfinite(result).all():
            raise ValueError("Cached vectors have inconsistent dimensions or non-finite values")
        if missing:
            updated = pd.DataFrame([(cid, digest, vec) for (cid, digest), vec in cache.items()],
                                   columns=keys + ["embedding_vector"])
            # Atomic replace prevents interrupted writes from leaving a partial parquet file.
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                updated.to_parquet(tmp_path, index=False)
                tmp_path.replace(cache_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        logger.info("[EMBED] final shape: %s", result.shape)
        return result


def get_embedder(cfg) -> CachingEmbedder:
    """Factory: instantiate the configured backend and wrap it with CachingEmbedder."""
    backend = cfg.embed.backend.lower()
    if backend == "local":
        base = LocalEmbedder(cfg)
    elif backend == "openai":
        base = OpenAIEmbedder(cfg)
    else:
        raise ValueError(f"Unsupported embedding backend: '{backend}'")
    return CachingEmbedder(base, cfg)

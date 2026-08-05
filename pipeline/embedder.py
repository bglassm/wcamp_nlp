import logging
import os
import time
from typing import List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# module-level SBERT model cache to avoid reloading across calls
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, device: Optional[str]) -> SentenceTransformer:
    """Load and cache an SBERT model by (model_name, device) key.
    Kept at module scope so pipeline.mergy and other callers can share the cache."""
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

        # cache key encodes backend + model + batch_size + device/api_base
        key_parts = [
            cfg.embed.backend,
            cfg.embed.model,
            str(cfg.embed.batch_size),
            cfg.embed.device if cfg.embed.backend == "local" else cfg.embed.api_base,
        ]
        self.cache_key = hashlib.sha256("-".join(key_parts).encode()).hexdigest()[:16]
        logger.info("[EMBED] cache key: %s", self.cache_key)

    def _get_cache_path(self, product_id: str) -> Path:
        return self.cache_dir / f"{product_id}_{self.cache_key}.parquet"

    def embed(self, texts: List[str], clause_ids: List[str], product_id: str) -> np.ndarray:
        """Return embeddings for texts, using cached vectors where available."""
        if not texts:
            return np.array([])

        cache_path = self._get_cache_path(product_id)

        cached_df = pd.DataFrame()
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                logger.info("[EMBED] cache hit: %s (%d rows)", cache_path.name, len(cached_df))
            except Exception as e:
                logger.warning("[EMBED] cache load failed (%s), rebuilding: %s", cache_path.name, e)
                cached_df = pd.DataFrame()

        input_df = pd.DataFrame({"clause_id": clause_ids, "text": texts})

        if not cached_df.empty:
            cached_ids = set(cached_df["clause_id"])
            to_embed_df = input_df[~input_df["clause_id"].isin(cached_ids)].copy()
            embedded_df = input_df[input_df["clause_id"].isin(cached_ids)].merge(
                cached_df, on="clause_id", how="left"
            )
            logger.info("[EMBED] cache: %d hit, %d miss", len(embedded_df), len(to_embed_df))
        else:
            to_embed_df = input_df.copy()
            embedded_df = pd.DataFrame()
            logger.info("[EMBED] cache empty, embedding %d items", len(to_embed_df))

        if not to_embed_df.empty:
            new_embeddings = self.embedder.embed(to_embed_df["text"].tolist())
            to_embed_df = to_embed_df.copy()
            to_embed_df["embedding_vector"] = new_embeddings.tolist()
            updated_cache = pd.concat([cached_df, to_embed_df[["clause_id", "embedding_vector"]]])
            updated_cache.to_parquet(cache_path, index=False)
            logger.info("[EMBED] cache saved: %s (%d rows)", cache_path.name, len(updated_cache))
            embedded_df = pd.concat([embedded_df, to_embed_df])

        final_df = input_df.merge(
            embedded_df[["clause_id", "embedding_vector"]], on="clause_id", how="left"
        )
        final_embeddings = np.array(final_df["embedding_vector"].tolist(), dtype=np.float32)
        logger.info("[EMBED] final shape: %s", final_embeddings.shape)
        return final_embeddings


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

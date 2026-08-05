import logging
import os
import time
from typing import List, Union, Protocol, Optional
import openai
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import hashlib
import pickle
import config # Assuming config is available in the environment

# Internal cache to reuse loaded models
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, device: Optional[str]) -> SentenceTransformer:
    """Backward-compatible helper to lazily load and cache SBERT models.

    The merge/keyword modules historically imported ``_get_model`` directly.
    Instead of forcing every caller to instantiate ``LocalEmbedder`` (which also
    wraps caching and device selection) we keep exposing this lightweight helper
    so the legacy import continues to work.  The helper simply mirrors the logic
    from ``LocalEmbedder._get_model`` but lives at module scope so that
    ``pipeline.mergy`` and others can reuse the same cache without duplicating
    code.
    """

    cache_key = f"{model_name}@{device or 'auto'}"
    if cache_key not in _MODEL_CACHE:
        logger.info("⬇️ Loading SBERT model %s on %s", model_name, device or "auto")
        kwargs = {}
        if device:
            kwargs["device"] = device
        _MODEL_CACHE[cache_key] = SentenceTransformer(model_name, **kwargs)
    return _MODEL_CACHE[cache_key]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- 1. Embedder Interface ---

# Import config (will be available in the environment)
# import config # Assuming config is imported or available globally in the project

# --- 1. Embedder Interface ---

class Embedder(Protocol):
    """
    텍스트 리스트를 임베딩 벡터(numpy.ndarray)로 변환하는 인터페이스.
    """
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        주어진 텍스트 리스트를 임베딩합니다.
        """
        ...

# --- 2. LocalEmbedder (SBERT/MiniLM) ---

class LocalEmbedder(Embedder):
    """
    로컬 SBERT 모델을 사용하여 임베딩을 수행하는 클래스.
    기존 embedder.py의 로직을 캡슐화합니다.
    """
    def __init__(self, cfg):
        self.model_name = cfg.embed.model
        self.batch_size = cfg.embed.batch_size
        self.device = cfg.embed.device

        # Fallback to CPU if CUDA unavailable
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not detected; switching to CPU")
            self.device = "cpu"

        # Load / cache model
        self.model = _get_model(self.model_name, self.device)
        logger.info("✅ LocalEmbedder initialized with model: %s on %s", self.model_name, self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of review texts to SBERT embeddings (np.ndarray).
        """
        if not texts:
            raise ValueError("Input 'texts' is empty")

        # Encode
        logger.info("🔢 Encoding %s sentences (batch=%d) with %s...",
                    f"{len(texts):,}", self.batch_size, self.model_name)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )
        logger.info("✅ Embedding complete — shape: %s", embeddings.shape)
        return embeddings

# --- 3. OpenAIEmbedder ---

class OpenAIEmbedder(Embedder):
    """
    OpenAI API를 사용하여 임베딩을 수행하는 클래스.
    """
    def __init__(self, cfg):
        self.model_name = cfg.embed.model
        self.batch_size = cfg.embed.batch_size
        self.max_retries = cfg.embed.max_retries
        self.timeout_sec = cfg.embed.timeout_sec
        self.api_base = cfg.embed.api_base

        load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_base,
            timeout=self.timeout_sec
        )
        logger.info("✅ OpenAIEmbedder initialized with model: %s", self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        주어진 텍스트 리스트를 OpenAI API를 사용하여 임베딩합니다.
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        num_texts = len(texts)
        
        # Batching
        for i in range(0, num_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            logger.info("🔢 Encoding batch %d/%d (size=%d) with %s...",
                        i // self.batch_size + 1, 
                        (num_texts + self.batch_size - 1) // self.batch_size,
                        len(batch), self.model_name)

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    
                    # API 응답에서 임베딩 추출 및 순서 확인
                    batch_embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    logger.info("✅ Batch %d complete.", i // self.batch_size + 1)
                    break # 성공 시 retry 루프 탈출
                
                except RateLimitError as e:
                    delay = 2 ** attempt + 1 # Exponential backoff
                    logger.warning("⚠️ Rate limit exceeded (429). Retrying in %d seconds (Attempt %d/%d). Error: %s", 
                                   delay, attempt + 1, self.max_retries, e)
                    if attempt + 1 == self.max_retries:
                        logger.error("❌ Max retries reached for rate limit.")
                        raise
                    time.sleep(delay)
                
                except (APIError, APIConnectionError) as e:
                    delay = 2 ** attempt + 1
                    logger.warning("⚠️ API Error (500/503/Connection). Retrying in %d seconds (Attempt %d/%d). Error: %s", 
                                   delay, attempt + 1, self.max_retries, e)
                    if attempt + 1 == self.max_retries:
                        logger.error("❌ Max retries reached for API error.")
                        raise
                    time.sleep(delay)
                
                except Exception as e:
                    logger.error("❌ An unexpected error occurred during embedding: %s", e)
                    raise

        final_embeddings = np.array(all_embeddings, dtype=np.float32)
        logger.info("✅ All embedding complete — shape: %s", final_embeddings.shape)
        return final_embeddings

# --- 4. Factory Function ---

# --- 4. Caching Wrapper ---

class CachingEmbedder(Embedder):
    """
    임베딩 결과를 캐시 파일(parquet)에 저장하고 재사용하는 래퍼 클래스.
    """
    def __init__(self, embedder: Embedder, cfg):
        self.embedder = embedder
        self.cfg = cfg
        self.cache_dir = Path(cfg.embed.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 키 생성: backend + model + batch_size + device/api_base
        # 이 키가 변경되면 캐시를 무효화해야 함.
        cache_key_parts = [
            cfg.embed.backend,
            cfg.embed.model,
            str(cfg.embed.batch_size),
            cfg.embed.device if cfg.embed.backend == "local" else cfg.embed.api_base
        ]
        self.cache_key = hashlib.sha256("-".join(cache_key_parts).encode()).hexdigest()[:16]
        logger.info("Cache Key: %s", self.cache_key)

    def _get_cache_path(self, product_id: str) -> Path:
        """제품 ID와 캐시 키를 포함하는 캐시 파일 경로를 반환합니다."""
        return self.cache_dir / f"{product_id}_{self.cache_key}.parquet"

    def embed(self, texts: List[str], clause_ids: List[str], product_id: str) -> np.ndarray:
        """
        텍스트 리스트를 임베딩하고 캐시를 관리합니다.
        
        Args:
            texts: 임베딩할 텍스트 리스트.
            clause_ids: 각 텍스트에 대응하는 고유 ID (product_id, review_id, clause_idx).
            product_id: 캐시 파일 이름에 사용될 제품 ID.
            
        Returns:
            임베딩 벡터 numpy.ndarray.
        """
        if not texts:
            return np.array([])

        cache_path = self._get_cache_path(product_id)
        
        # 1. 기존 캐시 로드
        cached_df = pd.DataFrame()
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                logger.info("Cache loaded from %s (rows: %d)", cache_path, len(cached_df))
            except Exception as e:
                logger.warning("Failed to load cache from %s: %s. Rebuilding cache.", cache_path, e)
                cached_df = pd.DataFrame()

        # 2. 캐시된 ID와 임베딩할 ID 분리
        input_df = pd.DataFrame({"clause_id": clause_ids, "text": texts})
        
        if not cached_df.empty:
            # 캐시된 ID를 set으로 변환하여 빠르게 조회
            cached_ids = set(cached_df["clause_id"])
            
            # 캐시에 없는 항목만 필터링
            to_embed_df = input_df[~input_df["clause_id"].isin(cached_ids)].copy()
            
            # 캐시에 있는 항목은 임베딩을 가져옴
            embedded_df = input_df[input_df["clause_id"].isin(cached_ids)].merge(
                cached_df, on="clause_id", how="left"
            )
            logger.info("Found %d items in cache. %d items need embedding.", 
                        len(embedded_df), len(to_embed_df))
        else:
            to_embed_df = input_df.copy()
            embedded_df = pd.DataFrame()
            logger.info("Cache is empty. %d items need embedding.", len(to_embed_df))

        # 3. 미존재 항목만 API 호출
        if not to_embed_df.empty:
            new_embeddings = self.embedder.embed(to_embed_df["text"].tolist())
            
            # 4. 새로 생성된 벡터를 데이터프레임에 추가
            to_embed_df["embedding_vector"] = new_embeddings.tolist()
            
            # 5. 캐시 업데이트 및 저장
            new_cached_df = pd.concat([cached_df, to_embed_df[["clause_id", "embedding_vector"]]])
            new_cached_df.to_parquet(cache_path, index=False)
            logger.info("Cache updated and saved to %s (total rows: %d)", cache_path, len(new_cached_df))
            
            # 임베딩된 항목 병합
            embedded_df = pd.concat([embedded_df, to_embed_df])
        
        # 6. 최종 결과 정렬 및 반환
        # 입력 순서대로 정렬하기 위해 input_df와 병합
        final_df = input_df.merge(embedded_df[["clause_id", "embedding_vector"]], 
                                  on="clause_id", how="left")
        
        # embedding_vector 컬럼의 리스트를 numpy array로 변환
        final_embeddings = np.array(final_df["embedding_vector"].tolist(), dtype=np.float32)
        
        logger.info("✅ Final embedding array shape: %s", final_embeddings.shape)
        return final_embeddings

# --- 5. Factory Function ---

def get_embedder(cfg) -> CachingEmbedder:
    """
    설정(cfg)에 따라 적절한 Embedder 인스턴스를 생성하고 CachingEmbedder로 래핑하여 반환합니다.
    """
    backend = cfg.embed.backend.lower()
    
    if backend == "local":
        base_embedder = LocalEmbedder(cfg)
    elif backend == "openai":
        base_embedder = OpenAIEmbedder(cfg)
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
        
    return CachingEmbedder(base_embedder, cfg)

# --- 6. Backward Compatibility (Removed the old function) ---
# The original `embed_reviews` is removed as it's replaced by the factory pattern.
# Any code calling `embed_reviews` must be updated to use `get_embedder(cfg).embed(texts, clause_ids, product_id)`.
# This is a necessary refactoring step.

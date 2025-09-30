import logging
from typing import List, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Internal cache to reuse loaded models
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str, device: str) -> SentenceTransformer:
    """
    Load & cache the SBERT model; reuse across calls to avoid re-download.
    """
    key = f"{model_name}@{device}"
    if key not in _MODEL_CACHE:
        logger.info("⬇️ Loading SBERT model %s on %s", model_name, device)
        _MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[key]


def embed_reviews(
    texts: List[str],
    model_name: str = config.MODEL_NAME,
    batch_size: int = config.BATCH_SIZE,
    device: str = config.DEVICE,
) -> np.ndarray:
    """
    Convert a list of review texts to SBERT embeddings (np.ndarray).

    Defaults are taken from config.py.
    Raises ValueError if input is empty.
    """
    if not texts:
        raise ValueError("Input 'texts' is empty")

    # Fallback to CPU if CUDA unavailable
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not detected; switching to CPU")
        device = "cpu"

    # Load / cache model
    model = _get_model(model_name, device)

    # Encode
    logger.info("🔢 Encoding %s sentences (batch=%d)...", f"{len(texts):,}", batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=True,
    )
    logger.info("✅ Embedding complete — shape: %s", embeddings.shape)
    return embeddings

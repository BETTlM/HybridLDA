"""Sentence-transformer embeddings with on-disk caching."""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from src.config import CACHE_DIR, EMBEDDING_MODEL, RANDOM_SEED

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}


def _cache_path(corpus: str, model_name: str, n_docs: int, seed: int) -> str:
    key = f"{corpus}|{model_name}|{n_docs}|{seed}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(CACHE_DIR / f"emb_{corpus}_{digest}.npy")


def get_model(model_name: str = EMBEDDING_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def encode_documents(
    texts: list[str],
    corpus: str,
    model_name: str = EMBEDDING_MODEL,
    seed: int = RANDOM_SEED,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Return L2-normalized document embeddings, cached under data/cache."""
    path = _cache_path(corpus, model_name, len(texts), seed)
    try:
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    except OSError:
        pass

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(path, embeddings)
    logger.info("Saved embeddings %s shape=%s", path, embeddings.shape)
    return embeddings

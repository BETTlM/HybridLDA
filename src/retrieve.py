"""Query-to-topic and related-paper retrieval using hybrid vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import GAMMA
from src.embeddings import get_model
from src.hybrid import build_hybrid_matrix, l2_normalize
from src.lda_baseline import LDAResult
from src.preprocess import tokenize


def query_topic_distribution(query: str, lda: LDAResult) -> np.ndarray:
    bow = lda.dictionary.doc2bow(tokenize(query))
    dist = np.zeros(lda.n_topics, dtype=np.float32)
    for topic_id, prob in lda.model.get_document_topics(bow, minimum_probability=0.0):
        if topic_id < lda.n_topics:
            dist[topic_id] = prob
    total = dist.sum()
    if total > 0:
        dist /= total
    return dist


def query_embedding(query: str, model_name: str | None = None) -> np.ndarray:
    model = get_model(model_name) if model_name else get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def rank_topics_for_query(
    query: str,
    lda: LDAResult,
    topic_words: list[list[str]],
    embeddings: np.ndarray | None = None,
    doc_theta: np.ndarray | None = None,
    gamma: float = GAMMA,
    top_k: int = 5,
) -> pd.DataFrame:
    """Blend LDA query-topic probs with embedding similarity to topic centroids."""
    lda_dist = query_topic_distribution(query, lda)
    scores = lda_dist.copy()

    if embeddings is not None and doc_theta is not None:
        q_emb = query_embedding(query)
        centroids = []
        for k in range(lda.n_topics):
            weights = doc_theta[:, k]
            if weights.sum() <= 1e-8:
                centroids.append(np.zeros(embeddings.shape[1], dtype=np.float32))
            else:
                centroids.append(np.average(embeddings, axis=0, weights=weights))
        centroids = l2_normalize(np.vstack(centroids))
        emb_sim = cosine_similarity(q_emb, centroids).ravel()
        emb_sim = (emb_sim - emb_sim.min()) / (emb_sim.max() - emb_sim.min() + 1e-9)
        scores = gamma * lda_dist + (1.0 - gamma) * emb_sim.astype(np.float32)

    order = np.argsort(scores)[::-1][:top_k]
    rows = []
    for rank, k in enumerate(order, start=1):
        words = topic_words[k] if k < len(topic_words) else []
        rows.append(
            {
                "rank": rank,
                "topic_id": int(k),
                "score": float(scores[k]),
                "lda_prob": float(lda_dist[k]),
                "top_words": ", ".join(words),
            }
        )
    return pd.DataFrame(rows)


def related_documents(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    lda: LDAResult | None = None,
    gamma: float = GAMMA,
    top_k: int = 8,
) -> pd.DataFrame:
    """Return the nearest corpus papers to a query abstract (hybrid cosine)."""
    q_emb = query_embedding(query)
    if lda is not None:
        q_theta = query_topic_distribution(query, lda).reshape(1, -1)
        q_h = build_hybrid_matrix(q_theta, q_emb, gamma=gamma)
        doc_h = build_hybrid_matrix(lda.theta, embeddings, gamma=gamma)
        sims = cosine_similarity(q_h, doc_h).ravel()
    else:
        sims = cosine_similarity(q_emb, embeddings).ravel()

    idx = np.argsort(sims)[::-1][:top_k]
    out = df.iloc[idx].copy()
    out["similarity"] = sims[idx]
    cols = [c for c in ("title", "year", "label_name", "text", "similarity") if c in out.columns]
    return out[cols].reset_index(drop=True)

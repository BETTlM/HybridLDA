"""Coherence (C_v, NPMI, UMass), silhouette, and topic diversity."""

from __future__ import annotations

import logging

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import silhouette_score

from src.config import TOP_N_WORDS

logger = logging.getLogger(__name__)


def topic_coherence(
    topic_words: list[list[str]],
    texts: list[list[str]],
    dictionary: Dictionary | None = None,
    measures: tuple[str, ...] = ("c_v", "c_npmi", "u_mass"),
) -> dict[str, float]:
    """Compute gensim coherence scores for a list of top-word lists."""
    if dictionary is None:
        dictionary = Dictionary(texts)
    cleaned = []
    for words in topic_words:
        kept = [w for w in words if w in dictionary.token2id]
        if len(kept) >= 2:
            cleaned.append(kept)
    if len(cleaned) < 2:
        return {m: float("nan") for m in measures}

    scores: dict[str, float] = {}
    for measure in measures:
        try:
            cm = CoherenceModel(
                topics=cleaned,
                texts=texts,
                dictionary=dictionary,
                coherence=measure,
                topn=min(TOP_N_WORDS, min(len(t) for t in cleaned)),
                processes=1,
            )
            scores[measure] = float(cm.get_coherence())
        except Exception as exc:
            logger.warning("Coherence %s failed: %s", measure, exc)
            scores[measure] = float("nan")
    return scores


def clustering_silhouette(reduced: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(labels) <= len(unique):
        return float("nan")
    try:
        return float(silhouette_score(reduced, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def topic_diversity(topic_words: list[list[str]], top_n: int = TOP_N_WORDS) -> float:
    """Fraction of unique top words across topics (Dieng et al.)."""
    bag: list[str] = []
    for words in topic_words:
        bag.extend(words[:top_n])
    if not bag:
        return float("nan")
    return len(set(bag)) / len(bag)


def summarize_topics(topic_words: list[list[str]]) -> list[dict]:
    return [
        {"topic_id": i, "top_words": ", ".join(words)}
        for i, words in enumerate(topic_words)
    ]

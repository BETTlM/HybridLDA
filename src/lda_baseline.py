"""Baseline Latent Dirichlet Allocation (gensim)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from src.config import (
    LDA_CHUNKSIZE,
    LDA_ITERATIONS,
    LDA_PASSES,
    MAX_DF,
    MIN_DF,
    RANDOM_SEED,
    TOP_N_WORDS,
)


@dataclass
class LDAResult:
    model: LdaModel
    dictionary: Dictionary
    corpus: list
    theta: np.ndarray
    topic_words: list[list[str]]
    n_topics: int
    texts: list[list[str]] = field(default_factory=list)


def build_dictionary(texts: list[list[str]], min_df: int = MIN_DF, max_df: float = MAX_DF) -> Dictionary:
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=20000)
    dictionary.compactify()
    return dictionary


def fit_lda(
    texts: list[list[str]],
    n_topics: int,
    seed: int = RANDOM_SEED,
    passes: int = LDA_PASSES,
    iterations: int = LDA_ITERATIONS,
    top_n_words: int = TOP_N_WORDS,
) -> LDAResult:
    dictionary = build_dictionary(texts)
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=seed,
        passes=passes,
        iterations=iterations,
        chunksize=LDA_CHUNKSIZE,
        alpha="auto",
        eta="auto",
        eval_every=None,
    )
    theta = document_topic_matrix(model, corpus, n_topics)
    topic_words = extract_topic_words(model, n_topics, top_n_words)
    return LDAResult(
        model=model,
        dictionary=dictionary,
        corpus=corpus,
        theta=theta,
        topic_words=topic_words,
        n_topics=n_topics,
        texts=texts,
    )


def document_topic_matrix(model: LdaModel, corpus: list, n_topics: int) -> np.ndarray:
    theta = np.zeros((len(corpus), n_topics), dtype=np.float32)
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            if topic_id < n_topics:
                theta[i, topic_id] = prob
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return theta / row_sums


def extract_topic_words(model: LdaModel, n_topics: int, top_n: int = TOP_N_WORDS) -> list[list[str]]:
    topics: list[list[str]] = []
    for k in range(n_topics):
        words = [w for w, _ in model.show_topic(k, topn=top_n)]
        topics.append(words)
    return topics

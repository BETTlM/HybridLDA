"""Hybrid LDA + embedding vectors, dimensionality reduction, clustering, c-TF-IDF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.config import (
    GAMMA,
    PCA_COMPONENTS,
    RANDOM_SEED,
    TOP_N_WORDS,
    TSNE_COMPONENTS,
    UMAP_COMPONENTS,
    UMAP_N_NEIGHBORS,
)


@dataclass
class ClusterResult:
    labels: np.ndarray
    reduced: np.ndarray
    topic_words: list[list[str]]
    reducer: str
    n_topics: int
    hybrid: np.ndarray | None = None


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(matrix, dtype=np.float32), norm="l2", axis=1)


def build_hybrid_matrix(theta: np.ndarray, embeddings: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Concatenate γ·θ (LDA) with (1−γ)·e (embeddings). γ is α in the project abstract."""
    theta_n = l2_normalize(theta)
    emb_n = l2_normalize(embeddings)
    return np.hstack([gamma * theta_n, (1.0 - gamma) * emb_n]).astype(np.float32)


def reduce_dimensions(
    matrix: np.ndarray,
    method: str = "umap",
    n_components: int | None = None,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    method = method.lower()
    n = len(matrix)
    if method == "umap":
        import umap

        n_comp = n_components or UMAP_COMPONENTS
        n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n - 1))
        kwargs = dict(
            n_components=n_comp,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
            verbose=False,
        )
        try:
            reducer = umap.UMAP(n_jobs=1, **kwargs)
        except TypeError:
            reducer = umap.UMAP(**kwargs)
        return reducer.fit_transform(matrix)
    if method == "pca":
        n_comp = min(n_components or PCA_COMPONENTS, matrix.shape[1], n - 1)
        return PCA(n_components=n_comp, random_state=seed).fit_transform(matrix)
    if method in {"tsne", "t-sne"}:
        n_comp = n_components or TSNE_COMPONENTS
        perplexity = min(30, max(5, (n - 1) // 3))
        return TSNE(
            n_components=n_comp,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(matrix)
    raise ValueError(f"Unknown reducer '{method}'")


def cluster_kmeans(matrix: np.ndarray, n_topics: int, seed: int = RANDOM_SEED) -> np.ndarray:
    n_clusters = min(n_topics, len(matrix))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return model.fit_predict(matrix)


def cluster_topic_words(
    tokenized_docs: list[list[str]],
    labels: np.ndarray,
    n_topics: int,
    top_n: int = TOP_N_WORDS,
) -> list[list[str]]:
    """Class-based TF-IDF (BERTopic-style) over cluster-concatenated documents."""
    cluster_texts = []
    for k in range(n_topics):
        words: list[str] = []
        for tokens, lab in zip(tokenized_docs, labels):
            if int(lab) == k:
                words.extend(tokens)
        cluster_texts.append(" ".join(words) if words else "empty")

    vectorizer = CountVectorizer(max_features=20000, token_pattern=r"(?u)\b\w+\b")
    tf = vectorizer.fit_transform(cluster_texts).toarray().astype(np.float64)
    df = (tf > 0).sum(axis=0)
    idf = np.log(1.0 + (tf.shape[0] / (df + 1e-9)))
    ctfidf = tf * idf
    vocab = np.array(vectorizer.get_feature_names_out())

    topics: list[list[str]] = []
    for k in range(n_topics):
        if ctfidf[k].sum() == 0:
            topics.append(["empty"] * min(top_n, 1))
            continue
        idx = np.argsort(ctfidf[k])[::-1][:top_n]
        topics.append([str(w) for w in vocab[idx]])
    return topics


def fit_cluster_topics(
    features: np.ndarray,
    tokenized_docs: list[list[str]],
    n_topics: int,
    reducer: str = "umap",
    seed: int = RANDOM_SEED,
    top_n: int = TOP_N_WORDS,
) -> ClusterResult:
    reduced = reduce_dimensions(features, method=reducer, seed=seed)
    labels = cluster_kmeans(reduced, n_topics=n_topics, seed=seed)
    words = cluster_topic_words(tokenized_docs, labels, n_topics=n_topics, top_n=top_n)
    return ClusterResult(
        labels=labels,
        reduced=reduced,
        topic_words=words,
        reducer=reducer,
        n_topics=n_topics,
        hybrid=features,
    )

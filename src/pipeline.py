"""End-to-end experiment: baseline LDA vs BERT vs hybrid, with coherence."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import (
    ALPHA_GRID,
    DEFAULT_K_GRID,
    DEFAULT_N_DOCS,
    EMBEDDING_MODEL,
    FIGURES_DIR,
    GAMMA,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.data import load_corpus
from src.embeddings import encode_documents
from src.evaluate import clustering_silhouette, summarize_topics, topic_coherence, topic_diversity
from src.hybrid import build_hybrid_matrix, fit_cluster_topics, reduce_dimensions
from src.lda_baseline import build_dictionary, fit_lda
from src.preprocess import preprocess_frame
from src.temporal import topic_prevalence_by_year
from src.visualize import (
    plot_alpha_sweep,
    plot_coherence_vs_k,
    plot_model_comparison_bars,
    plot_silhouette_reducers,
    plot_temporal_trends,
    plot_umap_clusters,
)

logger = logging.getLogger(__name__)


def _row(corpus: str, model: str, k: int, reducer: str, coh: dict, sil: float, div: float, alpha: float) -> dict:
    return {
        "corpus": corpus,
        "model": model,
        "n_topics": k,
        "reducer": reducer,
        "alpha": alpha,
        "c_v": coh.get("c_v"),
        "c_npmi": coh.get("c_npmi"),
        "u_mass": coh.get("u_mass"),
        "silhouette": sil,
        "diversity": div,
    }


def run_corpus(
    corpus: str,
    n_docs: int = DEFAULT_N_DOCS,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    embedding_model: str = EMBEDDING_MODEL,
    gamma: float = GAMMA,
    seed: int = RANDOM_SEED,
    reducers_for_silhouette: tuple[str, ...] = ("pca", "tsne", "umap"),
) -> dict:
    logger.info("=== Corpus %s | n_docs=%s | K=%s ===", corpus, n_docs, k_grid)
    raw = load_corpus(corpus, n_docs=n_docs, seed=seed)
    df = preprocess_frame(raw)
    texts = df["tokens"].tolist()
    dictionary = build_dictionary(texts)

    embeddings = encode_documents(
        df["text"].fillna("").tolist(),
        corpus=corpus,
        model_name=embedding_model,
        seed=seed,
    )

    metric_rows: list[dict] = []
    sil_rows: list[dict] = []
    best_hybrid = {"c_v": -np.inf}
    lda_by_k: dict[int, object] = {}

    for k in k_grid:
        logger.info("[%s] Fitting LDA with K=%s", corpus, k)
        lda = fit_lda(texts, n_topics=k, seed=seed)
        lda_by_k[k] = lda
        lda_coh = topic_coherence(lda.topic_words, texts, dictionary=lda.dictionary)
        lda_div = topic_diversity(lda.topic_words)
        metric_rows.append(_row(corpus, "LDA", k, "none", lda_coh, float("nan"), lda_div, 1.0))

        logger.info("[%s] BERT-only clustering K=%s", corpus, k)
        bert = fit_cluster_topics(embeddings, texts, n_topics=k, reducer="umap", seed=seed)
        bert_coh = topic_coherence(bert.topic_words, texts, dictionary=dictionary)
        bert_sil = clustering_silhouette(bert.reduced, bert.labels)
        bert_div = topic_diversity(bert.topic_words)
        metric_rows.append(_row(corpus, "BERT", k, "umap", bert_coh, bert_sil, bert_div, 0.0))

        logger.info("[%s] Hybrid BERT-LDA clustering K=%s α=%s", corpus, k, gamma)
        hybrid = build_hybrid_matrix(lda.theta, embeddings, gamma=gamma)
        hyb = fit_cluster_topics(hybrid, texts, n_topics=k, reducer="umap", seed=seed)
        hyb_coh = topic_coherence(hyb.topic_words, texts, dictionary=dictionary)
        hyb_sil = clustering_silhouette(hyb.reduced, hyb.labels)
        hyb_div = topic_diversity(hyb.topic_words)
        metric_rows.append(_row(corpus, "Hybrid-BERT-LDA", k, "umap", hyb_coh, hyb_sil, hyb_div, gamma))

        cv = hyb_coh.get("c_v") or -np.inf
        if cv > best_hybrid["c_v"]:
            best_hybrid = {
                "c_v": cv,
                "k": k,
                "lda": lda,
                "hybrid_matrix": hybrid,
                "cluster": hyb,
                "topic_words": hyb.topic_words,
                "coherence": hyb_coh,
            }

    metrics = pd.DataFrame(metric_rows)
    best_k = int(best_hybrid["k"])
    lda = best_hybrid["lda"]
    hybrid_matrix = best_hybrid["hybrid_matrix"]
    hyb = best_hybrid["cluster"]

    logger.info("[%s] Silhouette comparison across reducers at K=%s", corpus, best_k)
    feature_sets = {
        "LDA": lda.theta,
        "BERT": embeddings,
        "Hybrid-BERT-LDA": hybrid_matrix,
    }
    for model_name, feats in feature_sets.items():
        for reducer in reducers_for_silhouette:
            try:
                reduced = reduce_dimensions(feats, method=reducer, seed=seed)
                from src.hybrid import cluster_kmeans

                labels = cluster_kmeans(reduced, n_topics=best_k, seed=seed)
                sil = clustering_silhouette(reduced, labels)
            except Exception as exc:
                logger.warning("Reducer %s/%s failed: %s", model_name, reducer, exc)
                sil = float("nan")
            sil_rows.append(
                {
                    "corpus": corpus,
                    "model": model_name,
                    "reducer": reducer,
                    "n_topics": best_k,
                    "silhouette": sil,
                }
            )
    sil_df = pd.DataFrame(sil_rows)

    logger.info("[%s] α sweep at K=%s", corpus, best_k)
    alpha_rows = []
    best_alpha = gamma
    best_alpha_cv = -np.inf
    best_alpha_cluster = hyb
    best_alpha_matrix = hybrid_matrix
    for alpha in ALPHA_GRID:
        h = build_hybrid_matrix(lda.theta, embeddings, gamma=alpha)
        clustered = fit_cluster_topics(h, texts, n_topics=best_k, reducer="umap", seed=seed)
        coh = topic_coherence(clustered.topic_words, texts, dictionary=dictionary)
        cv = coh.get("c_v") if coh.get("c_v") is not None else float("nan")
        alpha_rows.append(
            {
                "corpus": corpus,
                "alpha": alpha,
                "n_topics": best_k,
                "c_v": cv,
                "c_npmi": coh.get("c_npmi"),
                "u_mass": coh.get("u_mass"),
                "diversity": topic_diversity(clustered.topic_words),
                "silhouette": clustering_silhouette(clustered.reduced, clustered.labels),
            }
        )
        if pd.notna(cv) and cv > best_alpha_cv:
            best_alpha_cv = cv
            best_alpha = alpha
            best_alpha_cluster = clustered
            best_alpha_matrix = h
    alpha_df = pd.DataFrame(alpha_rows)

    vis_2d = reduce_dimensions(best_alpha_matrix, method="umap", n_components=2, seed=seed)
    plot_umap_clusters(
        vis_2d,
        best_alpha_cluster.labels,
        corpus,
        title=f"Hybrid UMAP clusters ({corpus}, K={best_k}, α={best_alpha})",
    )

    trend_df = pd.DataFrame()
    if "year" in df.columns and df["year"].notna().sum() >= 5:
        trend_df = topic_prevalence_by_year(df["year"], lda.theta, lda.topic_words)
        if not trend_df.empty:
            plot_temporal_trends(trend_df, corpus)
            trend_df.to_csv(RESULTS_DIR / f"temporal_{corpus}.csv", index=False)

    topics_payload = {
        "corpus": corpus,
        "n_docs": int(len(df)),
        "best_k": best_k,
        "best_alpha": best_alpha,
        "embedding_model": embedding_model,
        "lda_topics": summarize_topics(lda.topic_words),
        "hybrid_topics": summarize_topics(best_alpha_cluster.topic_words),
        "lda_coherence": topic_coherence(lda.topic_words, texts, dictionary=lda.dictionary),
        "hybrid_coherence": topic_coherence(
            best_alpha_cluster.topic_words, texts, dictionary=dictionary
        ),
    }
    (RESULTS_DIR / f"topics_{corpus}.json").write_text(json.dumps(topics_payload, indent=2))

    sample = df[["doc_id", "title", "text", "year", "label_name"]].copy()
    sample["hybrid_topic"] = best_alpha_cluster.labels
    sample["lda_topic"] = np.argmax(lda.theta, axis=1)
    sample.to_csv(RESULTS_DIR / f"docs_{corpus}.csv", index=False)

    np.savez_compressed(
        RESULTS_DIR / f"arrays_{corpus}.npz",
        embeddings=embeddings.astype(np.float32),
        theta=lda.theta.astype(np.float32),
        umap2d=vis_2d.astype(np.float32),
        hybrid_labels=best_alpha_cluster.labels.astype(np.int16),
    )

    return {
        "metrics": metrics,
        "silhouette": sil_df,
        "alpha": alpha_df,
        "topics": topics_payload,
        "trend": trend_df,
        "df": df,
        "lda": lda,
        "embeddings": embeddings,
        "best_k": best_k,
        "best_alpha": best_alpha,
    }


def run_experiments(
    corpora: tuple[str, ...] = ("20newsgroups", "arxiv"),
    n_docs: int = DEFAULT_N_DOCS,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    embedding_model: str = EMBEDDING_MODEL,
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics, all_sil, all_alpha = [], [], []
    demo = {"corpora": {}}

    for corpus in corpora:
        out = run_corpus(
            corpus,
            n_docs=n_docs,
            k_grid=k_grid,
            embedding_model=embedding_model,
        )
        all_metrics.append(out["metrics"])
        all_sil.append(out["silhouette"])
        all_alpha.append(out["alpha"])
        demo["corpora"][corpus] = {
            "best_k": out["best_k"],
            "best_alpha": out["best_alpha"],
            "topics": out["topics"],
        }

    metrics = pd.concat(all_metrics, ignore_index=True)
    sil_df = pd.concat(all_sil, ignore_index=True)
    alpha_df = pd.concat(all_alpha, ignore_index=True)

    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    sil_df.to_csv(RESULTS_DIR / "silhouette.csv", index=False)
    alpha_df.to_csv(RESULTS_DIR / "alpha_sweep.csv", index=False)

    for corpus in corpora:
        plot_coherence_vs_k(metrics, corpus, "c_v")
        plot_coherence_vs_k(metrics, corpus, "c_npmi")
        plot_model_comparison_bars(metrics, corpus, "c_v")
        plot_silhouette_reducers(sil_df, corpus)
        plot_alpha_sweep(alpha_df, corpus)

    demo["metrics_preview"] = metrics.to_dict(orient="records")
    (RESULTS_DIR / "demo_metrics.json").write_text(json.dumps(demo, indent=2, default=str))
    logger.info("Wrote metrics to %s", RESULTS_DIR / "metrics.csv")
    return metrics

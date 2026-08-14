"""Matplotlib / seaborn figures for coherence, silhouette, UMAP, and trends."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coherence_vs_k(metrics: pd.DataFrame, corpus: str, measure: str = "c_v") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    subset = metrics[metrics["corpus"] == corpus]
    sns.lineplot(data=subset, x="n_topics", y=measure, hue="model", marker="o", ax=ax)
    ax.set_title(f"{measure} coherence vs number of topics ({corpus})")
    ax.set_xlabel("Number of topics K")
    ax.set_ylabel(measure)
    return _save(fig, f"coherence_{measure}_{corpus}.png")


def plot_model_comparison_bars(metrics: pd.DataFrame, corpus: str, measure: str = "c_v") -> Path:
    subset = metrics[metrics["corpus"] == corpus]
    best = (
        subset.sort_values(measure, ascending=False)
        .groupby("model", as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=best, x="model", y=measure, ax=ax, hue="model", legend=False)
    ax.set_title(f"Best-{measure} by model ({corpus})")
    ax.set_ylabel(measure)
    ax.set_xlabel("")
    for i, row in enumerate(best.itertuples()):
        val = getattr(row, measure)
        if pd.notna(val):
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    return _save(fig, f"best_{measure}_{corpus}.png")


def plot_silhouette_reducers(sil_df: pd.DataFrame, corpus: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = sil_df[sil_df["corpus"] == corpus]
    sns.barplot(data=subset, x="reducer", y="silhouette", hue="model", ax=ax)
    ax.set_title(f"Silhouette by reducer ({corpus})")
    ax.set_ylim(0, max(0.6, float(subset["silhouette"].max()) + 0.05) if len(subset) else 1)
    return _save(fig, f"silhouette_{corpus}.png")


def plot_alpha_sweep(alpha_df: pd.DataFrame, corpus: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = alpha_df[alpha_df["corpus"] == corpus]
    sns.lineplot(data=subset, x="alpha", y="c_v", marker="o", ax=ax)
    ax.set_title(f"Hybrid α sweep (coherence C_v) — {corpus}")
    ax.set_xlabel("α (weight on LDA topic vector)")
    ax.set_ylabel("C_v")
    return _save(fig, f"alpha_sweep_{corpus}.png")


def plot_umap_clusters(coords: np.ndarray, labels: np.ndarray, corpus: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=12, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.colorbar(scatter, ax=ax, label="topic")
    return _save(fig, f"umap_{corpus}.png")


def plot_temporal_trends(trend_df: pd.DataFrame, corpus: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend_df, x="year", y="prevalence", hue="topic", marker="o", ax=ax)
    ax.set_title(f"Topic prevalence over time ({corpus})")
    ax.set_ylabel("Mean document-topic probability")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="Topic")
    return _save(fig, f"temporal_{corpus}.png")

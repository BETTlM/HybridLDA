"""Temporal topic prevalence: probability mass by publication year."""

from __future__ import annotations

import numpy as np
import pandas as pd


def topic_prevalence_by_year(
    years: pd.Series,
    theta: np.ndarray,
    topic_words: list[list[str]] | None = None,
) -> pd.DataFrame:
    """
    For each year, average the document-topic distribution (LDA θ or a
    one-hot / soft cluster assignment). Mirrors the abstract's 'temporal
    probability drift' output.
    """
    years_num = pd.to_numeric(years, errors="coerce")
    mask = years_num.notna()
    if mask.sum() < 5:
        return pd.DataFrame(columns=["year", "topic", "prevalence", "label"])

    theta = np.asarray(theta, dtype=float)
    n_topics = theta.shape[1]
    work = pd.DataFrame(
        {
            "year": years_num.astype("Int64"),
            "idx": np.arange(len(years_num)),
        }
    ).dropna(subset=["year"])
    rows = []
    for year, group in work.groupby("year", sort=True):
        mean_p = theta[group["idx"].to_numpy()].mean(axis=0)
        for k in range(n_topics):
            label = f"T{k}"
            if topic_words and k < len(topic_words) and topic_words[k]:
                label = f"T{k}: {', '.join(topic_words[k][:3])}"
            rows.append(
                {
                    "year": int(year),
                    "topic": k,
                    "prevalence": float(mean_p[k]),
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def labels_to_theta(labels: np.ndarray, n_topics: int) -> np.ndarray:
    """Hard cluster assignments as a one-hot document-topic matrix."""
    theta = np.zeros((len(labels), n_topics), dtype=float)
    for i, lab in enumerate(labels):
        k = int(lab)
        if 0 <= k < n_topics:
            theta[i, k] = 1.0
    return theta

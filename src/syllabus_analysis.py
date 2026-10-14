"""Unified syllabus analysis runner.

Orchestrates all syllabus-aligned probabilistic reasoning modules:
  - Bayesian Network analysis (BN structure, exact/approximate inference, Gibbs LDA)
  - Hidden Markov Model (temporal topic dynamics)
  - Kalman Filter (topic prevalence smoothing)
  - Markov Random Field (topic co-occurrence, message passing)
  - Decision Network (research planning under uncertainty)

This module integrates the new syllabus modules into the existing pipeline,
providing a single entry point that the main pipeline and Streamlit app can call.

Course: 22AIE301 — Probability and Reasoning
Syllabus Topics Covered:
  1. Uncertain Knowledge Representation
  2. Introduction to Bayesian Networks
  3. Representation Learning in Bayesian Networks
  4. Inference in Bayesian Networks
  5. Exact and Approximate Inference
  6. Markov Chain–Monte Carlo Method
  7. Hidden Markov Models
  8. Kalman Filter
  9. Markov Random Fields & Markov Networks
 10. Message Passing
 11. Learning in Markov Networks
 12. Decision Networks
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RESULTS_DIR

logger = logging.getLogger(__name__)


def run_syllabus_analysis(
    texts: list[list[str]],
    theta: np.ndarray,
    topic_words: list[list[str]],
    dictionary,
    n_topics: int,
    corpus: str,
    years: pd.Series | None = None,
    coherence_scores: dict | None = None,
    trend_df: pd.DataFrame | None = None,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run all syllabus-aligned analyses and save results.

    Parameters
    ----------
    texts : list of token lists (preprocessed documents)
    theta : document-topic matrix from LDA (n_docs x n_topics)
    topic_words : list of top-word lists per topic
    dictionary : gensim Dictionary
    n_topics : number of topics
    corpus : corpus name (for file naming)
    years : publication years (optional, for HMM/Kalman)
    coherence_scores : dict with 'c_v', 'c_npmi', etc.
    trend_df : temporal prevalence DataFrame
    seed : random seed

    Returns
    -------
    dict with keys for each analysis module's results
    """
    results = {}

    # ── 1. Bayesian Network Analysis ──────────────────────────────────
    logger.info("[%s] Running Bayesian Network analysis (BN, inference, Gibbs MCMC)…", corpus)
    try:
        from src.bayesian_network import run_bayesian_network_analysis

        bn_results = run_bayesian_network_analysis(
            texts=texts,
            dictionary=dictionary,
            n_topics=n_topics,
            seed=seed,
        )
        results["bayesian_network"] = bn_results
        logger.info("[%s] Bayesian Network analysis complete.", corpus)
    except Exception as exc:
        logger.warning("[%s] Bayesian Network analysis failed: %s", corpus, exc)
        results["bayesian_network"] = {"error": str(exc)}

    # ── 2. Hidden Markov Model ────────────────────────────────────────
    if years is not None and pd.to_numeric(years, errors="coerce").notna().sum() >= 5:
        logger.info("[%s] Running HMM analysis on temporal data…", corpus)
        try:
            from src.hmm_module import run_hmm_analysis

            years_num = pd.to_numeric(years, errors="coerce")
            hmm_results = run_hmm_analysis(
                years=years_num,
                theta=theta,
                topic_words=topic_words,
                n_states=min(n_topics, 5),
            )
            results["hmm"] = hmm_results
            logger.info("[%s] HMM analysis complete.", corpus)
        except Exception as exc:
            logger.warning("[%s] HMM analysis failed: %s", corpus, exc)
            results["hmm"] = {"error": str(exc)}
    else:
        results["hmm"] = {"skipped": "No temporal data available"}

    # ── 3. Kalman Filter ──────────────────────────────────────────────
    if years is not None and pd.to_numeric(years, errors="coerce").notna().sum() >= 5:
        logger.info("[%s] Running Kalman Filter analysis…", corpus)
        try:
            from src.kalman_filter import run_kalman_analysis

            years_num = pd.to_numeric(years, errors="coerce")
            kalman_results = run_kalman_analysis(
                years=years_num,
                theta=theta,
                topic_words=topic_words,
                n_topics=n_topics,
            )
            results["kalman"] = kalman_results
            logger.info("[%s] Kalman Filter analysis complete.", corpus)
        except Exception as exc:
            logger.warning("[%s] Kalman Filter analysis failed: %s", corpus, exc)
            results["kalman"] = {"error": str(exc)}
    else:
        results["kalman"] = {"skipped": "No temporal data available"}

    # ── 4. Markov Random Field / Markov Network ───────────────────────
    logger.info("[%s] Running MRF / Markov Network analysis…", corpus)
    try:
        from src.markov_network import run_mrf_analysis

        mrf_results = run_mrf_analysis(
            theta=theta,
            topic_words=topic_words,
            n_topics=n_topics,
        )
        results["mrf"] = mrf_results
        logger.info("[%s] MRF analysis complete.", corpus)
    except Exception as exc:
        logger.warning("[%s] MRF analysis failed: %s", corpus, exc)
        results["mrf"] = {"error": str(exc)}

    # ── 5. Decision Network ───────────────────────────────────────────
    logger.info("[%s] Running Decision Network analysis…", corpus)
    try:
        from src.decision_network import run_decision_analysis

        # Build coherence scores per topic if not provided
        coh = coherence_scores or {}
        # Build temporal trends summary
        trends = None
        if trend_df is not None and not trend_df.empty:
            trends = trend_df

        dn_results = run_decision_analysis(
            topic_words=topic_words,
            coherence_scores=coh,
            temporal_trends=trends,
            n_topics=n_topics,
        )
        results["decision_network"] = dn_results
        logger.info("[%s] Decision Network analysis complete.", corpus)
    except Exception as exc:
        logger.warning("[%s] Decision Network analysis failed: %s", corpus, exc)
        results["decision_network"] = {"error": str(exc)}

    # ── Save results ──────────────────────────────────────────────────
    save_path = RESULTS_DIR / f"syllabus_{corpus}.json"
    try:
        serializable = _make_serializable(results)
        save_path.write_text(json.dumps(serializable, indent=2, default=str))
        logger.info("[%s] Saved syllabus analysis to %s", corpus, save_path)
    except Exception as exc:
        logger.warning("[%s] Could not save syllabus results: %s", corpus, exc)

    return results


def _make_serializable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size <= 200:
            return obj.tolist()
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.DataFrame):
        if len(obj) <= 50:
            return obj.to_dict(orient="records")
        return f"<DataFrame shape={obj.shape}>"
    return obj


def syllabus_coverage_table() -> pd.DataFrame:
    """Return a DataFrame mapping each syllabus topic to project components."""
    rows = [
        {
            "Syllabus Topic": "1. Uncertain Knowledge Representation",
            "Project Component": "LDA generative model, Bayesian Network formulation",
            "Module": "src/lda_baseline.py, src/bayesian_network.py",
            "Description": "Documents as probabilistic mixtures of latent topics; "
                           "uncertainty represented via Dirichlet priors and posterior distributions",
        },
        {
            "Syllabus Topic": "2. Introduction to Bayesian Networks",
            "Project Component": "LDA as a BN, plate notation, DAG structure",
            "Module": "src/bayesian_network.py",
            "Description": "LDA's generative process modeled as a directed acyclic graph: "
                           "α→θ→z→w←φ←β with plate replication",
        },
        {
            "Syllabus Topic": "3. Representation Learning in BNs",
            "Project Component": "Structure learning, mutual information scoring",
            "Module": "src/bayesian_network.py",
            "Description": "Learn compact BN structure from document-topic distributions "
                           "using MI-based edge scoring and greedy search",
        },
        {
            "Syllabus Topic": "4. Inference in Bayesian Networks",
            "Project Component": "LDA inference, query-to-topic mapping",
            "Module": "src/bayesian_network.py, src/retrieve.py",
            "Description": "Posterior inference P(topic|document) via variational methods "
                           "and exact enumeration on small BNs",
        },
        {
            "Syllabus Topic": "5. Exact and Approximate Inference",
            "Project Component": "Variable Elimination, Likelihood Weighting",
            "Module": "src/bayesian_network.py",
            "Description": "Exact inference via variable elimination for small topic models; "
                           "approximate inference via likelihood-weighted sampling",
        },
        {
            "Syllabus Topic": "6. MCMC Method",
            "Project Component": "Collapsed Gibbs Sampler for LDA",
            "Module": "src/bayesian_network.py",
            "Description": "Markov Chain Monte Carlo via collapsed Gibbs sampling: "
                           "iteratively resample topic assignments z_{d,n} from conditional posterior",
        },
        {
            "Syllabus Topic": "7. Hidden Markov Models",
            "Project Component": "HMM for temporal topic dynamics",
            "Module": "src/hmm_module.py",
            "Description": "Model topic evolution over years as hidden states; "
                           "Forward algorithm, Viterbi decoding, Baum-Welch learning",
        },
        {
            "Syllabus Topic": "8. Kalman Filter",
            "Project Component": "Kalman Filter/Smoother for topic prevalence",
            "Module": "src/kalman_filter.py",
            "Description": "Continuous-state linear dynamical system tracking topic prevalence; "
                           "predict-update cycle with RTS backward smoother and uncertainty bands",
        },
        {
            "Syllabus Topic": "9. MRF & Markov Networks",
            "Project Component": "Topic co-occurrence MRF, potential functions",
            "Module": "src/markov_network.py",
            "Description": "Undirected graphical model capturing topic affinities; "
                           "pairwise potential functions from document co-occurrence",
        },
        {
            "Syllabus Topic": "10. Message Passing",
            "Project Component": "Belief Propagation on topic MRF",
            "Module": "src/markov_network.py",
            "Description": "Sum-product message passing (loopy BP) on the topic co-occurrence graph "
                           "to compute marginal beliefs over topic activations",
        },
        {
            "Syllabus Topic": "11. Learning in Markov Networks",
            "Project Component": "Pseudo-likelihood parameter estimation",
            "Module": "src/markov_network.py",
            "Description": "Learn MRF edge potentials from data via maximum pseudo-likelihood; "
                           "partition function approximation",
        },
        {
            "Syllabus Topic": "12. Decision Networks",
            "Project Component": "Research topic selection under uncertainty",
            "Module": "src/decision_network.py",
            "Description": "Influence diagram with decision (topic choice), chance (growth, coherence), "
                           "and utility nodes; MEU computation and Value of Information",
        },
    ]
    return pd.DataFrame(rows)

"""Interactive dashboard for hybrid LDA + embedding topic modeling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quiet import configure_warnings

configure_warnings()

from src.config import RESULTS_DIR
from src.embeddings import get_model

st.set_page_config(page_title="Hybrid LDA–LLM Topic Explorer", layout="wide")


def _chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True)


def _table(df) -> None:
    st.dataframe(df, width="stretch")


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    demo = _load_json(RESULTS_DIR / "demo_metrics.json") or {}
    rows = demo.get("metrics_preview") or []
    return pd.DataFrame(rows)


@st.cache_data
def load_topics(corpus: str) -> dict:
    path = RESULTS_DIR / f"topics_{corpus}.json"
    if path.exists():
        return json.loads(path.read_text())
    demo = _load_json(RESULTS_DIR / "demo_metrics.json") or {}
    return (demo.get("corpora") or {}).get(corpus, {}).get("topics") or {}


@st.cache_data
def load_docs(corpus: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"docs_{corpus}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_temporal(corpus: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"temporal_{corpus}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_arrays(corpus: str) -> dict | None:
    path = RESULTS_DIR / f"arrays_{corpus}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    st.title("Probabilistic Topic Modeling for Research Trend Analysis")
    st.caption("Hybrid LDA–LLM approach · Course 22AIE301 · Coherence-scored comparison vs baseline LDA")

    metrics = load_metrics()
    if metrics.empty:
        st.warning(
            "No results yet. Run `python run_pipeline.py` first. "
            "The dashboard will populate charts, topics, and retrieval from `results/`."
        )
        st.stop()

    corpora = sorted(metrics["corpus"].dropna().unique().tolist())
    corpus = st.sidebar.selectbox("Corpus", corpora, index=0)
    measure = st.sidebar.selectbox("Coherence measure", ["c_v", "c_npmi", "u_mass"], index=0)

    subset = metrics[metrics["corpus"] == corpus]
    topics = load_topics(corpus)
    docs = load_docs(corpus)
    trends = load_temporal(corpus)
    arrays = load_arrays(corpus)

    tab_eval, tab_topics, tab_umap, tab_trend, tab_query = st.tabs(
        [
            "Coherence vs LDA",
            "Topic explorer",
            "UMAP clusters",
            "Temporal trends",
            "Related-work search",
        ]
    )

    with tab_eval:
        st.subheader("Quantitative comparison against baseline LDA")
        st.markdown(
            "Professor feedback: *Augmenting LDA with semantic embeddings should improve "
            "topic coherence. Calculate a Coherence Score to compare hybrid results against baseline LDA.*"
        )
        fig = px.line(
            subset,
            x="n_topics",
            y=measure,
            color="model",
            markers=True,
            title=f"{measure} vs number of topics ({corpus})",
        )
        _chart(fig)
        best = (
            subset.sort_values(measure, ascending=False)
            .groupby("model", as_index=False)
            .first()[["model", "n_topics", "c_v", "c_npmi", "u_mass", "diversity", "silhouette"]]
        )
        _table(best)

        if "LDA" in best["model"].values and "Hybrid-BERT-LDA" in best["model"].values:
            lda_cv = float(best.loc[best["model"] == "LDA", "c_v"].iloc[0])
            hyb_cv = float(best.loc[best["model"] == "Hybrid-BERT-LDA", "c_v"].iloc[0])
            if np.isfinite(lda_cv) and lda_cv != 0:
                delta = 100.0 * (hyb_cv - lda_cv) / abs(lda_cv)
                st.metric("Hybrid C_v vs LDA", f"{hyb_cv:.3f}", f"{delta:+.1f}% vs LDA ({lda_cv:.3f})")

        sil_path = RESULTS_DIR / "silhouette.csv"
        if sil_path.exists():
            sil = pd.read_csv(sil_path)
            sil_c = sil[sil["corpus"] == corpus]
            _chart(
                px.bar(
                    sil_c,
                    x="reducer",
                    y="silhouette",
                    color="model",
                    barmode="group",
                    title="Silhouette score by dimensionality reduction (PMC paper comparison)",
                )
            )

        alpha_path = RESULTS_DIR / "alpha_sweep.csv"
        if alpha_path.exists():
            alpha = pd.read_csv(alpha_path)
            alpha_c = alpha[alpha["corpus"] == corpus]
            _chart(
                px.line(
                    alpha_c,
                    x="alpha",
                    y="c_v",
                    markers=True,
                    title="α sweep: weight on LDA topic vector vs C_v",
                )
            )

    with tab_topics:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline LDA topics**")
            _table(pd.DataFrame(topics.get("lda_topics") or []))
        with col2:
            st.markdown("**Hybrid BERT–LDA topics**")
            _table(pd.DataFrame(topics.get("hybrid_topics") or []))
        if not docs.empty:
            topic_id = st.slider("Inspect hybrid cluster", 0, int(docs["hybrid_topic"].max()), 0)
            show = docs[docs["hybrid_topic"] == topic_id][["title", "year", "text"]].head(8)
            _table(show)

    with tab_umap:
        if arrays is None:
            st.info("UMAP coordinates appear after a pipeline run (`arrays_{corpus}.npz`).")
        else:
            umap2d = arrays["umap2d"]
            labels = arrays["hybrid_labels"]
            n = min(len(labels), umap2d.shape[0], len(docs) if not docs.empty else umap2d.shape[0])
            plot_df = pd.DataFrame(
                {
                    "UMAP-1": umap2d[:n, 0],
                    "UMAP-2": umap2d[:n, 1],
                    "topic": labels[:n],
                }
            )
            if not docs.empty:
                plot_df["title"] = docs["title"].fillna("").astype(str).values[:n]
            hover = [c for c in plot_df.columns if c not in {"UMAP-1", "UMAP-2"}]
            _chart(
                px.scatter(
                    plot_df,
                    x="UMAP-1",
                    y="UMAP-2",
                    color="topic",
                    hover_data=hover,
                    title=f"Hybrid embedding space ({corpus})",
                )
            )

    with tab_trend:
        if trends.empty:
            st.info("Temporal trends require publication years (arXiv). 20 Newsgroups has no timestamps.")
        else:
            _chart(
                px.line(
                    trends,
                    x="year",
                    y="prevalence",
                    color="label",
                    markers=True,
                    title="Topic prevalence over years (mean θ)",
                )
            )

    with tab_query:
        st.markdown("Paste an abstract. The system returns nearest papers and a soft topic ranking.")
        query = st.text_area("Query abstract", height=140, placeholder="Paste a paper abstract…")
        if st.button("Find related work") and query.strip():
            if arrays is None or docs.empty:
                st.error("Need pipeline artifacts (`docs_*.csv` and `arrays_*.npz`).")
            else:
                model = get_model()
                q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                n = min(len(docs), arrays["embeddings"].shape[0])
                sims = cosine_similarity(q, arrays["embeddings"][:n]).ravel()
                top = np.argsort(sims)[::-1][:8]
                result = docs.iloc[top].copy()
                result["similarity"] = sims[top]
                st.dataframe(result[["title", "year", "similarity", "text"]], width="stretch")

                labels = arrays["hybrid_labels"][:n]
                scores = []
                for k in sorted(set(int(x) for x in labels)):
                    mask = labels == k
                    scores.append((int(k), float(sims[mask].mean()) if mask.any() else 0.0))
                scores.sort(key=lambda x: x[1], reverse=True)
                hyb_topics = {t["topic_id"]: t["top_words"] for t in topics.get("hybrid_topics") or []}
                rank_df = pd.DataFrame(
                    [
                        {"topic_id": k, "score": s, "top_words": hyb_topics.get(k, "")}
                        for k, s in scores[:6]
                    ]
                )
                st.markdown("**Probabilistic / embedding topic ranking**")
                _table(rank_df)


if __name__ == "__main__":
    main()

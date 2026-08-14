---
marp: true
theme: default
paginate: true
title: Hybrid LDA–LLM Topic Modeling
---

# Probabilistic Topic Modeling for Research Trend Analysis
## A Hybrid LDA–LLM Approach

**22AIE301** · Hybrid LDA + Sentence-BERT · Coherence vs baseline LDA

---

# Problem

Scientific literature grows faster than anyone can read.

- Manual review does not scale
- Need interpretable **topics**, **trends over years**, and **related-work search**

---

# Why LDA is not enough

LDA is a generative probabilistic model (course core):

- Document = mixture of topics (θ)
- Topic = distribution over words (φ)

Limitation: it uses **word co-occurrence**, so topics can be statistically valid but **semantically incoherent**.

---

# Professor feedback (evaluation)

> Augmenting LDA with semantic embeddings should significantly improve the coherence of the identified topics in large scientific corpora.
>
> **Calculate a Coherence Score** to compare hybrid results against a **baseline LDA** model.

Primary metric: **C_v** (also NPMI, UMass).

---

# Hybrid idea

Keep LDA’s probability structure. Refine it with LLM embeddings.

```
h = [ α · θ_LDA  |  (1−α) · e_SBERT ]
```

Then: **UMAP → k-means → c-TF-IDF**

α is tuned to **maximise C_v** (abstract novelty).

---

# Pipeline

1. Corpus → clean / tokenize / lemmatize
2. Baseline **LDA**
3. **Sentence-BERT** embeddings
4. Hybrid concat + clustering
5. **C_v vs LDA**
6. Topic prevalence **over years**
7. Query → ranked papers / topics

---

# Datasets

| Corpus | Role |
|---|---|
| 20 Newsgroups | Sanity check (Paul et al.) |
| arXiv ML/AI abstracts | Scientific corpus + timestamps |

CORD-19 (800k) is noted but not run in full on a laptop.

Embeddings: `all-MiniLM-L6-v2` (CPU). Optional SciBERT/SPECTER.

---

# Coherence score

For top words \(w_1 \ldots w_n\) of a topic:

- **C_v** — NPMI + cosine over a sliding window (best human correlation)
- **NPMI** — normalized pointwise mutual information
- **UMass** — document co-occurrence (often negative)

Higher C_v = more interpretable topics.

---

# Results (seeded run, 600 docs, K ∈ {8,12})

**C_v vs baseline LDA**

| Corpus | LDA | Hybrid | Lift |
|---|---:|---:|---:|
| 20 Newsgroups | 0.51 | **0.79** | **+55%** |
| arXiv ML/AI | 0.44 | **0.57** (α=0.5) / **0.72** (α*=0.25) | **+29% / +63%** |

Semantic embeddings improve topic coherence, as requested.

---

# Relation to source papers

- **Paul et al. 2025** — BERT+LDA, coherence on news / 20 Newsgroups
- **George & Sumathy 2023** — BERT-LDA + UMAP + k-means, silhouette on CORD-19

This project adds: **C_v vs LDA on scientific arXiv**, α-tuning, **temporal drift**, related-work query.

---

# Demo

```bash
python run_pipeline.py
streamlit run app/streamlit_app.py
```

Dashboard: coherence charts · topic explorer · UMAP · yearly trends · paste-an-abstract search

---

# Limitations and next steps

- Default run uses 2k docs/corpus (scale with `--full`)
- MiniLM is general-purpose; SciBERT/SPECTER for domain terms
- Intruder-word human study left as optional
- Dynamic LDA / DBN-style state evolution is future work

---

# Takeaway

**LDA for interpretable probabilities. Embeddings for meaning. C_v to prove the hybrid is better.**

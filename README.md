# Hybrid LDA–LLM Topic Modeling

**Probabilistic Topic Modeling for Research Trend Analysis: A Hybrid LDA–LLM Approach**  
Course: **22AIE301**

Augment LDA with Sentence-BERT embeddings, then **quantitatively compare topic coherence (`C_v`, NPMI, UMass) against a baseline LDA model** on 20 Newsgroups and arXiv ML/AI abstracts. This is the evaluation the professor requested.

## What you get

| Path | Role |
|---|---|
| `src/` | Shared library (data, LDA, embeddings, hybrid, coherence, temporal, retrieval) |
| `run_pipeline.py` | One-command experiment → `results/metrics.csv` + figures |
| `notebooks/` | Walkthrough: prep → LDA → hybrid → evaluation |
| `app/streamlit_app.py` | Dashboard: coherence charts, topics, UMAP, trends, related-work search |
| `docs/report.md` | Academic report |
| `docs/slides.md` | 12-slide Marp deck |

## Setup

Needs **Python 3.9–3.12**. Homebrew’s current `python3` is often **3.14**, which cannot install this stack (NumPy has no wheel, pip tries to compile, and that fails). Do not run `pip install -r requirements.txt` inside a 3.14 conda env.

**macOS / Linux**

```bash
# if `python3 --version` is 3.13 or 3.14:
brew install python@3.12
conda deactivate   # if a conda env is active
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

**Windows** (Command Prompt or double-click)

```bat
setup.bat
.venv\Scripts\activate
```

After that, the same commands work on every OS.

```bash
python run_pipeline.py --smoke
```

That is a small 20 Newsgroups run (150 documents, K=8) that checks the install. Output goes to `results/smoke/` so the main experiment files are left alone.

## Run the experiment

Laptop-friendly default (2k docs/corpus, K ∈ {8, 12, 16}):

```bash
python run_pipeline.py
```

Scale up (8k docs, K ∈ {5,10,15,20,25,30}):

```bash
python run_pipeline.py --full
```

Single corpus:

```bash
python run_pipeline.py --corpora arxiv --n-docs 2000 --k 8,12,16
```

Then:

```bash
jupyter notebook notebooks/04_evaluation.ipynb
python run_dashboard.py
```

(`streamlit run app/streamlit_app.py` also works; `run_dashboard.py` starts Streamlit after silencing third-party import warnings.)

## Seeded results (already in `results/`)

600 documents/corpus, K ∈ {8,12}, seed 42:

| Corpus | Best LDA C_v | Best hybrid C_v | Lift |
|---|---:|---:|---:|
| 20 Newsgroups | 0.51 | 0.79 | +55% |
| arXiv (2018–2025) | 0.44 | 0.57 (α=0.5) / 0.72 (α*=0.25) | +29% / +63% |

That is the professor’s coherence comparison. Re-run with `--n-docs 2000` or `--full` to scale up.

## Method (matches the abstract)

1. **Baseline LDA** — gensim LDA; each document is a probability mixture of topics.
2. **LLM embeddings** — `all-MiniLM-L6-v2` Sentence-BERT vectors.
3. **Hybrid** — concatenate `h = [α · θ | (1−α) · e]`, reduce with UMAP (also PCA / t-SNE), cluster with k-means, label clusters with c-TF-IDF.
4. **Coherence** — `C_v` (primary), NPMI, UMass vs LDA. Topic diversity and silhouette as secondary metrics.
5. **α sweep** — empirically pick the blend that maximises `C_v`.
6. **Temporal trends** — mean topic probability by publication year (arXiv).
7. **Related-work search** — query abstract → ranked papers and topics.

## Datasets

- **20 Newsgroups** — sanity check (Paul et al. also evaluated on this English corpus).
- **arXiv cs.LG / cs.AI / cs.CL / cs.CV abstracts** — scientific corpus with timestamps (recommended in the abstract; CORD-19’s 800k+ papers is too large for a laptop).

## Papers implemented

- Paul et al., *Combining BERT with LDA*, IAENG IJCS 52(2), 2025.
- George & Sumathy, *An integrated clustering and BERT framework for improved topic modeling*, Int J Inf Technol, 2023 (PMC10163298).

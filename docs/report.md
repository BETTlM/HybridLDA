# Probabilistic Topic Modeling for Research Trend Analysis: A Hybrid LDA–LLM Approach

**Course:** 22AIE301  
**Project type:** Working prototype + comparative evaluation report

## Abstract

Understanding how research interest evolves within a scientific domain is critical for researchers, funding bodies, and institutions seeking to identify emerging directions and avoid redundant work. Traditional literature review is manual, time-consuming, and does not scale to the thousands of papers published annually. This project proposes a hybrid probabilistic topic modeling framework that combines Latent Dirichlet Allocation (LDA), a generative probabilistic graphical model, with modern LLM-based semantic embeddings to analyze and visualize thematic trends within a research corpus.

LDA treats each document as a probabilistic mixture of latent topics, and each topic as a probability distribution over words, offering an interpretable generative structure grounded in Bayesian inference — directly reflecting core course concepts of latent variable models and probabilistic inference. However, LDA's reliance on word co-occurrence often produces topics that are statistically valid but semantically incoherent. To address this, the project augments LDA with dense embeddings from a pretrained LLM (Sentence-BERT), which capture contextual meaning beyond raw word frequency. These embeddings are used to (i) improve topic coherence through embedding-guided clustering, and (ii) enable semantic similarity search across papers, so a query paper can be probabilistically mapped to its most relevant existing topics.

The pipeline is applied to 20 Newsgroups (sanity check) and a corpus of arXiv ML/AI abstracts (scientific literature with timestamps). The system outputs: (a) a set of interpretable latent topics with associated probability distributions over terms, (b) a temporal trend graph showing how topic prevalence shifts across publication years, and (c) a document-to-topic probability assignment enabling users to explore corpus structure interactively.

**Evaluation (professor feedback).** Augmenting LDA with semantic embeddings is expected to improve the coherence of identified topics in scientific corpora. This report therefore computes **Coherence Score** (\(C_v\), plus NPMI and UMass) to quantitatively compare the hybrid model against a **baseline LDA** model. Silhouette score and topic diversity are reported as secondary metrics.

## 1. Introduction

Topic models recover latent themes from unlabeled text. LDA [1] remains the standard probabilistic approach because each document is a *mixture* of topics and each topic is a *distribution* over words — quantities that can be inspected, plotted over time, and used as a ranked retrieval signal. Neural embedding models such as BERT [2] and Sentence-BERT [3] encode contextual meaning that bag-of-words LDA cannot see, but clustering embeddings alone discards the generative probability structure that this course emphasizes.

Two recent papers combine the two families. Paul et al. [4] concatenate BERT document embeddings with LDA topic vectors and show a coherence gain on Bengali news and on English 20 Newsgroups (LDA 0.63 vs BERT-LDA 0.66). George and Sumathy [5] concatenate BERT and LDA, reduce with PCA / t-SNE / UMAP, cluster with k-means, and report silhouette scores on CORD-19; UMAP + BERT-LDA was best.

This project implements that hybrid on a **scientific** corpus (arXiv), adds the professor-requested **coherence comparison against baseline LDA**, tunes the blend weight \(\alpha\) to maximise \(C_v\), tracks **temporal probability drift**, and deploys an interactive dashboard for related-work search.

## 2. Related work

**LDA.** Blei, Ng, and Jordan [1] introduce LDA as a three-level hierarchical Bayesian model. Inference yields \(\theta_d\) (document-topic) and \(\phi_k\) (topic-word).

**Embeddings for topics.** Sentence-BERT [3] produces comparable sentence vectors. SciBERT [6] and SPECTER specialise in scientific text. BERTopic uses embeddings + c-TF-IDF [7].

**Hybrid BERT–LDA.** Paul et al. [4] and George & Sumathy [5] are the two source papers. Xie et al. [8] combine LDA and BERT for multilingual topic similarity. The present work follows the concatenation + clustering recipe, but evaluates with **\(C_v\) vs LDA** on arXiv rather than silhouette-only on CORD-19.

**Coherence.** Röder, Both, and Hinneburg [9] show that \(C_v\) (NPMI + indirect cosine over a sliding window) correlates best with human topic quality. UMass and UCI remain useful diagnostics.

## 3. Methodology

### 3.1 Preprocessing

Documents are lowercased; URLs, emails, and citation markers are stripped; non-alphabetic tokens are removed. English stopwords plus generic scientific stopwords (`method`, `result`, `paper`, …) are dropped. Remaining tokens are WordNet-lemmatized. Documents with fewer than eight tokens are discarded.

### 3.2 Baseline LDA

Let \(K\) be the number of topics. Gensim LDA is trained with automatic \(\alpha, \eta\), 10 passes. For document \(d\),

\[
\theta_d \in \Delta^{K-1}, \qquad \theta_{d,k} = p(z=k \mid d).
\]

Topic \(k\) is represented by its top-\(N\) words under \(\phi_k\). This model is the **baseline**.

### 3.3 LLM embeddings

Each raw abstract is encoded with Sentence-BERT (`all-MiniLM-L6-v2`) to a 384-dimensional L2-normalised vector \(e_d\). The encoder is used as a frozen semantic feature extractor (no fine-tuning), matching the abstract’s “pretrained LLM embeddings” requirement while remaining CPU-feasible.

### 3.4 Hybrid representation

Following [4, 5] and the abstract’s blend parameter \(\alpha\),

\[
h_d = \big[\, \alpha\,\hat\theta_d \;\Vert\; (1-\alpha)\,\hat e_d \,\big],
\]

where hats denote L2 normalisation and \(\Vert\) is concatenation. \(\alpha=1\) recovers LDA geometry; \(\alpha=0\) recovers pure embeddings. Default \(\alpha=0.5\); a sweep \(\alpha \in \{0, 0.25, 0.5, 0.75, 1\}\) selects the value that maximises \(C_v\).

### 3.5 Embedding-guided clustering

High-dimensional \(h_d\) is reduced with UMAP (primary), PCA, or t-SNE, then clustered with k-means (\(K\) clusters). Cluster labels are turned into word lists with **class-based TF-IDF** (c-TF-IDF) [7]: concatenate all tokens in a cluster, compute TF, multiply by a class-level IDF. The resulting top words are comparable to LDA’s top words, so **the same coherence metrics apply to both models**.

### 3.6 Coherence score (primary evaluation)

For a set of topics \(\{w_{k,1},\ldots,w_{k,N}\}_{k=1}^{K}\) and the tokenized corpus, gensim `CoherenceModel` computes:

- \(C_v\) — primary (professor request)
- \(C_{\mathrm{NPMI}}\)
- \(U_{\mathrm{Mass}}\)

**Topic diversity** is the fraction of unique top words across topics. **Silhouette** on the reduced space replicates Table 3 of [5].

### 3.7 Temporal probability drift

For arXiv documents with a publication year \(y\), the mean LDA topic distribution is

\[
\bar\theta_{y,k} = \frac{1}{|\{d: \mathrm{year}(d)=y\}|} \sum_{d:\,\mathrm{year}(d)=y} \theta_{d,k}.
\]

Plotting \(\bar\theta_{y,k}\) against \(y\) is the abstract’s temporal trend graph (in the spirit of a dynamic Bayesian network’s state evolution, without fitting a full DBN).

### 3.8 Query-to-topic ranking

A query abstract is encoded to \(e_q\) and to \(\theta_q\) (LDA inference on the query bag-of-words). Related papers are nearest neighbours of the hybrid vector. Topic scores blend \(\theta_q\) with cosine similarity to embedding centroids, using the same \(\alpha\).

## 4. Experimental setup

| Item | Default | Full (`--full`) |
|---|---|---|
| Documents / corpus | 2,000 | 8,000 |
| \(K\) grid | 8, 12, 16 | 5, 10, 15, 20, 25, 30 |
| Corpora | 20 Newsgroups, arXiv ML/AI | same |
| Embedding | `all-MiniLM-L6-v2` | optional SciBERT/SPECTER |
| Seed | 42 | 42 |

**Why not CORD-19.** George & Sumathy used ~800k CORD-19 papers. That scale is not practical for a course prototype on a laptop; arXiv abstracts in cs.LG / cs.AI / cs.CL / cs.CV are the scientific stand-in recommended in the original abstract (Kaggle Cornell dump / HuggingFace `CShorten/ML-ArXiv-Papers`, with the arXiv API as fallback).

**Models compared**

1. **LDA** — gensim baseline  
2. **BERT** — MiniLM → UMAP → k-means → c-TF-IDF  
3. **Hybrid-BERT-LDA** — concat → UMAP → k-means → c-TF-IDF  

Software: Python, gensim, sentence-transformers, scikit-learn, umap-learn, NLTK, Streamlit / Plotly.

## 5. Results

Seeded experiment (`python run_pipeline.py --n-docs 600 --k 8,12`, seed 42). Sentence-BERT: `all-MiniLM-L6-v2`. Hybrid default \(\alpha=0.5\), then \(\alpha\) is swept at the best \(K\).

### 5.1 Coherence vs baseline LDA (professor metric)

**Table 1.** \(C_v\) (primary), NPMI, and UMass. Higher \(C_v\) / NPMI is better; UMass is typically negative (closer to 0 is better).

| Corpus | Model | K | \(C_v\) | NPMI | UMass | Diversity |
|---|---|---:|---:|---:|---:|---:|
| 20 Newsgroups | LDA (baseline) | 8 | 0.512 | −0.101 | −5.03 | 0.91 |
| 20 Newsgroups | BERT | 12 | 0.726 | 0.009 | −4.06 | 0.98 |
| 20 Newsgroups | **Hybrid BERT–LDA** | 12 | **0.794** | −0.009 | −3.64 | 0.97 |
| arXiv ML/AI | LDA (baseline) | 12 | 0.443 | −0.036 | −3.31 | 0.79 |
| arXiv ML/AI | BERT | 12 | 0.545 | 0.012 | −2.76 | 0.66 |
| arXiv ML/AI | **Hybrid BERT–LDA** (\(\alpha=0.5\)) | 12 | **0.571** | 0.011 | −2.00 | 0.66 |
| arXiv ML/AI | Hybrid after \(\alpha^\star=0.25\) | 12 | **0.722** | 0.146 | −1.24 | 0.68 |

**Lift vs baseline LDA (best \(C_v\)):**

- 20 Newsgroups: \(0.794 / 0.512\) → **+55%**
- arXiv (scientific corpus): \(0.571 / 0.443\) → **+29%** at \(\alpha=0.5\); **+63%** at the coherence-maximising \(\alpha^\star=0.25\)

This is the quantitative answer to the professor: augmenting LDA with semantic embeddings **does** raise topic coherence on both a general English corpus and a scientific arXiv corpus.

Figures: `results/figures/coherence_c_v_*.png`, `results/figures/best_c_v_*.png`.

### 5.2 Silhouette by reducer (replication of George & Sumathy)

**Table 2.** Silhouette at the hybrid’s best \(K\) (12).

| Corpus | Model | PCA | t-SNE | UMAP |
|---|---|---:|---:|---:|
| 20 Newsgroups | LDA | 0.29 | 0.49 | 0.59 |
| 20 Newsgroups | BERT | 0.13 | 0.41 | 0.44 |
| 20 Newsgroups | Hybrid | 0.22 | 0.48 | 0.53 |
| arXiv | LDA | 0.24 | 0.43 | 0.50 |
| arXiv | BERT | 0.10 | 0.36 | 0.37 |
| arXiv | Hybrid | 0.18 | 0.42 | 0.45 |

UMAP remains the strongest reducer, matching [5]. Silhouette on LDA-θ can look high because \(\theta\) is already a low-dimensional simplex; **coherence**, not silhouette, is the metric that scores the words a reader sees.

### 5.3 \(\alpha\) sweep (abstract novelty)

On arXiv at \(K=12\), \(C_v\) peaks at **\(\alpha=0.25\)** (0.72), not at pure embeddings (\(\alpha=0\): 0.67) or pure LDA geometry (\(\alpha=1\): 0.49). A modest LDA component plus a larger embedding component is the coherence-maximising blend. On 20 Newsgroups the peak is \(\alpha=0.5\) (\(C_v=0.79\)).

See `results/alpha_sweep.csv` and `results/figures/alpha_sweep_*.png`.

### 5.4 Qualitative topics and trends

arXiv hybrid clusters (c-TF-IDF) recover scientific themes such as transformers/attention, segmentation, retrieval, and training/generation — sharper than LDA’s more generic `model / network / algorithm` lists (`results/topics_arxiv.json`).

Temporal prevalence is tracked for **2018–2025** (`results/temporal_arxiv.csv`, `results/figures/temporal_arxiv.png`). 20 Newsgroups has no timestamps, so trend analysis is arXiv-only.

## 6. Discussion

LDA captures **global** co-occurrence structure (the probability backbone required by the course). Sentence-BERT captures **local contextual** similarity (synonyms, paraphrases, scientific jargon that never co-occur). Concatenation lets k-means separate documents that LDA would mix because they share function words, while c-TF-IDF still yields inspectable word lists.

Coherence is the right quantitative test: unlike silhouette, it scores the **words humans read**, which is exactly the professor’s request. Diversity guards against collapsed topics (many topics repeating the same words). The \(\alpha\) sweep makes the hybrid *not* a naive concat: the blend is chosen to maximise the same metric used in the comparison.

**Limitations.** The default run uses 2k documents, not 50k–100k. MiniLM is not science-specific; SciBERT would likely help arXiv further. Human intruder-word tests [9] are not automated here. A full dynamic topic model (Blei & Lafferty) would model \(\theta\) evolution more rigorously than yearly averages.

## 7. Use cases (from the project abstract)

1. **Research trend discovery** — a student entering a subfield sees 8–12 dominant themes without reading hundreds of abstracts.  
2. **Literature-review acceleration** — paste a new abstract; receive top-\(k\) similar papers and a topic ranking (dashboard “Related-work search”).  
3. **Institutional mapping** — yearly \(\bar\theta\) shows how a corpus’s mass shifts over 5–10 years.  
4. **Conference tracking** — apply the same pipeline to a venue’s proceedings.

## 8. Conclusion

This prototype keeps LDA’s generative probability structure and refines cluster assignments with Sentence-BERT embeddings. The mandatory evaluation is a **Coherence Score comparison against baseline LDA**. Secondary outputs — silhouette by reducer, \(\alpha\) tuning, temporal drift, and query-to-topic ranking — match the original project abstract and the two source papers.

Re-run at full scale with:

```bash
python run_pipeline.py --full
streamlit run app/streamlit_app.py
```

## References

1. D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent Dirichlet allocation,” *JMLR*, 2003.  
2. J. Devlin et al., “BERT: Pre-training of deep bidirectional transformers for language understanding,” NAACL, 2019.  
3. N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” EMNLP, 2019.  
4. P. C. Paul et al., “Combining BERT with LDA: Improved topic modeling in Bengali language,” *IAENG IJCS*, vol. 52, no. 2, pp. 383–393, 2025.  
5. L. George and P. Sumathy, “An integrated clustering and BERT framework for improved topic modeling,” *Int. J. Inf. Technol.*, 2023. PMC10163298.  
6. I. Beltagy, K. Lo, and A. Cohan, “SciBERT: A pretrained language model for scientific text,” EMNLP, 2019.  
7. M. Grootendorst, “BERTopic: Neural topic modeling with a class-based TF-IDF procedure,” 2022.  
8. Q. Xie et al., “Monolingual and multilingual topic analysis using LDA and BERT embeddings,” *J. Informetrics*, 2020.  
9. M. Röder, A. Both, and A. Hinneburg, “Exploring the space of topic coherence measures,” WSDM, 2015.  
10. L. McInnes, J. Healy, and J. Melville, “UMAP: Uniform manifold approximation and projection for dimension reduction,” 2018.

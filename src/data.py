"""Dataset loaders for 20 Newsgroups and arXiv ML/AI abstracts."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from src.config import ARXIV_CATEGORIES, ARXIV_HF_DATASET, RAW_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(from|subject|organization|lines|nntp-posting-host|x-newsreader|reply-to):", re.I)


def _strip_newsgroup_headers(text: str) -> str:
    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1
            break
        if not _HEADER_RE.match(line) and i > 0:
            body_start = i
            break
    body = "\n".join(lines[body_start:])
    body = re.sub(r"^>.*$", "", body, flags=re.M)
    return body.strip()


def load_20newsgroups(n_docs: int | None = None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Load 20 Newsgroups posts (sklearn). Cached as CSV under data/raw."""
    cache_path = RAW_DIR / "20newsgroups.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        from sklearn.datasets import fetch_20newsgroups

        bunch = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=seed,
        )
        df = pd.DataFrame(
            {
                "doc_id": [f"ng-{i}" for i in range(len(bunch.data))],
                "text": [_strip_newsgroup_headers(t) for t in bunch.data],
                "label": bunch.target,
                "label_name": [bunch.target_names[t] for t in bunch.target],
                "title": "",
                "year": pd.NA,
            }
        )
        df = df[df["text"].str.len() >= 80].reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Cached 20 Newsgroups to %s (%d docs)", cache_path, len(df))

    if "year" not in df.columns:
        df["year"] = pd.NA
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n_docs is not None:
        df = df.head(n_docs).reset_index(drop=True)
    df["corpus"] = "20newsgroups"
    return df


def year_from_arxiv_id(arxiv_id: str) -> int | None:
    """Parse publication year from an arXiv id such as 1706.03762 or hep-th/9901001."""
    if arxiv_id is None:
        return None
    text = str(arxiv_id)
    m = re.search(r"(?:^|[^\d])(\d{2})(\d{2})\.(\d{4,5})", text)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 91 else 2000 + yy
    m = re.search(r"/(\d{2})(\d{2})\d+", text)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 91 else 2000 + yy
    return None


def _row_year(row: dict, arxiv_id: str) -> int | None:
    for key in ("update_date", "published", "date", "year", "versions"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, int) and 1990 <= val <= 2030:
            return val
        text = str(val)
        m = re.search(r"(19|20)\d{2}", text)
        if m:
            return int(m.group(0))
    return year_from_arxiv_id(arxiv_id)


def _load_arxiv_huggingface(n_docs: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(ARXIV_HF_DATASET, split="train")
    records = []
    for row in ds:
        abstract = (row.get("abstract") or "").strip()
        title = (row.get("title") or "").strip()
        if len(abstract) < 80:
            continue
        arxiv_id = str(row.get("id") or row.get("ids") or row.get("Unnamed: 0") or len(records))
        cats = row.get("categories") or row.get("terms") or row.get("label") or "cs.LG"
        records.append(
            {
                "doc_id": arxiv_id,
                "title": title,
                "text": abstract,
                "label": cats,
                "label_name": cats,
                "year": _row_year(row, arxiv_id),
            }
        )
        if len(records) >= max(n_docs * 4, n_docs):
            break
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("HuggingFace arXiv dataset returned no abstracts")
    return df.sample(frac=1.0, random_state=seed).head(n_docs).reset_index(drop=True)


def _load_arxiv_api(n_docs: int, seed: int) -> pd.DataFrame:
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=1.0, num_retries=3)
    cat_q = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    per_year = max(30, n_docs // 8 + 5)
    records = []
    # Stratify by year so temporal trend analysis has a real time axis.
    for year in range(2018, 2026):
        search = arxiv.Search(
            query=f"({cat_q}) AND submittedDate:[{year}01010000 TO {year}12312359]",
            max_results=per_year,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        try:
            for result in client.results(search):
                abstract = (result.summary or "").replace("\n", " ").strip()
                if len(abstract) < 80:
                    continue
                cats = " ".join(result.categories)
                records.append(
                    {
                        "doc_id": result.entry_id.split("/")[-1],
                        "title": result.title.replace("\n", " ").strip(),
                        "text": abstract,
                        "label": cats,
                        "label_name": cats,
                        "year": result.published.year if result.published else year,
                    }
                )
        except Exception as exc:
            logger.warning("arXiv API year %s failed: %s", year, exc)
            continue
    if len(records) < 50:
        raise RuntimeError("arXiv API returned too few abstracts")
    df = pd.DataFrame(records).drop_duplicates(subset=["doc_id"])
    return df.sample(frac=1.0, random_state=seed).head(n_docs).reset_index(drop=True)


def load_arxiv(n_docs: int | None = None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Load arXiv ML/AI abstracts with publication years (needed for trend plots).

    Prefers the arXiv API (has `published`), then HuggingFace, then a bundled sample.
    """
    cache_path = RAW_DIR / "arxiv_ml.csv"
    target = n_docs or 8000

    df = None
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        year_ok = "year" in cached.columns and pd.to_numeric(cached["year"], errors="coerce").notna().sum() >= 5
        if year_ok and (n_docs is None or len(cached) >= min(n_docs, 200)):
            df = cached

    if df is None:
        try:
            logger.info("Downloading arXiv abstracts via API (includes publication years)…")
            df = _load_arxiv_api(max(target, 200), seed)
        except Exception as api_exc:
            logger.warning("arXiv API failed (%s); trying HuggingFace", api_exc)
            try:
                df = _load_arxiv_huggingface(max(target, 8000), seed)
            except Exception as exc:
                logger.warning("HuggingFace arXiv load failed (%s); using bundled sample", exc)
                df = _bundled_arxiv_sample(max(target, 400))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Cached arXiv abstracts to %s (%d docs)", cache_path, len(df))

    if "year" not in df.columns:
        df["year"] = df["doc_id"].map(year_from_arxiv_id)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n_docs is not None:
        df = df.head(n_docs).reset_index(drop=True)
    df["corpus"] = "arxiv"
    return df


def _bundled_arxiv_sample(n_docs: int = 400) -> pd.DataFrame:
    """Last-resort scientific abstracts so the pipeline never hard-fails."""
    sample_path = RAW_DIR / "arxiv_sample.json"
    if sample_path.exists():
        return pd.DataFrame(json.loads(Path(sample_path).read_text()))
    topics = [
        ("transformers", "attention", "encoder", "decoder", "self-attention", "sequence"),
        ("topic modeling", "latent dirichlet", "coherence", "document cluster", "word distribution"),
        ("computer vision", "convolution", "residual network", "image recognition", "object detection"),
        ("optimization", "stochastic gradient", "adaptive moment", "learning rate", "convergence"),
        ("generative models", "adversarial", "diffusion", "likelihood", "latent space"),
        ("reinforcement learning", "policy gradient", "reward", "agent", "environment"),
        ("language models", "pretraining", "fine-tuning", "bidirectional", "masked token"),
        ("graph neural", "node embedding", "message passing", "citation network", "homophily"),
    ]
    papers = []
    seed_titles = [t for t, _ in _SAMPLE_ABSTRACTS]
    seed_abs = [a for _, a in _SAMPLE_ABSTRACTS]
    for i in range(max(n_docs, 80)):
        topic = topics[i % len(topics)]
        base = seed_abs[i % len(seed_abs)]
        extra = " ".join(topic[1:])
        papers.append(
            {
                "doc_id": f"synth-{i:04d}",
                "title": f"{topic[0].title()} study {i}: {seed_titles[i % len(seed_titles)]}",
                "text": f"{base} This work focuses on {topic[0]} using {extra}.",
                "label": "cs.LG",
                "label_name": "cs.LG",
                "year": 2016 + (i % 9),
            }
        )
    return pd.DataFrame(papers)


_SAMPLE_ABSTRACTS = [
    (
        "Attention Is All You Need",
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
    ),
    (
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.",
    ),
    (
        "Latent Dirichlet Allocation",
        "We describe latent Dirichlet allocation (LDA), a generative probabilistic model for collections of discrete data such as text corpora. LDA is a three-level hierarchical Bayesian model, in which each item of a collection is modeled as a finite mixture over an underlying set of topics. Each topic is, in turn, modeled as an infinite mixture over an underlying set of topic probabilities. In the context of text modeling, the topic probabilities provide an explicit representation of a document.",
    ),
    (
        "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "BERT and RoBERTa have set a new state-of-the-art performance on sentence-pair regression tasks like semantic textual similarity. However, it requires that both sentences are fed into the network, which causes a massive computational overhead. We present Sentence-BERT (SBERT), a modification of the pretrained BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity.",
    ),
    (
        "UMAP: Uniform Manifold Approximation and Projection",
        "UMAP is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data. The UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance.",
    ),
    (
        "SciBERT: A Pretrained Language Model for Scientific Text",
        "Obtaining large-scale annotated data for NLP tasks in the scientific domain is challenging and expensive. We release SciBERT, a pretrained language model based on BERT to address the lack of high-quality, large-scale labeled scientific data. SciBERT leverages unsupervised pretraining on a large multi-domain corpus of scientific publications to improve performance on downstream scientific NLP tasks.",
    ),
    (
        "Deep Residual Learning for Image Recognition",
        "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.",
    ),
    (
        "Generative Adversarial Nets",
        "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game.",
    ),
    (
        "Adam: A Method for Stochastic Optimization",
        "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.",
    ),
    (
        "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Dropout is a technique for addressing this problem. The key idea is to randomly drop units from the neural network during training. This prevents units from co-adapting too much. Dropout samples from an exponential number of different thinned networks.",
    ),
]


def load_corpus(name: str, n_docs: int | None = None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    name = name.lower().strip()
    if name in {"20newsgroups", "20news", "newsgroups"}:
        return load_20newsgroups(n_docs=n_docs, seed=seed)
    if name in {"arxiv", "arxiv_ml"}:
        return load_arxiv(n_docs=n_docs, seed=seed)
    raise ValueError(f"Unknown corpus '{name}'. Expected one of: 20newsgroups, arxiv")

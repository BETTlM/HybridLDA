"""Text cleaning, tokenization, and lemmatization."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import pandas as pd

from src.config import CACHE_DIR, MIN_TOKEN_LEN, PROCESSED_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"\S+@\S+")
_CITE_RE = re.compile(r"\[[0-9,\s;]+\]|\([A-Z][A-Za-z\-]+ et al\.,? \d{4}\)")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")

_EXTRA_STOPWORDS = {
    "would", "could", "should", "also", "however", "using", "used", "use",
    "one", "two", "may", "might", "among", "within", "across", "via",
    "et", "al", "fig", "figure", "table", "paper", "propose", "proposed",
    "show", "shown", "based", "approach", "method", "methods", "result",
    "results", "experiment", "experimental", "dataset", "data",
}


def _ensure_nltk() -> None:
    import nltk

    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("corpora/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    )
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


@lru_cache(maxsize=1)
def _nlp_tools():
    _ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stops = set(stopwords.words("english")) | _EXTRA_STOPWORDS
    return stops, WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _CITE_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    stops, lemmatizer = _nlp_tools()
    tokens = []
    for tok in clean_text(text).split():
        if len(tok) < MIN_TOKEN_LEN or tok in stops:
            continue
        lemma = lemmatizer.lemmatize(tok)
        if len(lemma) < MIN_TOKEN_LEN or lemma in stops:
            continue
        tokens.append(lemma)
    return tokens


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add `clean_text` and `tokens` columns. Drops empty documents."""
    out = df.copy()
    out["clean_text"] = out["text"].map(clean_text)
    out["tokens"] = out["text"].map(tokenize)
    out = out[out["tokens"].map(len) >= 8].reset_index(drop=True)
    return out


def load_or_preprocess(df: pd.DataFrame, corpus: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Cache preprocessed frames so notebooks and the pipeline share work."""
    path = PROCESSED_DIR / f"{corpus}_preprocessed.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists() and len(pd.read_parquet(path)) >= len(df):
        cached = pd.read_parquet(path)
        cached["tokens"] = cached["tokens"].map(lambda t: list(t) if not isinstance(t, list) else t)
        return cached.head(len(df)).reset_index(drop=True)

    logger.info("Preprocessing %d documents for corpus=%s", len(df), corpus)
    processed = preprocess_frame(df)
    processed.to_parquet(path, index=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return processed

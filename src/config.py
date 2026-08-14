"""Project-wide paths and experiment defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _path in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR):
    _path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Default experiment size is laptop-friendly. Scale up via CLI / notebooks.
DEFAULT_N_DOCS = 2000
FULL_N_DOCS = 8000

# Default K grid (fast). Full sweep from the plan: [5, 10, 15, 20, 25, 30]
DEFAULT_K_GRID = (8, 12, 16)
FULL_K_GRID = (5, 10, 15, 20, 25, 30)

DATASETS = ("20newsgroups", "arxiv")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
SPECTER_MODEL = "allenai-specter"

GAMMA = 0.5  # α in the abstract: weight on the LDA topic vector
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TOP_N_WORDS = 10
UMAP_COMPONENTS = 5
UMAP_N_NEIGHBORS = 15
PCA_COMPONENTS = 20
TSNE_COMPONENTS = 2

LDA_PASSES = 10
LDA_ITERATIONS = 50
LDA_CHUNKSIZE = 200
MIN_DF = 5
MAX_DF = 0.5
MIN_TOKEN_LEN = 3

ARXIV_CATEGORIES = ("cs.LG", "cs.AI", "cs.CL", "cs.CV")
ARXIV_HF_DATASET = "CShorten/ML-ArXiv-Papers"

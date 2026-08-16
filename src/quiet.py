"""Silence third-party noise and pin caches to writable project paths.

Call ``configure_warnings()`` as the first import in every entry point.
Must run *before* umap/numba/matplotlib so their cache locators exist.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "data" / "cache"


def _require_supported_python() -> None:
    if (3, 9) <= sys.version_info[:2] <= (3, 12):
        return
    ver = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"This project needs Python 3.9–3.12 (you have {ver}).\n"
        "Recreate the venv with ./setup.sh (Windows: setup.bat).\n"
    )


def configure_warnings() -> None:
    try:
        from src.python_compat import require_supported_python

        require_supported_python()
    except ImportError:
        _require_supported_python()

    numba_dir = _CACHE / "numba"
    mpl_dir = _CACHE / "matplotlib"
    hf_dir = _CACHE / "huggingface"
    st_dir = _CACHE / "sentence_transformers"
    nltk_dir = _ROOT / "data" / "nltk_data"
    sklearn_dir = _CACHE / "sklearn"
    for path in (numba_dir, mpl_dir, hf_dir, st_dir, nltk_dir, sklearn_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("HF_HOME", str(hf_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(st_dir))
    os.environ.setdefault("NLTK_DATA", str(nltk_dir))
    os.environ.setdefault("SCIKIT_LEARN_DATA", str(sklearn_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["PYTHONWARNINGS"] = "ignore"

    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ImportWarning)
    try:
        from urllib3.exceptions import NotOpenSSLWarning

        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except Exception:
        pass

    for name in (
        "gensim",
        "gensim.models",
        "gensim.corpora",
        "gensim.utils",
        "gensim.topic_coherence",
        "gensim.topic_coherence.text_analysis",
        "gensim.topic_coherence.probability_estimation",
        "sentence_transformers",
        "umap",
        "numba",
        "numba.core",
        "matplotlib",
        "matplotlib.font_manager",
        "urllib3",
        "huggingface_hub",
        "datasets",
        "sklearn",
        "torch",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

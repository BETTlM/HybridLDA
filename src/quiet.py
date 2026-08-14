"""Silence known third-party noise before gensim/UMAP/streamlit import workers.

Call ``configure_warnings()`` as the first import in every entry point.
Child processes inherit PYTHONWARNINGS, which stops urllib3 from reprinting
NotOpenSSLWarning once per coherence worker (hundreds of lines otherwise).
"""

from __future__ import annotations

import logging
import os
import warnings


def configure_warnings() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
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
        "matplotlib",
        "matplotlib.font_manager",
        "urllib3",
        "huggingface_hub",
        "datasets",
        "sklearn",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

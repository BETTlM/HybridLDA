"""Hybrid LDA + semantic embedding topic modeling."""

from src.quiet import configure_warnings

configure_warnings()

from src.config import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR

__all__ = ["PROJECT_ROOT", "DATA_DIR", "RESULTS_DIR", "FIGURES_DIR"]

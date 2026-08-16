#!/usr/bin/env bash
# Create a local virtualenv and install dependencies (macOS / Linux).
# Requires Python 3.9–3.12. Homebrew's current `python3` is often 3.14 and will fail.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pick_python() {
  local cand
  local -a candidates=(
    python3.12 python3.11 python3.10 python3.9
    /opt/homebrew/opt/python@3.12/bin/python3.12
    /opt/homebrew/opt/python@3.11/bin/python3.11
    /opt/homebrew/opt/python@3.10/bin/python3.10
    /usr/local/opt/python@3.12/bin/python3.12
    /usr/local/opt/python@3.11/bin/python3.11
    python3 python
  )
  for cand in "${candidates[@]}"; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "Python 3.9–3.12 is required. Homebrew python3 is often 3.14, which cannot install this stack." >&2
  echo "On macOS:  brew install python@3.12 && ./setup.sh" >&2
  echo "Or install 3.12 from https://www.python.org/downloads/" >&2
  exit 1
fi

echo "Using $($PY -c 'import sys; print(sys.executable)') ($($PY -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

if [ -x "$ROOT/.venv/bin/python" ]; then
  if ! "$ROOT/.venv/bin/python" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
    echo "Existing .venv is $($ROOT/.venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || echo unknown) — recreating with a supported Python."
    rm -rf "$ROOT/.venv"
  fi
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi

VENV_PY="$ROOT/.venv/bin/python"
export NLTK_DATA="$ROOT/data/nltk_data"
export NUMBA_CACHE_DIR="$ROOT/data/cache/numba"
export MPLCONFIGDIR="$ROOT/data/cache/matplotlib"
mkdir -p "$NLTK_DATA" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

echo "Installing packages from requirements.txt ..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install --prefer-binary -r "$ROOT/requirements.txt"

echo "Downloading NLTK data ..."
"$VENV_PY" - <<'PY'
import os
from pathlib import Path
import nltk

nltk_dir = Path("data/nltk_data")
nltk_dir.mkdir(parents=True, exist_ok=True)
os.environ["NLTK_DATA"] = str(nltk_dir.resolve())
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, download_dir=str(nltk_dir), quiet=True)
    print(f"  nltk:{pkg}")
PY

echo "Checking imports ..."
"$VENV_PY" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from src.quiet import configure_warnings
configure_warnings()
import numpy, pandas, sklearn, scipy, gensim, nltk
import torch
from sentence_transformers import SentenceTransformer
import umap
import matplotlib
print("imports ok")
print("  python", ".".join(map(str, __import__("sys").version_info[:3])))
print("  numpy", numpy.__version__)
print("  gensim", gensim.__version__)
print("  torch", torch.__version__)
print("  umap", umap.__version__)
PY

echo
echo "Setup finished."
echo "Activate THIS venv (not a conda env named Kaushik):"
echo
echo "  source .venv/bin/activate"
echo "  python run_pipeline.py --smoke"
echo
echo "Full experiment:"
echo "  python run_pipeline.py"
echo
echo "Optional:"
echo "  python run_pipeline.py --full"
echo "  python run_dashboard.py"
echo

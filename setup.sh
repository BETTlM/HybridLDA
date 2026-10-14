#!/usr/bin/env bash
# Create a local virtualenv and install dependencies (macOS / Linux).
# Supports Python 3.9–3.15. Uses binary wheels for compiled packages.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ONLY_BIN="numpy,pandas,scipy,scikit-learn,numba,llvmlite,pyarrow,pillow,matplotlib,torch,PyYAML"

wheel_fail() {
  echo
  echo "=========================================================================="
  echo "  pip could not find a BINARY wheel (no compiler / Visual Studio needed)."
  echo "=========================================================================="
  echo
  echo "A package likely has no wheel for this Python yet."
  echo "Recommended fix: use Python 3.12, then:"
  echo "  rm -rf .venv && ./setup.sh"
  echo
  exit 1
}

pick_python() {
  local cand
  local -a candidates=(
    python3.15 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9
    /opt/homebrew/opt/python@3.15/bin/python3.15
    /opt/homebrew/opt/python@3.14/bin/python3.14
    /opt/homebrew/opt/python@3.13/bin/python3.13
    /opt/homebrew/opt/python@3.12/bin/python3.12
    /opt/homebrew/opt/python@3.11/bin/python3.11
    /opt/homebrew/opt/python@3.10/bin/python3.10
    /usr/local/opt/python@3.15/bin/python3.15
    /usr/local/opt/python@3.14/bin/python3.14
    /usr/local/opt/python@3.13/bin/python3.13
    /usr/local/opt/python@3.12/bin/python3.12
    /usr/local/opt/python@3.11/bin/python3.11
    python3 python
  )
  for cand in "${candidates[@]}"; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 15) else 1)' 2>/dev/null; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "Python 3.9–3.15 is required." >&2
  echo "On macOS:  brew install python@3.12   # or python@3.15" >&2
  echo "Or install from https://www.python.org/downloads/" >&2
  exit 1
fi

echo "Using $($PY -c 'import sys; print(sys.executable)') ($($PY -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

if [ -x "$ROOT/.venv/bin/python" ]; then
  if ! "$ROOT/.venv/bin/python" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 15) else 1)' 2>/dev/null; then
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
export HF_HOME="$ROOT/data/cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/data/cache/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$NLTK_DATA" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$ROOT/results/figures"

echo "Upgrading pip / setuptools / wheel ..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

PY_MINOR="$("$VENV_PY" -c 'import sys; print(sys.version_info.minor)')"

# PyTorch cp315 wheels are on download.pytorch.org, not PyPI.
if [ "$PY_MINOR" -ge 15 ]; then
  echo "Python 3.${PY_MINOR}: installing torch from download.pytorch.org ..."
  "$VENV_PY" -m pip install --only-binary=:all: "torch>=2.2.0,<3" \
    --index-url https://download.pytorch.org/whl/cpu || wheel_fail

  echo "Python 3.${PY_MINOR}: requiring packages with cp315 wheels ..."
  "$VENV_PY" -m pip install --only-binary="$ONLY_BIN" \
    "numpy>=2.5.0" "pandas>=3.0.6" "scipy>=1.18.0" "scikit-learn>=1.9.0" \
    "pyarrow>=26.0.0" "pillow>=12.0.0" "numba>=0.68.0" "llvmlite>=0.50.0" \
    "matplotlib>=3.10.0" || wheel_fail
fi

echo "Installing packages from requirements.txt ..."
"$VENV_PY" -m pip install --prefer-binary --only-binary="$ONLY_BIN" \
  -r "$ROOT/requirements.txt" || wheel_fail

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
import streamlit
print("imports ok")
print("  python", ".".join(map(str, __import__("sys").version_info[:3])))
print("  numpy", numpy.__version__)
print("  pandas", pandas.__version__)
print("  torch", torch.__version__)
print("  umap", umap.__version__)
print("  streamlit", streamlit.__version__)
PY

echo
echo "Setup finished."
echo
echo "  source .venv/bin/activate"
echo "  python run_pipeline.py --smoke"
echo

#!/usr/bin/env bash
# Create a local virtualenv and install dependencies (macOS / Linux).
# After this finishes:
#   source .venv/bin/activate
#   python run_pipeline.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

pick_python() {
  local cand
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

if ! PY="$(pick_python)"; then
  echo "Python 3.9 or newer is required." >&2
  echo "Install Python from https://www.python.org/downloads/ and re-run ./setup.sh" >&2
  exit 1
fi

echo "Using $($PY -c 'import sys; print(sys.executable)') ($($PY -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'))"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi

VENV_PY="$ROOT/.venv/bin/python"
echo "Installing packages from requirements.txt ..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r "$ROOT/requirements.txt"

echo "Downloading NLTK data ..."
"$VENV_PY" - <<'PY'
import nltk

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
        print(f"  nltk:{pkg}")
    except Exception as exc:
        print(f"  nltk:{pkg} skipped ({exc})")
PY

echo
echo "Setup finished."
echo "Activate the environment, then run the pipeline with the same command on every OS:"
echo
echo "  source .venv/bin/activate"
echo "  python run_pipeline.py"
echo
echo "Optional:"
echo "  python run_pipeline.py --full"
echo "  python run_dashboard.py"
echo

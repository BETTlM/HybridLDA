#!/bin/bash
set -e

echo "=========================================================="
echo "   Hybrid LDA-LLM Topic Modeling - End-to-End Execution   "
echo "=========================================================="

# 1. Activate the virtual environment if it was created via setup.sh
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[!] No .venv directory found. Assuming global Python environment."
fi

# 2. Run the main analysis pipeline
# Any arguments passed to this script (e.g., ./run.sh --smoke or ./run.sh --full) 
# will be forwarded to the pipeline.
echo "[*] Running the analysis pipeline (run_pipeline.py)..."
python run_pipeline.py "$@"

echo "[*] Pipeline execution completed successfully."

# 3. Launch the Streamlit dashboard
# Note: run_dashboard.py is a Python wrapper that automatically executes 
# `streamlit run app/streamlit_app.py` while safely silencing terminal warnings.
echo "[*] Starting the Streamlit dashboard..."
python run_dashboard.py

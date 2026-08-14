#!/usr/bin/env python3
"""Launch the dashboard without third-party import warnings.

    python run_dashboard.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quiet import configure_warnings

configure_warnings()

from streamlit.web import cli as stcli

app = str(ROOT / "app" / "streamlit_app.py")
extra = sys.argv[1:]
sys.argv = ["streamlit", "run", app, *extra]
raise SystemExit(stcli.main())

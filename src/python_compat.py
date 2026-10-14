"""Fail fast on Python versions that cannot install the scientific stack."""

from __future__ import annotations

import sys

# Keep in sync with setup.sh / setup.bat / src/quiet.py
MIN_PY = (3, 9)
MAX_PY = (3, 15)


def require_supported_python() -> None:
    if MIN_PY <= sys.version_info[:2] <= MAX_PY:
        return
    ver = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"This project needs Python {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]} "
        f"(you have {ver}).\n"
        "Recreate the venv with ./setup.sh (Windows: setup.bat).\n"
    )

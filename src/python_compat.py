"""Fail fast on Python versions that cannot install the scientific stack."""

from __future__ import annotations

import sys


def require_supported_python() -> None:
    if (3, 9) <= sys.version_info[:2] <= (3, 12):
        return
    ver = ".".join(map(str, sys.version_info[:3]))
    raise SystemExit(
        f"This project needs Python 3.9–3.12 (you have {ver}).\n"
        "Recreate the venv with ./setup.sh (Windows: setup.bat).\n"
    )

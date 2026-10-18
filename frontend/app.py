"""Compatibility launcher for the current ForteAI interface."""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "cash_ml" / "app.py"),
    run_name="__main__",
)

"""Launchable wrapper for the six-cancer fixed-graph train-only fast audit."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOG = HERE / "run_nested_all6_fast_background.log"

os.environ["NESTED_CANCERS"] = "LUAD,LIHC,KIRC,COAD,STAD,HNSC"
os.environ["NESTED_BOOTSTRAP"] = "3"
os.environ["NESTED_RESUME"] = "1"

with LOG.open("w", encoding="utf-8") as stream:
    sys.stdout = stream
    sys.stderr = stream
    os.chdir(HERE)
    runpy.run_path(str(HERE / "nested_deleakage.py"), run_name="__main__")


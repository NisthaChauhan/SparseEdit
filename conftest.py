"""Pytest root conftest — ensures project root is on sys.path so that
top-level scripts (ui_backend.py, app.py, run_edit.py, etc.) are importable
from the tests/ directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#!/usr/bin/env python3
"""
Run the face pipeline from the project root.

  conda activate ml
  python run.py doctor
  python run.py test

See README.md for full usage.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())

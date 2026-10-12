#!/usr/bin/env python3
"""Запуск GUI приложения Music Framework."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import main

if __name__ == "__main__":
    main()

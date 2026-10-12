#!/usr/bin/env python3
"""Запуск CLI Music Framework (если пакет не установлен)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from music_framework.cli import main

if __name__ == "__main__":
    sys.exit(main())

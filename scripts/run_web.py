#!/usr/bin/env python3
"""
Запуск веб-интерфейса Music Framework.
Откроется в браузере: http://127.0.0.1:7860
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.web_app import main

if __name__ == "__main__":
    main()

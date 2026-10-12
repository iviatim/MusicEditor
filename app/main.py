"""
Точка входа GUI: CustomTkinter, тёмная тема, вкладки по модулям фреймворка.
"""

import sys
from pathlib import Path

# Добавляем корень проекта и src в path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import customtkinter as ctk
from app.ui.main_window import MainWindow


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ctk.CTk()
    app.title("Music Framework — Цифровая обработка музыки с нейросетями")
    app.geometry("1100x720")
    app.minsize(900, 600)
    win = MainWindow(app)
    win.pack(fill="both", expand=True)
    app.mainloop()


if __name__ == "__main__":
    main()

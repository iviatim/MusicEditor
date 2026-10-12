"""
Главное окно: вкладки — Загрузка/Анализ, Разделение, Семантика, Аугментация, Метрики.
"""

import customtkinter as ctk
from app.ui.tabs import (
    LoadAnalyzeTab,
    SeparationTab,
    SemanticTab,
    AugmentationTab,
    MetricsTab,
)


class MainWindow(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=("gray92", "gray14"))
        self._build_ui()

    def _build_ui(self):
        # Заголовок
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        title = ctk.CTkLabel(
            header,
            text="Music Framework",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=("#1a1a2e", "#eaeaea"),
        )
        title.pack(side="left")
        subtitle = ctk.CTkLabel(
            header,
            text="Цифровая обработка музыки с нейросетями · Локальный фреймворк",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
        )
        subtitle.pack(side="left", padx=(15, 0), pady=(8, 0))

        # Вкладки
        self.tabview = ctk.CTkTabview(self, width=900, height=600)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.tabview.configure(
            segmented_button_fg_color=("gray85", "gray20"),
            segmented_button_selected_color=("#3b82f6", "#2563eb"),
            segmented_button_selected_hover_color=("#2563eb", "#1d4ed8"),
            text_color=("gray10", "gray90"),
        )

        self.tabview.add("Загрузка и анализ")
        self.tabview.add("Разделение треков (Demucs)")
        self.tabview.add("Темп и тон (Pyrubberband)")
        self.tabview.add("Аугментация")
        self.tabview.add("Метрики SDR/SAR")

        self.tabview.set("Загрузка и анализ")

        # Контент вкладок
        LoadAnalyzeTab(self.tabview.tab("Загрузка и анализ"))
        SeparationTab(self.tabview.tab("Разделение треков (Demucs)"))
        SemanticTab(self.tabview.tab("Темп и тон (Pyrubberband)"))
        AugmentationTab(self.tabview.tab("Аугментация"))
        MetricsTab(self.tabview.tab("Метрики SDR/SAR"))

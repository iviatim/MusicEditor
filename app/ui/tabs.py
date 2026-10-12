"""
Вкладки главного окна: загрузка/анализ, разделение, семантика, аугментация, метрики.
"""

import threading
from pathlib import Path

import customtkinter as ctk
import numpy as np

# Импорты фреймворка (могут быть недоступны при отсутствии зависимостей)
def _safe_import(module_path, name, default=None):
    try:
        mod = __import__(module_path, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return default

load_audio = _safe_import("music_framework.core.io", "load_audio")
save_audio = _safe_import("music_framework.core.io", "save_audio")
get_audio_info = _safe_import("music_framework.core.io", "get_audio_info")
AudioAnalyzer = _safe_import("music_framework.core.analysis", "AudioAnalyzer")
DemucsSeparator = _safe_import("music_framework.separation.demucs_separator", "DemucsSeparator")
SemanticTransformer = _safe_import("music_framework.semantic.transform", "SemanticTransformer")
AugmentationPipeline = _safe_import("music_framework.augmentation.pipeline", "AugmentationPipeline")
get_preset_pipelines = _safe_import("music_framework.augmentation.pipeline", "get_preset_pipelines")
SeparationMetrics = _safe_import("music_framework.metrics.separation_metrics", "SeparationMetrics")


def _run_in_thread(target, on_done=None):
    def run():
        try:
            target()
        except Exception as e:
            if on_done:
                on_done(str(e))
            return
        if on_done:
            on_done(None)
    t = threading.Thread(target=run, daemon=True)
    t.start()


def _file_dialog(parent, title="Выберите файл", filetypes=None):
    from tkinter import filedialog
    return filedialog.askopenfilename(
        parent=parent.winfo_toplevel(),
        title=title,
        filetypes=filetypes or [
            ("Аудио", "*.wav *.flac *.ogg *.mp3"),
            ("WAV", "*.wav"),
            ("Все", "*.*"),
        ],
    )


def _folder_dialog(parent, title="Выберите папку"):
    from tkinter import filedialog
    return filedialog.askdirectory(parent=parent.winfo_toplevel(), title=title)


# --- Загрузка и анализ ---
class LoadAnalyzeTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._path_var = ctk.StringVar(value="")
        self._log_var = ctk.StringVar(value="Выберите файл и нажмите «Анализ».")
        self._build_ui()

    def _build_ui(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", pady=5)
        ctk.CTkLabel(row, text="Файл:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(row, textvariable=self._path_var, width=500).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="Обзор…", width=100, command=self._browse).pack(side="left")
        ctk.CTkButton(row, text="Анализ", width=120, command=self._analyze).pack(side="left", padx=(10, 0))

        self._log = ctk.CTkTextbox(self, height=200, font=ctk.CTkFont(family="Consolas", size=12))
        self._log.pack(fill="both", expand=True, pady=10)
        self._log.insert("1.0", self._log_var.get())

    def _browse(self):
        path = _file_dialog(self)
        if path:
            self._path_var.set(path)

    def _analyze(self):
        path = self._path_var.get().strip()
        if not path or not Path(path).exists():
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Укажите существующий файл.")
            return
        self._log.delete("1.0", "end")
        self._log.insert("1.0", "Анализ… Подождите.\n")

        def do():
            info = get_audio_info(path) if get_audio_info else None
            analyzer = AudioAnalyzer(sr=22050) if AudioAnalyzer else None
            if not analyzer:
                self.after(0, lambda: self._set_log("Модуль анализа (Librosa) недоступен."))
                return
            result = analyzer.load_and_analyze(path)
            lines = [result.summary(), f"Длительность: {result.duration_sec:.2f} с", f"Темп: {result.tempo_bpm:.1f} BPM"]
            if info:
                lines.insert(0, str(info))
            self.after(0, lambda: self._set_log("\n".join(lines)))

        _run_in_thread(do)

    def _set_log(self, text):
        self._log.delete("1.0", "end")
        self._log.insert("1.0", text)


# --- Разделение (Demucs) ---
class SeparationTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._input_var = ctk.StringVar(value="")
        self._output_var = ctk.StringVar(value="")
        self._log_var = ctk.StringVar(value="Укажите входной файл и папку вывода.")
        self._build_ui()

    def _build_ui(self):
        r1 = ctk.CTkFrame(self, fg_color="transparent")
        r1.pack(fill="x", pady=5)
        ctk.CTkLabel(r1, text="Вход:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r1, textvariable=self._input_var, width=400).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r1, text="Файл…", width=80, command=lambda: self._set_path(self._input_var, is_file=True)).pack(side="left")
        r2 = ctk.CTkFrame(self, fg_color="transparent")
        r2.pack(fill="x", pady=5)
        ctk.CTkLabel(r2, text="Папка вывода:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r2, textvariable=self._output_var, width=400).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r2, text="Папка…", width=80, command=lambda: self._set_path(self._output_var, is_file=False)).pack(side="left")
        ctk.CTkButton(self, text="Разделить (Demucs)", width=180, command=self._separate).pack(anchor="w", pady=10)
        self._log = ctk.CTkTextbox(self, height=250, font=ctk.CTkFont(family="Consolas", size=12))
        self._log.pack(fill="both", expand=True, pady=5)
        self._log.insert("1.0", self._log_var.get())

    def _set_path(self, var, is_file=True):
        if is_file:
            path = _file_dialog(self)
        else:
            path = _folder_dialog(self)
        if path:
            var.set(path)

    def _separate(self):
        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp or not Path(inp).exists():
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Укажите существующий входной файл.")
            return
        if not out:
            out = str(Path(inp).parent / "separated")
        if not DemucsSeparator:
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Demucs не установлен (pip install demucs).")
            return
        self._log.delete("1.0", "end")
        self._log.insert("1.0", "Разделение… Может занять несколько минут.\n")

        def do():
            sep = DemucsSeparator()
            result = sep.separate_file(inp)
            paths = result.save_all(out, prefix=Path(inp).stem)
            msg = "Готово. Источники: " + ", ".join(result.keys()) + "\nСохранено: " + "\n".join(str(p) for p in paths)
            self.after(0, lambda: self._set_log(msg))

        def on_done(err):
            if err:
                self.after(0, lambda: self._set_log("Ошибка: " + err))

        _run_in_thread(do, on_done=on_done)

    def _set_log(self, text):
        self._log.delete("1.0", "end")
        self._log.insert("1.0", text)


# --- Семантика (темп/тон) ---
class SemanticTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._path_var = ctk.StringVar(value="")
        self._tempo_var = ctk.StringVar(value="1.0")
        self._pitch_var = ctk.StringVar(value="0")
        self._build_ui()

    def _build_ui(self):
        r = ctk.CTkFrame(self, fg_color="transparent")
        r.pack(fill="x", pady=5)
        ctk.CTkLabel(r, text="Файл:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r, textvariable=self._path_var, width=400).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r, text="Обзор…", command=lambda: self._set_path()).pack(side="left")
        r2 = ctk.CTkFrame(self, fg_color="transparent")
        r2.pack(fill="x", pady=8)
        ctk.CTkLabel(r2, text="Темп (коэф.):").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r2, textvariable=self._tempo_var, width=80).pack(side="left", padx=(0, 20))
        ctk.CTkLabel(r2, text="Тон (полутоны):").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r2, textvariable=self._pitch_var, width=80).pack(side="left")
        ctk.CTkButton(self, text="Применить и сохранить…", width=200, command=self._apply).pack(anchor="w", pady=10)
        self._log = ctk.CTkTextbox(self, height=150, font=ctk.CTkFont(family="Consolas", size=12))
        self._log.pack(fill="both", expand=True, pady=5)
        self._log.insert("1.0", "Pyrubberband: темп и тон. Результат: <имя>_transformed.wav в той же папке.")

    def _set_path(self):
        path = _file_dialog(self)
        if path:
            self._path_var.set(path)

    def _apply(self):
        path = self._path_var.get().strip()
        if not path or not Path(path).exists():
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Укажите существующий файл.")
            return
        if not SemanticTransformer:
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "pyrubberband не установлен.")
            return
        try:
            tempo = float(self._tempo_var.get().replace(",", "."))
            pitch = float(self._pitch_var.get().replace(",", "."))
        except ValueError:
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Темп и тон — числа.")
            return
        out = str(Path(path).parent / (Path(path).stem + "_transformed.wav"))

        def do():
            tr = SemanticTransformer()
            tr.process_file(path, out_path=out, tempo_ratio=tempo, pitch_semitones=pitch)
            self.after(0, lambda: self._set_log("Сохранено: " + out))

        _run_in_thread(do)

    def _set_log(self, text):
        self._log.delete("1.0", "end")
        self._log.insert("1.0", text)


# --- Аугментация ---
class AugmentationTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._path_var = ctk.StringVar(value="")
        self._preset_var = ctk.StringVar(value="default")
        self._build_ui()

    def _build_ui(self):
        r = ctk.CTkFrame(self, fg_color="transparent")
        r.pack(fill="x", pady=5)
        ctk.CTkLabel(r, text="Файл:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r, textvariable=self._path_var, width=400).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r, text="Обзор…", command=self._browse).pack(side="left")
        r2 = ctk.CTkFrame(self, fg_color="transparent")
        r2.pack(fill="x", pady=8)
        ctk.CTkLabel(r2, text="Пресет:").pack(side="left", padx=(0, 8))
        ctk.CTkOptionMenu(r2, values=["light", "default", "aggressive"], variable=self._preset_var, width=150).pack(side="left")
        ctk.CTkButton(self, text="Применить аугментацию и сохранить…", width=260, command=self._apply).pack(anchor="w", pady=10)
        self._log = ctk.CTkTextbox(self, height=180, font=ctk.CTkFont(family="Consolas", size=12))
        self._log.pack(fill="both", expand=True, pady=5)
        self._log.insert("1.0", "Audiomentations: light / default / aggressive.")

    def _browse(self):
        path = _file_dialog(self)
        if path:
            self._path_var.set(path)

    def _apply(self):
        path = self._path_var.get().strip()
        if not path or not Path(path).exists():
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Укажите существующий файл.")
            return
        if not AugmentationPipeline or not get_preset_pipelines:
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "audiomentations не установлен.")
            return
        out = str(Path(path).parent / (Path(path).stem + "_aug.wav"))

        def do():
            pipelines = get_preset_pipelines(44100)
            pipe = pipelines.get(self._preset_var.get(), pipelines["default"])
            pipe.transform_file(path, out_path=out, sample_rate=44100)
            self.after(0, lambda: self._set_log("Сохранено: " + out))

        _run_in_thread(do)

    def _set_log(self, text):
        self._log.delete("1.0", "end")
        self._log.insert("1.0", text)


# --- Метрики SDR/SAR ---
class MetricsTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._ref_var = ctk.StringVar(value="")
        self._est_var = ctk.StringVar(value="")
        self._sources_var = ctk.StringVar(value="vocals,drums,bass,other")
        self._build_ui()

    def _build_ui(self):
        r1 = ctk.CTkFrame(self, fg_color="transparent")
        r1.pack(fill="x", pady=5)
        ctk.CTkLabel(r1, text="Папка эталонов:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r1, textvariable=self._ref_var, width=350).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r1, text="Обзор…", command=lambda: self._set_dir(self._ref_var)).pack(side="left")
        r2 = ctk.CTkFrame(self, fg_color="transparent")
        r2.pack(fill="x", pady=5)
        ctk.CTkLabel(r2, text="Папка оценок:").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r2, textvariable=self._est_var, width=350).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r2, text="Обзор…", command=lambda: self._set_dir(self._est_var)).pack(side="left")
        r3 = ctk.CTkFrame(self, fg_color="transparent")
        r3.pack(fill="x", pady=5)
        ctk.CTkLabel(r3, text="Источники (через запятую):").pack(side="left", padx=(0, 8))
        ctk.CTkEntry(r3, textvariable=self._sources_var, width=300).pack(side="left")
        ctk.CTkButton(self, text="Вычислить метрики", width=180, command=self._compute).pack(anchor="w", pady=10)
        self._log = ctk.CTkTextbox(self, height=220, font=ctk.CTkFont(family="Consolas", size=12))
        self._log.pack(fill="both", expand=True, pady=5)
        self._log.insert("1.0", "Torchmetrics: SDR, SAR, SIR. Укажите папки с эталонами и оценками (файлы source.wav).")

    def _set_dir(self, var):
        path = _folder_dialog(self)
        if path:
            var.set(path)

    def _compute(self):
        ref_dir = self._ref_var.get().strip()
        est_dir = self._est_var.get().strip()
        sources = [s.strip() for s in self._sources_var.get().split(",") if s.strip()]
        if not ref_dir or not Path(ref_dir).exists():
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Укажите папку эталонов.")
            return
        if not est_dir or not Path(est_dir).exists():
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "Укажите папку оценок.")
            return
        if not SeparationMetrics:
            self._log.delete("1.0", "end")
            self._log.insert("1.0", "torchmetrics не установлен.")
            return
        self._log.delete("1.0", "end")
        self._log.insert("1.0", "Вычисление…\n")

        def do():
            metrics = SeparationMetrics()
            report = metrics.evaluate_from_files(ref_dir, est_dir, sources, sample_rate=44100)
            self.after(0, lambda: self._set_log(report.summary()))

        _run_in_thread(do)

    def _set_log(self, text):
        self._log.delete("1.0", "end")
        self._log.insert("1.0", text)

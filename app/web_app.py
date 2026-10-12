"""
Веб-интерфейс Music Framework на Gradio.
Запуск: python app/web_app.py
Откроется в браузере по адресу http://127.0.0.1:7860
"""

import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

# На Windows: чтобы pyrubberband нашёл rubberband, добавьте папку с rubberband.exe в PATH.
# Можно задать переменную RUBBERBAND_PATH = полный путь к rubberband.exe — тогда её папка будет добавлена в PATH.
if sys.platform == "win32":
    rb_path = os.environ.get("RUBBERBAND_PATH", "").strip()
    if rb_path and os.path.isfile(rb_path):
        rb_dir = os.path.dirname(rb_path)
        os.environ["PATH"] = rb_dir + os.pathsep + os.environ.get("PATH", "")

import gradio as gr

from music_framework.core.io import load_audio, save_audio, get_audio_info
from music_framework.core.analysis import AudioAnalyzer
from music_framework.separation.demucs_separator import DemucsSeparator
from music_framework.semantic.transform import SemanticTransformer
from music_framework.augmentation.pipeline import get_preset_pipelines
from music_framework.metrics.separation_metrics import SeparationMetrics

# Кэш моделей — один раз загрузили, дальше без ожидания
_demucs_create_lock = threading.Lock()
_demucs_use_lock = threading.Lock()
_demucs_instance = None
_semantic_instance = None


def _get_demucs():
    global _demucs_instance
    with _demucs_create_lock:
        if _demucs_instance is None:
            _demucs_instance = DemucsSeparator(model_name="htdemucs")
        return _demucs_instance


def _get_semantic():
    """Один экземпляр SemanticTransformer — быстрый повторный вызов темпа/тона."""
    global _semantic_instance
    if SemanticTransformer is None:
        return None
    if _semantic_instance is None:
        _semantic_instance = SemanticTransformer()
    return _semantic_instance


# --- Вспомогательное: путь из значения Gradio Audio (str / dict / list) ---
def _audio_path(audio_file):
    """Извлечь путь к файлу из значения компонента gr.Audio (поддержка str, dict, list)."""
    if audio_file is None:
        return None
    if isinstance(audio_file, str):
        return audio_file
    if isinstance(audio_file, dict) and "path" in audio_file:
        return audio_file["path"]
    if isinstance(audio_file, (list, tuple)) and len(audio_file) > 0:
        first = audio_file[0]
        return first if isinstance(first, str) else getattr(first, "path", getattr(first, "name", None))
    return getattr(audio_file, "path", None) or getattr(audio_file, "name", None)

# --- Обработчики ---

def run_analyze(audio_file):
    if audio_file is None:
        return "Загрузите аудиофайл и нажмите «Анализ»."
    path = _audio_path(audio_file)
    if not path or not Path(path).exists():
        return "Файл не найден."
    try:
        info = get_audio_info(path) if get_audio_info else None
        analyzer = AudioAnalyzer(sr=22050) if AudioAnalyzer else None
        if not analyzer:
            return "Модуль анализа (Librosa) недоступен. Установите: pip install librosa"
        result = analyzer.load_and_analyze(path)
        lines = [result.summary(), f"Длительность: {result.duration_sec:.2f} с", f"Темп: {result.tempo_bpm:.1f} BPM"]
        if info:
            lines.insert(0, str(info))
        return "\n".join(lines)
    except Exception as e:
        return f"Ошибка: {e}"


def _normalize_path(p):
    if p is None:
        return None
    # Gradio может передать str, dict с "path", или tuple (path, sample_rate) — всегда берём путь
    if isinstance(p, str):
        path = p
    elif isinstance(p, dict) and "path" in p:
        path = p["path"]
    elif isinstance(p, (list, tuple)) and len(p) > 0:
        first = p[0]
        if isinstance(first, str):
            path = first
        elif isinstance(first, dict):
            path = first.get("path") or first.get("name")
        else:
            path = getattr(first, "path", getattr(first, "name", None))
    else:
        path = _audio_path(p)
    path = str(path).strip() if path else None
    return path if path and Path(path).exists() else None


def run_separate_vocal_inst(audio_file, progress=gr.Progress()):
    """Вокал + инструментал: возвращает (state, mix_path, status). После разделения показывается одна дорожка микса."""
    path = _normalize_path(audio_file)
    if not path:
        return (None, None), None, "Загрузите трек."
    sep = _get_demucs()
    if sep is None:
        return (None, None), None, "Demucs не установлен."
    try:
        progress(0.1, desc="Разделение…")
        with _demucs_use_lock:
            result = sep.separate_file(path, clean=True)
        out_dir = Path(tempfile.mkdtemp(prefix="music_framework_sep_"))
        progress(0.85, desc="Сохранение…")
        result.save_all(out_dir, prefix=Path(path).stem)
        vocals_path = out_dir / f"{Path(path).stem}_vocals.wav"
        instrumental_path = out_dir / f"{Path(path).stem}_instrumental.wav"
        if save_audio and result.sources:
            sr = result.sample_rate
            parts = []
            for key in ("drums", "bass", "other"):
                if key in result.sources:
                    parts.append(result.get(key))
            if parts:
                inst = np.clip(sum(parts), -1.0, 1.0).astype(np.float32)
                save_audio(instrumental_path, inst, sr)
        v_str = str(vocals_path) if vocals_path.exists() else None
        i_str = str(instrumental_path) if instrumental_path.exists() else None
        state = (v_str, i_str)
        mix_path = _mix_two(v_str, i_str, 0.5) if (v_str and i_str) else (v_str or i_str)
        progress(1.0)
        return state, mix_path, "Готово."
    except Exception as e:
        return (None, None), None, f"Ошибка: {e}"


def _mix_two(path_v, path_i, vocal_gain):
    """Смешать два моно/стерео файла: vocal_gain 0..1 (остальное — инструментал). Возвращает путь к WAV."""
    if not path_v or not path_i or not load_audio or not save_audio:
        return None
    try:
        v, sr_v = load_audio(path_v, mono=True)
        i, sr_i = load_audio(path_i, mono=True)
        sr = max(sr_v, sr_i)
        if sr_v != sr or sr_i != sr:
            import librosa
            if sr_v != sr:
                v = librosa.resample(v.astype(np.float32), orig_sr=sr_v, target_sr=sr)
            if sr_i != sr:
                i = librosa.resample(i.astype(np.float32), orig_sr=sr_i, target_sr=sr)
        n = min(len(v), len(i))
        v, i = v[:n].astype(np.float32), i[:n].astype(np.float32)
        mix = np.clip(vocal_gain * v + (1.0 - vocal_gain) * i, -1.0, 1.0)
        out = Path(tempfile.gettempdir()) / f"mix_{id(mix)}.wav"
        save_audio(out, mix, sr)
        return str(out)
    except Exception:
        return None


def run_mix_balance(state, vocal_percent):
    """По ползунку громкости вокала (0–100) пересчитать микс и вернуть один аудиофайл."""
    if not state or len(state) != 2:
        return None
    v_path, i_path = state[0], state[1]
    gain = float(vocal_percent) / 100.0 if vocal_percent is not None else 0.5
    return _mix_two(v_path, i_path, gain)


def run_semantic_transform(audio_file, tempo_value, pitch_value):
    """Применить темп и тон через pyrubberband. Трансформер кэшируется — повторно без задержки."""
    path = _normalize_path(audio_file)
    if not path:
        return None
    transformer = _get_semantic()
    if transformer is None:
        return None
    try:
        out_path = Path(tempfile.gettempdir()) / f"sem_{Path(path).stem}_{id(path)}.wav"
        transformer.process_file(
            path,
            out_path=out_path,
            tempo_ratio=float(tempo_value) if tempo_value is not None else 1.0,
            pitch_semitones=float(pitch_value) if pitch_value is not None else 0.0,
        )
        return str(out_path) if out_path.exists() else None
    except Exception:
        return None


def run_separate_all_stems(audio_file, progress=gr.Progress()):
    """Разделение на стемы + lead/back vocal. Модель кэшируется. Пост-обработка включена."""
    path = _normalize_path(audio_file)
    if not path:
        return None, None, None, None, None, "Загрузите трек."
    sep = _get_demucs()
    if sep is None:
        return None, None, None, None, None, "Demucs не установлен."
    try:
        progress(0.1, desc="Разделение на стемы…")
        with _demucs_use_lock:
            result = sep.separate_file(path, split_vocals=True, clean=True)
        out_dir = Path(tempfile.mkdtemp(prefix="music_framework_stems_"))
        progress(0.85, desc="Сохранение…")
        result.save_all(out_dir, prefix=Path(path).stem)
        progress(1.0)
        prefix = Path(path).stem
        order = ["lead_vocal", "back_vocal", "drums", "bass", "other"]
        paths = []
        for name in order:
            p = out_dir / f"{prefix}_{name}.wav"
            paths.append(str(p) if p.exists() else None)
        msg = f"Готово. Стемы: {', '.join(k for k in order if (out_dir / f'{prefix}_{k}.wav').exists())}."
        return (*paths, msg)
    except Exception as e:
        return None, None, None, None, None, f"Ошибка: {e}"


def run_semantic(audio_file, tempo, pitch, progress=gr.Progress()):
    if audio_file is None:
        return None, "Загрузите аудиофайл и задайте параметры."
    path = _audio_path(audio_file)
    if not path or not Path(path).exists():
        return None, "Файл не найден."
    tr = _get_semantic()
    if tr is None:
        return None, "pyrubberband не установлен. pip install pyrubberband"
    try:
        tempo_val = float(str(tempo).replace(",", ".")) if tempo else 1.0
        pitch_val = float(str(pitch).replace(",", ".")) if pitch else 0.0
        progress(0.3, desc="Применение преобразований…")
        out_path = Path(tempfile.gettempdir()) / f"{Path(path).stem}_transformed.wav"
        tr.process_file(path, out_path=out_path, tempo_ratio=tempo_val, pitch_semitones=pitch_val)
        progress(1.0)
        return str(out_path), f"Готово. Темп: {tempo_val}x, Тон: {pitch_val} полутонов.\n\nФайл сохранён и доступен для прослушивания ниже."
    except Exception as e:
        return None, f"Ошибка: {e}"


def run_augment(audio_file, preset, progress=gr.Progress()):
    if audio_file is None:
        return None, "Загрузите аудиофайл и выберите пресет."
    path = _audio_path(audio_file)
    if not path or not Path(path).exists():
        return None, "Файл не найден."
    pipelines = get_preset_pipelines(44100) if get_preset_pipelines else {}
    if not pipelines:
        return None, "Audiomentations не установлен. pip install audiomentations"
    try:
        progress(0.3, desc="Применение аугментации…")
        pipe = pipelines.get(preset, pipelines.get("default", list(pipelines.values())[0]))
        out_path = Path(tempfile.gettempdir()) / f"{Path(path).stem}_aug.wav"
        pipe.transform_file(path, out_path=out_path, sample_rate=44100)
        progress(1.0)
        return str(out_path), f"Готово. Пресет: {preset}.\n\nФайл доступен для прослушивания ниже."
    except Exception as e:
        return None, f"Ошибка: {e}"


def run_metrics(ref_dir, est_dir, sources_text, progress=gr.Progress()):
    ref = (ref_dir or "").strip()
    est = (est_dir or "").strip()
    if not ref or not Path(ref).exists():
        return "Укажите папку с эталонными источниками (файлы: vocals.wav, drums.wav и т.д.)."
    if not est or not Path(est).exists():
        return "Укажите папку с оценками алгоритма."
    if not SeparationMetrics:
        return "torchmetrics не установлен. pip install torchmetrics"
    try:
        sources = [s.strip() for s in (sources_text or "vocals,drums,bass,other").split(",") if s.strip()]
        progress(0.5, desc="Вычисление метрик…")
        metrics = SeparationMetrics()
        report = metrics.evaluate_from_files(ref, est, sources, sample_rate=44100)
        progress(1.0)
        return report.summary()
    except Exception as e:
        return f"Ошибка: {e}"


# --- Интерфейс: меню слева, контент справа ---

custom_css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
#left_menu { min-width: 200px; padding: 1rem 0.5rem; border-right: 1px solid var(--border-color); }
#content_area { padding: 1rem 1.5rem; }
"""

with gr.Blocks(
    title="Music Framework",
    theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"),
    css=custom_css,
) as app:
    gr.Markdown("# Music Framework")

    with gr.Row():
        with gr.Column(scale=1, elem_id="left_menu"):
            menu_radio = gr.Radio(
                choices=["Вокал и инструментал", "Стемы", "Темп и тон", "Анализ", "Аугментация", "Метрики"],
                value="Вокал и инструментал",
                label="Раздел",
                show_label=False,
            )

        with gr.Column(scale=4, elem_id="content_area"):
            with gr.Group(visible=True) as gr_vocal:
                sep_state = gr.State(value=(None, None))
                sep_file = gr.File(label="Загрузите трек", file_types=["audio"], type="filepath")
                sep_status = gr.Textbox(label="Статус", lines=1, interactive=False)
                sep_mix_audio = gr.Audio(label="Воспроизведение", type="filepath", value=None, interactive=False)
                sep_balance = gr.Slider(0, 100, 50, step=1, label="Баланс вокал / инструментал")
                sep_file.change(
                    run_separate_vocal_inst,
                    inputs=[sep_file],
                    outputs=[sep_state, sep_mix_audio, sep_status],
                )
                sep_balance.change(
                    run_mix_balance,
                    inputs=[sep_state, sep_balance],
                    outputs=[sep_mix_audio],
                )

            with gr.Group(visible=False) as gr_stems:
                stems_file = gr.File(label="Загрузите трек", file_types=["audio"], type="filepath")
                stems_status = gr.Textbox(label="Статус", lines=1, interactive=False)
                stems_lead = gr.Audio(label="Lead vocal", type="filepath")
                stems_back = gr.Audio(label="Back vocal", type="filepath")
                stems_drums = gr.Audio(label="Барабаны", type="filepath")
                stems_bass = gr.Audio(label="Бас", type="filepath")
                stems_other = gr.Audio(label="Другое", type="filepath")
                stems_file.change(
                    run_separate_all_stems,
                    inputs=[stems_file],
                    outputs=[stems_lead, stems_back, stems_drums, stems_bass, stems_other, stems_status],
                )

            with gr.Group(visible=False) as gr_analyze:
                analyze_input = gr.Audio(label="Аудиофайл", type="filepath", sources=["upload"])
                analyze_btn = gr.Button("Анализ")
                analyze_output = gr.Textbox(label="Результат", lines=10)
                analyze_btn.click(run_analyze, inputs=[analyze_input], outputs=[analyze_output])

            with gr.Group(visible=False) as gr_tempo:
                sem_file = gr.File(label="Загрузите аудио", file_types=["audio"], type="filepath")
                sem_tempo = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Темп")
                sem_pitch = gr.Slider(-12, 12, 0.0, step=0.5, label="Тон, полутоны")
                sem_output = gr.Audio(label="Воспроизведение", type="filepath", value=None, interactive=False)

                def _apply_tempo_pitch(audio_file, tempo_val, pitch_val):
                    path = _normalize_path(audio_file)
                    if not path:
                        return None
                    t = float(tempo_val) if tempo_val is not None else 1.0
                    p = float(pitch_val) if pitch_val is not None else 0.0
                    if abs(t - 1.0) < 1e-6 and abs(p) < 1e-6:
                        return path
                    return run_semantic_transform(audio_file, t, p)

                sem_file.change(
                    _apply_tempo_pitch,
                    inputs=[sem_file, sem_tempo, sem_pitch],
                    outputs=[sem_output],
                )
                sem_tempo.change(
                    _apply_tempo_pitch,
                    inputs=[sem_file, sem_tempo, sem_pitch],
                    outputs=[sem_output],
                )
                sem_pitch.change(
                    _apply_tempo_pitch,
                    inputs=[sem_file, sem_tempo, sem_pitch],
                    outputs=[sem_output],
                )

            with gr.Group(visible=False) as gr_aug:
                aug_input = gr.Audio(label="Аудиофайл", type="filepath", sources=["upload"])
                aug_preset = gr.Radio(label="Пресет", choices=["light", "default", "aggressive"], value="default")
                aug_btn = gr.Button("Применить")
                aug_audio = gr.Audio(label="Результат", type="filepath")
                aug_output = gr.Textbox(label="Статус", lines=2)
                aug_btn.click(run_augment, inputs=[aug_input, aug_preset], outputs=[aug_audio, aug_output])

            with gr.Group(visible=False) as gr_metrics:
                metrics_ref = gr.Textbox(label="Папка эталонов")
                metrics_est = gr.Textbox(label="Папка оценок")
                metrics_sources = gr.Textbox(label="Источники", value="vocals,drums,bass,other")
                metrics_btn = gr.Button("Вычислить")
                metrics_output = gr.Textbox(label="Результат", lines=12)
                metrics_btn.click(run_metrics, inputs=[metrics_ref, metrics_est, metrics_sources], outputs=[metrics_output])

    def _toggle_section(choice):
        return (
            gr.update(visible=choice == "Вокал и инструментал"),
            gr.update(visible=choice == "Стемы"),
            gr.update(visible=choice == "Темп и тон"),
            gr.update(visible=choice == "Анализ"),
            gr.update(visible=choice == "Аугментация"),
            gr.update(visible=choice == "Метрики"),
        )

    menu_radio.change(
        _toggle_section,
        inputs=[menu_radio],
        outputs=[gr_vocal, gr_stems, gr_tempo, gr_analyze, gr_aug, gr_metrics],
    )


def main():
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

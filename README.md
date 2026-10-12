## Music Framework

Музыкальный редактор с разделением на стемы (Demucs), выделением вокала, изменением темпа и тона и веб‑интерфейсом на Gradio.

## Архитектура

| Модуль | Назначение | Библиотеки |
|--------|------------|------------|
| **Ядро I/O и анализ** | Загрузка/сохранение, спектрограммы, темп, тональность | SoundFile, Librosa |
| **Нейросети и аугментация** | Тензорный аудио-бэкенд, спектрограммы, шум/сдвиг | TorchAudio |
| **Разделение треков** | Выделение вокала и инструментов | Demucs |
| **Семантика** | Изменение темпа и тона | Pyrubberband |
| **Аугментация датасетов** | Расширение данных для дообучения | Audiomentations |
| **Оценка разделения** | Метрики SDR, SAR, SIR | Torchmetrics, mir_eval |

## Быстрый старт (Windows, CMD, Python 3.11)

1. Откройте **Командную строку (CMD)** и перейдите в папку проекта:

```cmd
cd "C:\Users\artem\OneDrive\Рабочий стол\studyings\политех\Music editor"
```

2. Создайте виртуальное окружение на Python 3.11:

```cmd
py -3.11 -m venv .venv
```

3. Активируйте окружение:

```cmd
.\.venv\Scripts\activate.bat
```

4. Установите зависимости:

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "demucs==4.0.1" "omegaconf==2.1.1" "hydra-core==1.1.0" "dora-search"
```

5. Запустите веб‑приложение:

```cmd
python scripts/run_web.py
```

6. Откройте в браузере:

`http://127.0.0.1:7860`

Остановить сервер: `Ctrl+C` в терминале.

### Примечание: Pyrubberband (темп и тон / pitch)

Для вкладки «Темп и тон» нужны **pyrubberband** (Python) и **Rubber Band** (исполняемый файл):

1. **Python-пакет:**  
   `pip install pyrubberband`

2. **Установка Rubber Band на Windows** (без него pitch/темп не работают):
   - Скачайте архив с [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/) (раздел Downloads → Windows), распакуйте. В папке должны быть `rubberband.exe` и при необходимости DLL.
   - **Вариант А:** добавьте папку с `rubberband.exe` в переменную окружения **PATH**. После этого перезапустите терминал (или Cursor/IDE), чтобы Python увидел новый PATH. Проверка: в **новом** окне CMD выполните `rubberband` — должна появиться справка.
   - **Вариант Б (если PATH не срабатывает):** создайте переменную окружения **RUBBERBAND_PATH** и укажите в ней **полный путь к файлу** `rubberband.exe` (например `C:\rubberband\rubberband.exe`). Приложение само добавит эту папку в PATH и передаст путь в pyrubberband.

3. **Другие ОС:**
   - **Conda:** `conda install -c conda-forge rubberband`
   - **Linux:** `sudo apt install rubberband-cli` (Ubuntu/Debian)
   - **macOS:** `brew install rubberband`

---

## Использование вкладок

- Вокал и инструментал: загрузка трека, разделение Demucs, один плеер с миксом и ползунок баланса вокала.
- Стемы: загрузка трека, разделение на `lead vocal`, `back vocal`, `drums`, `bass`, `other`.
- Темп и тон: загрузка аудио, слайдеры темпа и тона, результат в одном плеере.
- Анализ: базовый анализ аудио (длительность, темп и т.д.).
- Аугментация: применение пресетов аугментаций к аудио.
- Метрики: расчёт SDR/SAR по папкам эталонов и предсказаний.

## Структура проекта

```
Music editor/
├── src/music_framework/     # Ядро фреймворка
│   ├── core/                # I/O (SoundFile), анализ (Librosa)
│   ├── torch_audio_module/  # TorchAudio бэкенд и аугментации
│   ├── separation/          # Demucs
│   ├── semantic/            # Pyrubberband
│   ├── augmentation/        # Audiomentations
│   ├── metrics/             # SDR/SAR (Torchmetrics)
│   ├── pipeline/            # Сценарии (разделение+оценка, расширение датасета)
│   └── cli.py               # CLI
├── app/                     # Веб-интерфейс
│   ├── web_app.py           # Gradio (браузер)
│   └── main.py              # CustomTkinter (опционально)
├── scripts/                 # run_gui.py, run_cli.py
├── examples/                # Примеры использования
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Примеры кода

### Загрузка и анализ

```python
from music_framework.core.io import load_audio, save_audio
from music_framework.core.analysis import AudioAnalyzer

y, sr = load_audio("track.wav", sr=22050, mono=True)
analyzer = AudioAnalyzer(sr=sr)
result = analyzer.load_and_analyze("track.wav")
print(result.tempo_bpm, result.duration_sec)
```

### Разделение источников (Demucs)

```python
from music_framework.separation import DemucsSeparator

sep = DemucsSeparator(model_name="htdemucs")
result = sep.separate_file("mix.wav")
result.save_all("./output", prefix="mix")
vocals = result.get("vocals")
```

### Темп и тон (Pyrubberband)

```python
from music_framework.semantic import SemanticTransformer
from music_framework.core.io import load_audio, save_audio

tr = SemanticTransformer()
audio, sr = load_audio("track.wav", mono=True)
fast = tr.time_stretch(audio, sr, rate=1.2)
higher = tr.pitch_shift(audio, sr, n_semitones=2)
save_audio("out.wav", higher, sr)
```

### Аугментация датасета

```python
from music_framework.augmentation import get_preset_pipelines

pipelines = get_preset_pipelines(44100)
pipe = pipelines["default"]
augmented = pipe.transform_file("track.wav", out_path="track_aug.wav")
# или расширение папки:
pipe.expand_dataset("./wavs", "./augmented", num_augmented_per_file=3)
```

### Метрики разделения

```python
from music_framework.metrics import SeparationMetrics

metrics = SeparationMetrics()
report = metrics.evaluate_from_files(
    reference_dir="./ground_truth",
    estimate_dir="./separated",
    source_names=["vocals", "drums", "bass", "other"],
)
print(report.mean_sdr(), report.summary())
```


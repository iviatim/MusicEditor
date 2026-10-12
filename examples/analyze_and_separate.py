"""
Пример: загрузка трека, анализ, разделение на источники (Demucs).
Запуск из корня проекта: python examples/analyze_and_separate.py <путь_к_файлу>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from music_framework.core.io import load_audio, get_audio_info
from music_framework.core.analysis import AudioAnalyzer
from music_framework.separation import DemucsSeparator


def main():
    if len(sys.argv) < 2:
        print("Использование: python analyze_and_separate.py <аудиофайл>")
        return
    path = Path(sys.argv[1])
    if not path.exists():
        print("Файл не найден:", path)
        return

    print("=== Информация о файле ===")
    info = get_audio_info(path)
    print(info)

    print("\n=== Анализ (Librosa) ===")
    analyzer = AudioAnalyzer(sr=22050)
    result = analyzer.load_and_analyze(path)
    print(result.summary())

    print("\n=== Разделение (Demucs) ===")
    separator = DemucsSeparator(model_name="htdemucs")
    separated = separator.separate_file(path)
    out_dir = path.parent / "separated"
    out_dir.mkdir(exist_ok=True)
    saved = separated.save_all(out_dir, prefix=path.stem)
    print("Источники:", separated.keys())
    for p in saved:
        print("  ", p)


if __name__ == "__main__":
    main()

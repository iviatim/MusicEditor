"""
Пример: изменение темпа/тона (Pyrubberband) и аугментация (Audiomentations).
Запуск: python examples/semantic_and_augment.py <путь_к_файлу>
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from music_framework.core.io import save_audio
from music_framework.semantic import SemanticTransformer
from music_framework.augmentation import get_preset_pipelines


def main():
    if len(sys.argv) < 2:
        print("Использование: python semantic_and_augment.py <аудиофайл>")
        return
    path = Path(sys.argv[1])
    if not path.exists():
        print("Файл не найден:", path)
        return

    out_dir = path.parent / "processed"
    out_dir.mkdir(exist_ok=True)

    # Темп 1.2x, тон +2 полутона
    print("Семантика: темп 1.2, тон +2 полутона")
    tr = SemanticTransformer()
    audio, sr = tr.process_file(
        path,
        out_path=out_dir / (path.stem + "_fast_pitch.wav"),
        tempo_ratio=1.2,
        pitch_semitones=2.0,
    )
    print("  Сохранено:", out_dir / (path.stem + "_fast_pitch.wav"))

    # Аугментация
    print("Аугментация: пресет default")
    pipelines = get_preset_pipelines(44100)
    pipe = pipelines["default"]
    pipe.transform_file(
        path,
        out_path=out_dir / (path.stem + "_aug.wav"),
        sample_rate=44100,
    )
    print("  Сохранено:", out_dir / (path.stem + "_aug.wav"))


if __name__ == "__main__":
    main()

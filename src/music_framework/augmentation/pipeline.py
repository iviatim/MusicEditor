"""
Пайплайны аугментаций на базе Audiomentations.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

from music_framework.core.io import load_audio, save_audio

try:
    import audiomentations as am
    AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False
    am = None


class AugmentationPipeline:
    """
    Пайплайн аугментаций Audiomentations для расширения датасетов
    и дообучения существующих моделей.
    """

    def __init__(self, sample_rate: int = 44100):
        if not AUDIOMENTATIONS_AVAILABLE:
            raise RuntimeError(
                "audiomentations не установлен. Установите: pip install audiomentations"
            )
        self.sample_rate = sample_rate
        self._transform: Optional[Any] = None

    def add_compose(self, transform: Any) -> "AugmentationPipeline":
        """Установить композитный трансформ (Compose из audiomentations)."""
        self._transform = transform
        return self

    def add_default_music(self) -> "AugmentationPipeline":
        """Добавить набор типичных аугментаций для музыки."""
        self._transform = am.Compose([
            am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            am.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            am.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            am.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
            am.HighPassFilter(min_cutoff_freq=40, max_cutoff_freq=400, p=0.3),
            am.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=16000, p=0.3),
        ])
        return self

    def add_light(self) -> "AugmentationPipeline":
        """Лёгкие аугментации (шум + громкость)."""
        self._transform = am.Compose([
            am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            am.Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
        ])
        return self

    def add_aggressive(self) -> "AugmentationPipeline":
        """Более агрессивные (включая сдвиг по времени и частотные фильтры)."""
        self._transform = am.Compose([
            am.AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.03, p=0.6),
            am.TimeStretch(min_rate=0.7, max_rate=1.4, p=0.5),
            am.PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
            am.Gain(min_gain_in_db=-18, max_gain_in_db=18, p=0.5),
            am.HighPassFilter(min_cutoff_freq=20, max_cutoff_freq=800, p=0.4),
            am.LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=18000, p=0.4),
            am.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=20, p=0.2),
        ])
        return self

    def transform(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """Применить пайплайн к массиву (samples,) или (samples, channels)."""
        if self._transform is None:
            raise RuntimeError("Сначала настройте пайплайн (add_default_music, add_light и т.д.)")
        sr = sample_rate or self.sample_rate
        return self._transform(samples=audio, sample_rate=sr)

    def transform_file(
        self,
        path: Union[str, Path],
        out_path: Optional[Union[str, Path]] = None,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Загрузить файл, применить аугментации, опционально сохранить."""
        path = Path(path)
        sr = sample_rate or self.sample_rate
        audio, _ = load_audio(path, sr=sr, mono=True)
        augmented = self.transform(audio, sr)
        if out_path:
            save_audio(out_path, augmented, sr)
        return augmented

    def expand_dataset(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        num_augmented_per_file: int = 3,
        sample_rate: Optional[int] = None,
        extensions: tuple = (".wav", ".flac", ".mp3", ".ogg"),
    ) -> List[Path]:
        """
        Для каждого файла в input_dir создать num_augmented_per_file
        аугментированных версий в output_dir.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sr = sample_rate or self.sample_rate
        saved = []
        for f in input_dir.iterdir():
            if f.suffix.lower() not in extensions:
                continue
            base = f.stem
            for i in range(num_augmented_per_file):
                audio, _ = load_audio(f, sr=sr, mono=True)
                aug = self.transform(audio, sr)
                out_path = output_dir / f"{base}_aug{i}{f.suffix}"
                save_audio(out_path, aug, sr)
                saved.append(out_path)
        return saved


def get_preset_pipelines(sample_rate: int = 44100) -> dict:
    """Возвращает словарь готовых пайплайнов: light, default, aggressive."""
    return {
        "light": AugmentationPipeline(sample_rate).add_light(),
        "default": AugmentationPipeline(sample_rate).add_default_music(),
        "aggressive": AugmentationPipeline(sample_rate).add_aggressive(),
    }

"""
Изменение темпа и высоты тона через Pyrubberband (обёртка над Rubber Band).
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from music_framework.core.io import load_audio, save_audio

try:
    import pyrubberband as pyrb
    PYRUBBERBAND_AVAILABLE = True
    # На Windows: если задан RUBBERBAND_PATH (полный путь к rubberband.exe), используем его
    if sys.platform == "win32":
        _rb_path = os.environ.get("RUBBERBAND_PATH", "").strip()
        if _rb_path and os.path.isfile(_rb_path):
            import pyrubberband.pyrb as _pyrb
            _pyrb.__RUBBERBAND_UTIL = _rb_path
except ImportError:
    PYRUBBERBAND_AVAILABLE = False
    pyrb = None


class SemanticTransformer:
    """
    Семантические преобразования: темп (time stretch) и тон (pitch shift).
    Использует pyrubberband для высококачественного изменения без потери качества.
    """

    def __init__(self):
        if not PYRUBBERBAND_AVAILABLE:
            raise RuntimeError(
                "pyrubberband не установлен. Установите: pip install pyrubberband. "
                "Также нужна установка rubberband (например rubberband-cli)."
            )

    def time_stretch(
        self,
        audio: np.ndarray,
        sr: int,
        rate: float,
    ) -> np.ndarray:
        """
        Изменить темп без изменения высоты тона.
        rate > 1 — быстрее, rate < 1 — медленнее.
        """
        return pyrb.time_stretch(audio, sr, rate)

    def pitch_shift(
        self,
        audio: np.ndarray,
        sr: int,
        n_semitones: float,
    ) -> np.ndarray:
        """
        Изменить высоту тона без изменения темпа.
        n_semitones: полутоны (положительное — выше).
        """
        return pyrb.pitch_shift(audio, sr, n_semitones)

    def change_tempo_and_pitch(
        self,
        audio: np.ndarray,
        sr: int,
        tempo_ratio: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> np.ndarray:
        """
        Сначала темп, затем тон (порядок можно менять в коде при необходимости).
        """
        if tempo_ratio != 1.0:
            audio = self.time_stretch(audio, sr, tempo_ratio)
        if pitch_semitones != 0.0:
            audio = self.pitch_shift(audio, sr, pitch_semitones)
        return audio

    def process_file(
        self,
        path: Union[str, Path],
        out_path: Optional[Union[str, Path]] = None,
        tempo_ratio: float = 1.0,
        pitch_semitones: float = 0.0,
        sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Загрузить файл, применить преобразования, опционально сохранить.
        Returns:
            (audio, sample_rate)
        """
        path = Path(path)
        audio, file_sr = load_audio(path, sr=sr, mono=True)
        audio = self.change_tempo_and_pitch(
            audio, file_sr, tempo_ratio=tempo_ratio, pitch_semitones=pitch_semitones
        )
        if out_path:
            save_audio(out_path, audio, file_sr)
        return audio, file_sr

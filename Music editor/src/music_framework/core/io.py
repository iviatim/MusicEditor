"""
Модуль ввода/вывода аудио на базе SoundFile и Librosa.
Единый интерфейс для загрузки/сохранения с нормализацией форматов.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa


def load_audio(
    path: Union[str, Path],
    sr: Optional[int] = None,
    mono: bool = True,
    dtype: str = "float32",
    res_type: str = "soxr_hq",
) -> Tuple[np.ndarray, int]:
    """
    Загрузка аудиофайла через SoundFile + Librosa для ресемплинга при необходимости.

    Args:
        path: Путь к файлу (wav, flac, ogg и др.).
        sr: Желаемая частота дискретизации (None — нативная).
        mono: Привести к моно путём усреднения каналов.
        dtype: Тип данных сэмплов ('float32', 'float64').
        res_type: Метод ресемплинга Librosa.

    Returns:
        (samples, sample_rate)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    data, file_sr = sf.read(str(path), dtype=dtype, always_2d=True)

    if mono and data.shape[1] > 1:
        data = np.mean(data, axis=1)
    elif not mono and data.ndim == 1:
        data = np.stack([data, data], axis=1)

    if sr is not None and file_sr != sr:
        data = librosa.resample(
            data.T if data.ndim == 2 else data,
            orig_sr=file_sr,
            target_sr=sr,
            res_type=res_type,
        )
        if data.ndim == 2:
            data = data.T
        file_sr = sr

    if mono and data.ndim == 2:
        data = data.squeeze()

    return np.ascontiguousarray(data), int(file_sr)


def save_audio(
    path: Union[str, Path],
    data: np.ndarray,
    sr: int,
    subtype: Optional[str] = None,
) -> Path:
    """
    Сохранение аудио через SoundFile.

    Args:
        path: Путь для сохранения.
        data: Массив сэмплов (float32, shape (samples,) или (samples, channels)).
        sr: Частота дискретизации.
        subtype: Подтип формата (например 'PCM_24' для wav).

    Returns:
        Path сохранённого файла.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    sf.write(str(path), data, sr, subtype=subtype)
    return path


def get_audio_info(path: Union[str, Path]) -> "AudioInfo":
    """Чтение метаданных без загрузки всего файла."""
    path = Path(path)
    info = sf.info(str(path))
    return AudioInfo(
        path=path,
        samplerate=info.samplerate,
        frames=info.frames,
        channels=info.channels,
        duration_sec=info.duration,
        subtype=info.subtype,
        format=info.format,
    )


class AudioInfo:
    """Метаданные аудиофайла."""

    __slots__ = ("path", "samplerate", "frames", "channels", "duration_sec", "subtype", "format")

    def __init__(
        self,
        path: Path,
        samplerate: int,
        frames: int,
        channels: int,
        duration_sec: float,
        subtype: str,
        format: str,
    ):
        self.path = path
        self.samplerate = samplerate
        self.frames = frames
        self.channels = channels
        self.duration_sec = duration_sec
        self.subtype = subtype
        self.format = format

    def __repr__(self) -> str:
        return (
            f"AudioInfo(path={self.path}, sr={self.samplerate}, "
            f"duration={self.duration_sec:.2f}s, channels={self.channels})"
        )

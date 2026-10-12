"""
Music Framework — модульный фреймворк для цифровой обработки музыки с нейросетями.
Работает локально, без выгрузки на внешние хосты.
"""

__version__ = "0.1.0"

from music_framework.core.io import load_audio, save_audio, get_audio_info, AudioInfo
from music_framework.core.analysis import AudioAnalyzer

__all__ = [
    "load_audio",
    "save_audio",
    "get_audio_info",
    "AudioInfo",
    "AudioAnalyzer",
    "__version__",
]

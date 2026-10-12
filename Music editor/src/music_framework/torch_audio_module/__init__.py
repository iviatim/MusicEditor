"""
Интеграция с нейросетями и аугментация на базе TorchAudio.
"""

from music_framework.torch_audio_module.backend import TorchAudioBackend
from music_framework.torch_audio_module.augment import TorchAudioAugmentor

__all__ = ["TorchAudioBackend", "TorchAudioAugmentor"]

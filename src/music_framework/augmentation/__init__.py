"""
Модуль аугментации: Audiomentations для расширения датасетов и дообучения моделей.
"""

from music_framework.augmentation.pipeline import AugmentationPipeline, get_preset_pipelines

__all__ = ["AugmentationPipeline", "get_preset_pipelines"]

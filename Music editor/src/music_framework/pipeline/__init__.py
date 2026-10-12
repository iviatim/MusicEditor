"""
Пайплайн: объединение модулей в единые сценарии (разделение + оценка, аугментация датасета и т.д.).
"""

from music_framework.pipeline.workflows import separate_and_evaluate, augment_dataset

__all__ = ["separate_and_evaluate", "augment_dataset"]

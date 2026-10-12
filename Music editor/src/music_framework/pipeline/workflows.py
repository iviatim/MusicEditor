"""
Готовые сценарии: разделение с оценкой, расширение датасета аугментациями.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from music_framework.separation import DemucsSeparator, SeparationResult
from music_framework.metrics import SeparationMetrics, SeparationReport
from music_framework.augmentation import AugmentationPipeline, get_preset_pipelines


def separate_and_evaluate(
    mix_path: Path,
    reference_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_name: str = "htdemucs",
    sample_rate: int = 44100,
) -> Tuple[SeparationResult, Optional[SeparationReport]]:
    """
    Разделить смесь Demucs и при наличии эталонов вычислить SDR/SAR.
    reference_dir: папка с эталонными wav (vocals.wav, drums.wav, …).
    """
    mix_path = Path(mix_path)
    output_dir = output_dir or mix_path.parent / "separated"
    output_dir.mkdir(parents=True, exist_ok=True)

    sep = DemucsSeparator(model_name=model_name)
    result = sep.separate_file(mix_path, sample_rate=sample_rate)
    result.save_all(output_dir, prefix=mix_path.stem)

    report = None
    if reference_dir and reference_dir.exists():
        sources = result.keys()
        metrics = SeparationMetrics()
        report = metrics.evaluate_from_files(
            reference_dir, output_dir, list(sources), sample_rate=sample_rate
        )
    return result, report


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    preset: str = "default",
    num_per_file: int = 3,
    sample_rate: int = 44100,
    extensions: tuple = (".wav", ".flac", ".mp3", ".ogg"),
) -> List[Path]:
    """Расширить датасет: для каждого файла в input_dir создать num_per_file аугментированных копий."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pipelines = get_preset_pipelines(sample_rate)
    pipe = pipelines.get(preset, pipelines["default"])
    return pipe.expand_dataset(
        input_dir, output_dir,
        num_augmented_per_file=num_per_file,
        sample_rate=sample_rate,
        extensions=extensions,
    )

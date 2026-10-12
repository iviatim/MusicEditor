"""
CLI для фреймворка: анализ, разделение, семантика, аугментация, метрики.
"""

import argparse
import sys
from pathlib import Path

# Добавляем src в path при запуске как скрипт
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC.parent.parent))


def cmd_analyze(args):
    from music_framework.core.io import load_audio
    from music_framework.core.analysis import AudioAnalyzer
    p = Path(args.input)
    if not p.exists():
        print(f"Файл не найден: {p}", file=sys.stderr)
        return 1
    sr = args.sample_rate or 22050
    analyzer = AudioAnalyzer(sr=sr)
    result = analyzer.load_and_analyze(p)
    print(result.summary())
    print(f"Длительность: {result.duration_sec:.2f} с, Темп: {result.tempo_bpm:.1f} BPM")
    return 0


def cmd_separate(args):
    from music_framework.separation import DemucsSeparator
    inp = Path(args.input)
    out = Path(args.output or str(inp.parent / "separated"))
    out.mkdir(parents=True, exist_ok=True)
    sep = DemucsSeparator(model_name=args.model or "htdemucs")
    result = sep.separate_file(inp, sample_rate=args.sample_rate)
    paths = result.save_all(out, prefix=inp.stem)
    print("Источники:", ", ".join(result.keys()))
    for path in paths:
        print("  ", path)
    return 0


def cmd_semantic(args):
    from music_framework.core.io import save_audio
    from music_framework.semantic import SemanticTransformer
    inp = Path(args.input)
    out = Path(args.output or str(inp.parent / (inp.stem + "_transformed.wav")))
    tr = SemanticTransformer()
    audio, sr = tr.process_file(inp, out_path=out, tempo_ratio=args.tempo, pitch_semitones=args.pitch)
    print("Сохранено:", out)
    return 0


def cmd_augment(args):
    from music_framework.augmentation import get_preset_pipelines
    inp = Path(args.input)
    out = Path(args.output or str(inp.parent / (inp.stem + "_aug.wav")))
    pipelines = get_preset_pipelines(args.sample_rate or 44100)
    pipe = pipelines.get(args.preset, pipelines["default"])
    pipe.transform_file(inp, out_path=out, sample_rate=args.sample_rate)
    print("Сохранено:", out)
    return 0


def cmd_metrics(args):
    from music_framework.metrics import SeparationMetrics
    ref = Path(args.reference_dir)
    est = Path(args.estimate_dir)
    sources = [s.strip() for s in (args.sources or "vocals,drums,bass,other").split(",") if s.strip()]
    if not ref.exists() or not est.exists():
        print("Укажите существующие папки эталонов и оценок.", file=sys.stderr)
        return 1
    metrics = SeparationMetrics()
    report = metrics.evaluate_from_files(ref, est, sources, sample_rate=args.sample_rate or 44100)
    print(report.summary())
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Music Framework — цифровая обработка музыки с нейросетями (локально)"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", help="Команды")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Анализ аудио (Librosa)")
    p_analyze.add_argument("input", help="Входной аудиофайл")
    p_analyze.add_argument("--sample-rate", "-sr", type=int, default=None)
    p_analyze.set_defaults(run=cmd_analyze)

    # separate
    p_sep = sub.add_parser("separate", help="Разделение источников (Demucs)")
    p_sep.add_argument("input", help="Входной файл смеси")
    p_sep.add_argument("--output", "-o", help="Папка вывода")
    p_sep.add_argument("--model", "-m", default="htdemucs")
    p_sep.add_argument("--sample-rate", "-sr", type=int, default=None)
    p_sep.set_defaults(run=cmd_separate)

    # semantic
    p_sem = sub.add_parser("semantic", help="Темп и тон (Pyrubberband)")
    p_sem.add_argument("input", help="Входной файл")
    p_sem.add_argument("--output", "-o", help="Выходной файл")
    p_sem.add_argument("--tempo", "-t", type=float, default=1.0)
    p_sem.add_argument("--pitch", "-p", type=float, default=0.0)
    p_sem.set_defaults(run=cmd_semantic)

    # augment
    p_aug = sub.add_parser("augment", help="Аугментация (Audiomentations)")
    p_aug.add_argument("input", help="Входной файл")
    p_aug.add_argument("--output", "-o", help="Выходной файл")
    p_aug.add_argument("--preset", choices=["light", "default", "aggressive"], default="default")
    p_aug.add_argument("--sample-rate", "-sr", type=int, default=44100)
    p_aug.set_defaults(run=cmd_augment)

    # metrics
    p_met = sub.add_parser("metrics", help="Метрики SDR/SAR (Torchmetrics)")
    p_met.add_argument("reference_dir", help="Папка с эталонными источниками")
    p_met.add_argument("estimate_dir", help="Папка с оценками алгоритма")
    p_met.add_argument("--sources", "-s", default="vocals,drums,bass,other")
    p_met.add_argument("--sample-rate", "-sr", type=int, default=44100)
    p_met.set_defaults(run=cmd_metrics)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.run(args) or 0


if __name__ == "__main__":
    sys.exit(main())

"""
Объективное сравнение результатов разделения: SDR, SAR, SIR (Torchmetrics / mir_eval).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from music_framework.core.io import load_audio

try:
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    from torchmetrics.audio import SignalDistortionRatio
    from torchmetrics.audio import SignalNoiseRatio
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    ScaleInvariantSignalDistortionRatio = None
    SignalDistortionRatio = None
    SignalNoiseRatio = None

try:
    import mir_eval.separation
    MIR_EVAL_AVAILABLE = True
except ImportError:
    MIR_EVAL_AVAILABLE = False
    mir_eval = None


class SeparationReport:
    """Отчёт по метрикам разделения (SDR, SAR, SIR и др.)."""

    def __init__(
        self,
        sdr: Dict[str, float],
        sar: Optional[Dict[str, float]] = None,
        sir: Optional[Dict[str, float]] = None,
        per_source_sdr: Optional[Dict[str, float]] = None,
    ):
        self.sdr = sdr
        self.sar = sar or {}
        self.sir = sir or {}
        self.per_source_sdr = per_source_sdr or {}

    def mean_sdr(self) -> float:
        if not self.sdr:
            return 0.0
        return sum(self.sdr.values()) / len(self.sdr)

    def summary(self) -> str:
        lines = ["Separation metrics:"]
        if self.sdr:
            lines.append(f"  SDR (mean): {self.mean_sdr():.3f} dB")
            for k, v in self.sdr.items():
                lines.append(f"    {k}: {v:.3f} dB")
        if self.sar:
            lines.append("  SAR: " + ", ".join(f"{k}={v:.3f}" for k, v in self.sar.items()))
        if self.sir:
            lines.append("  SIR: " + ", ".join(f"{k}={v:.3f}" for k, v in self.sir.items()))
        return "\n".join(lines)


class SeparationMetrics:
    """
    Вычисление метрик качества разделения источников (SDR, SAR, SIR).
    Использует Torchmetrics и при наличии mir_eval для детальных отчётов.
    """

    def __init__(self, device: Optional[str] = None):
        if not TORCHMETRICS_AVAILABLE:
            raise RuntimeError(
                "torchmetrics не установлен. Установите: pip install torchmetrics"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sdr_metric = ScaleInvariantSignalDistortionRatio().to(self.device)
        self._sdr_non_inv = SignalDistortionRatio().to(self.device)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def compute_sdr(
        self,
        preds: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        scale_invariant: bool = True,
    ) -> float:
        """
        Signal-to-Distortion Ratio (или Scale-Invariant SDR).
        preds/target: (sources, samples) или (samples,) для одного источника.
        """
        if isinstance(preds, np.ndarray):
            preds = self._to_tensor(preds)
        if isinstance(target, np.ndarray):
            target = self._to_tensor(target)
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
        # Torchmetrics ожидает (batch, sources, time)
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0)
        m = self._sdr_metric if scale_invariant else self._sdr_non_inv
        return float(m(preds, target).cpu().item())

    def compute_sar_sir_mir_eval(
        self,
        references: Dict[str, np.ndarray],
        estimates: Dict[str, np.ndarray],
        sample_rate: int,
        source_names: Optional[List[str]] = None,
    ) -> tuple:
        """
        SAR/SIR через mir_eval.separation (если установлен).
        references/estimates: имя источника -> (samples,) или (samples, ch).
        source_names: порядок источников (по умолчанию sorted(references.keys())).
        Returns:
            (sdr, sir, sar) — каждый массив по источникам в порядке source_names.
        """
        if not MIR_EVAL_AVAILABLE:
            return (), (), ()
        ref_list = []
        est_list = []
        names = source_names if source_names is not None else sorted(references.keys())
        for name in names:
            r = references[name]
            e = estimates.get(name)
            if e is None:
                continue
            if r.ndim == 2:
                r = r.mean(axis=1)
            if e.ndim == 2:
                e = e.mean(axis=1)
            min_len = min(len(r), len(e))
            ref_list.append(r[:min_len])
            est_list.append(e[:min_len])
        if not ref_list:
            return (), (), ()
        ref_arr = np.array(ref_list)
        est_arr = np.array(est_list)
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref_arr, est_arr)
        return sdr, sir, sar

    def evaluate_separation(
        self,
        references: Dict[str, np.ndarray],
        estimates: Dict[str, np.ndarray],
        source_names: Optional[List[str]] = None,
        sample_rate: int = 44100,
    ) -> SeparationReport:
        """
        Полная оценка: SDR по каждому источнику и средний.
        references: эталонные сигналы (название -> массив).
        estimates: предсказанные сигналы (название -> массив).
        """
        names = source_names or sorted(set(references.keys()) & set(estimates.keys()))
        per_source = {}
        for name in names:
            ref = references[name]
            est = estimates.get(name)
            if est is None:
                continue
            ref = np.asarray(ref, dtype=np.float32)
            est = np.asarray(est, dtype=np.float32)
            if ref.ndim == 2:
                ref = ref.mean(axis=1)
            if est.ndim == 2:
                est = est.mean(axis=1)
            min_len = min(len(ref), len(est))
            ref = ref[:min_len]
            est = est[:min_len]
            sdr_val = self.compute_sdr(est, ref, scale_invariant=True)
            per_source[name] = sdr_val

        mean_sdr = sum(per_source.values()) / len(per_source) if per_source else 0.0
        sar_dict = {}
        sir_dict = {}
        if MIR_EVAL_AVAILABLE and references and estimates:
            sdr_arr, sir_arr, sar_arr = self.compute_sar_sir_mir_eval(
                references, estimates, sample_rate=sample_rate, source_names=names
            )
            if len(sdr_arr) == len(names):
                for i, name in enumerate(names):
                    sir_dict[name] = float(sir_arr[i])
                    sar_dict[name] = float(sar_arr[i])

        return SeparationReport(
            sdr={"mean": mean_sdr, **per_source},
            sar=sar_dict,
            sir=sir_dict,
            per_source_sdr=per_source,
        )

    def evaluate_from_files(
        self,
        reference_dir: Union[str, Path],
        estimate_dir: Union[str, Path],
        source_names: List[str],
        sample_rate: int = 44100,
    ) -> SeparationReport:
        """
        Загрузить эталоны и оценки из папок (файлы: {name}.wav) и вычислить метрики.
        """
        reference_dir = Path(reference_dir)
        estimate_dir = Path(estimate_dir)
        references = {}
        estimates = {}
        for name in source_names:
            ref_path = reference_dir / f"{name}.wav"
            est_path = estimate_dir / f"{name}.wav"
            if ref_path.exists():
                ref, _ = load_audio(ref_path, sr=sample_rate, mono=True)
                references[name] = ref
            if est_path.exists():
                est, _ = load_audio(est_path, sr=sample_rate, mono=True)
                estimates[name] = est
        return self.evaluate_separation(references, estimates, source_names)

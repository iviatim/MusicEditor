from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from music_framework.core.io import load_audio, save_audio

import sys as _sys
import types as _types
import typing as _typing

if "typing.io" not in _sys.modules:
    _typing_io = _types.ModuleType("typing.io")
    TextIO = getattr(_typing, "TextIO", getattr(_typing, "IO", object))
    setattr(_typing_io, "TextIO", TextIO)
    _sys.modules["typing.io"] = _typing_io

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    nr = None

from demucs.pretrained import get_model
from demucs.apply import apply_model


def _compute_envelope(x: np.ndarray, sr: int, window_ms: float) -> np.ndarray:
    env = np.abs(x)
    win = int(sr * window_ms / 1000.0)
    if win > 1:
        kernel = np.ones(win) / win
        env = np.convolve(env, kernel, mode="same")
    return env


def _adaptive_params_for_stem(name: str) -> dict:
    n = (name or "").lower()
    if "lead" in n:
        return {
            "prop_decrease": 0.5,
            "gate_threshold": 0.01,
            "gate_floor": 0.25,
            "env_window_ms": 25.0,
        }
    if "back" in n or "vocals" in n or "vocal" in n:
        return {
            "prop_decrease": 0.6,
            "gate_threshold": 0.008,
            "gate_floor": 0.2,
            "env_window_ms": 30.0,
        }
    if "drums" in n:
        return {
            "prop_decrease": 0.4,
            "gate_threshold": 0.015,
            "gate_floor": 0.35,
            "env_window_ms": 15.0,
        }
    if "bass" in n:
        return {
            "prop_decrease": 0.5,
            "gate_threshold": 0.012,
            "gate_floor": 0.3,
            "env_window_ms": 20.0,
        }
    return {
        "prop_decrease": 0.55,
        "gate_threshold": 0.01,
        "gate_floor": 0.3,
        "env_window_ms": 25.0,
    }


def _enhance_vocal_bandwise(x: np.ndarray, sr: int, is_lead: bool) -> np.ndarray:
    n_fft = 2048
    hop = n_fft // 4
    window = np.hanning(n_fft).astype(np.float64)
    spec = _stft(x.astype(np.float64), n_fft, hop, window)
    mag = np.abs(spec)
    phase = np.angle(spec)
    freqs = np.linspace(0.0, sr / 2.0, mag.shape[-1])

    low = freqs < 120.0
    mid = (freqs >= 120.0) & (freqs < 4000.0)
    high = freqs >= 4000.0

    mag_low = mag * low
    mag_mid = mag * mid
    mag_high = mag * high

    gain_low = 1.05 if is_lead else 1.0
    gain_mid = 1.1 if is_lead else 1.05
    deesser_strength = 0.25 if is_lead else 0.35

    band_energy = mag_high.mean(axis=-1, keepdims=True) + 1e-9
    local_high = mag_high
    deesser_mask = np.clip(local_high / (2.0 * band_energy), 0.0, 1.0)
    high_gain = 1.0 - deesser_strength * deesser_mask

    mag_enh = (
        mag_low * gain_low
        + mag_mid * gain_mid
        + mag_high * high_gain
    )

    enh_spec = mag_enh * np.exp(1j * phase)
    y = _istft(enh_spec, n_fft, hop, window, len(x))
    peak = np.max(np.abs(y)) + 1e-9
    y = 0.98 * y / peak
    return y.astype(np.float32)


def postprocess_audio(
    audio: np.ndarray,
    sr: int,
    stem_name: Optional[str] = None,
) -> np.ndarray:
    x = audio.astype(np.float32)
    params = _adaptive_params_for_stem(stem_name or "")

    if NOISEREDUCE_AVAILABLE:
        x = nr.reduce_noise(
            y=x,
            sr=sr,
            prop_decrease=params["prop_decrease"],
            stationary=False,
            n_fft=2048,
            hop_length=512,
        )

    env = _compute_envelope(x, sr, params["env_window_ms"])
    thr = params["gate_threshold"]
    ratio = np.clip(env / max(thr, 1e-9), 0.0, 1.0)
    gain = params["gate_floor"] + (1.0 - params["gate_floor"]) * ratio
    x = x * gain

    name = (stem_name or "").lower()
    if "lead" in name or "back" in name or "vocal" in name:
        x = _enhance_vocal_bandwise(x, sr, is_lead=("lead" in name))

    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Разделение lead vocal / back vocal
# Mid-side + Wiener-фильтр (STFT с overlap-add) + спектральное вычитание
# ---------------------------------------------------------------------------

def _stft(signal: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """Forward STFT → (n_frames, n_fft//2+1) complex."""
    pad_total = n_fft - (len(signal) % hop) if len(signal) % hop else 0
    signal = np.concatenate([signal, np.zeros(pad_total + n_fft, dtype=signal.dtype)])
    n_frames = (len(signal) - n_fft) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_frames, n_fft),
        strides=(signal.strides[0] * hop, signal.strides[0]),
    ).copy()
    frames *= window[np.newaxis, :]
    return np.fft.rfft(frames, axis=-1)


def _istft(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray, target_len: int) -> np.ndarray:
    """Inverse STFT (overlap-add) → (target_len,)."""
    n_frames = spec.shape[0]
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros(out_len, dtype=np.float64)
    win_sum = np.zeros(out_len, dtype=np.float64)
    frames = np.fft.irfft(spec, n=n_fft, axis=-1)
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i] * window
        win_sum[s:s + n_fft] += window ** 2
    nz = win_sum > 1e-10
    out[nz] /= win_sum[nz]
    return out[:target_len]


def _separate_lead_back_vocal(
    vocals_stereo: np.ndarray,
    sr: int,
) -> tuple:
    """
    Разделить стерео-вокал на lead (центр) и backing (стороны).
    Wiener-style soft masking в STFT-домене с жёстким экспонентом
    + спектральное вычитание утечки lead из back.
    vocals_stereo: shape (2, samples).
    Возвращает (lead_mono, back_mono).
    """
    if vocals_stereo.ndim == 1:
        return vocals_stereo, np.zeros_like(vocals_stereo)

    left = vocals_stereo[0].astype(np.float64)
    right = vocals_stereo[1].astype(np.float64)

    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    orig_len = len(mid)

    n_fft = 4096
    hop = n_fft // 4
    window = np.hanning(n_fft).astype(np.float64)

    mid_spec = _stft(mid, n_fft, hop, window)
    side_spec = _stft(side, n_fft, hop, window)

    mid_mag = np.abs(mid_spec)
    side_mag = np.abs(side_spec)

    # Wiener-фильтр с умеренной экспонентой
    exp = 2.0
    mid_pow = mid_mag ** exp
    side_pow = side_mag ** exp
    denom = mid_pow + side_pow + 1e-12

    center_ratio = mid_pow / denom

    # Lead маска: мягкий переход — центр оставляем, боковое ослабляем умеренно
    lead_mask = np.where(center_ratio > 0.6, center_ratio, center_ratio * 0.35)
    lead_mask = np.clip(lead_mask, 0.0, 1.0)

    # Back маска для mid-канала: пропускаем боковое, центр ослабляем, но не убиваем
    back_mid_mask = np.where(center_ratio < 0.5, 1.0 - center_ratio, (1.0 - center_ratio) * 0.45)
    back_mid_mask = np.clip(back_mid_mask, 0.0, 1.0)

    lead_spec_out = mid_spec * lead_mask
    back_mid_spec_out = mid_spec * back_mid_mask

    lead = _istft(lead_spec_out, n_fft, hop, window, orig_len)
    back_from_mid = _istft(back_mid_spec_out, n_fft, hop, window, orig_len)
    side_recon = _istft(side_spec, n_fft, hop, window, orig_len)

    n = min(orig_len, len(side_recon), len(back_from_mid))
    lead = lead[:n]
    back = side_recon[:n] + back_from_mid[:n] * 0.85

    # Мягкое спектральное вычитание: чуть убираем утечку lead из back
    lead_stft = _stft(lead, n_fft, hop, window)
    back_stft = _stft(back[:len(lead)], n_fft, hop, window)
    lead_mag_f = np.abs(lead_stft)
    back_mag_f = np.abs(back_stft)
    back_phase = np.angle(back_stft)
    subtracted_mag = np.maximum(back_mag_f - lead_mag_f * 0.15, back_mag_f * 0.25)
    back_clean_spec = subtracted_mag * np.exp(1j * back_phase)
    back = _istft(back_clean_spec, n_fft, hop, window, n)

    # Нормализация: back подтягиваем до нормальной громкости
    for arr in (lead, back):
        peak = np.max(np.abs(arr))
        if peak > 1e-6:
            arr *= min(0.95 / peak, 1.0)

    return lead[:n].astype(np.float32), back[:n].astype(np.float32)


class SeparationResult:
    def __init__(
        self,
        sources: Dict[str, np.ndarray],
        sample_rate: int,
        original_path: Optional[Path] = None,
    ):
        self.sources = sources
        self.sample_rate = sample_rate
        self.original_path = original_path

    def keys(self) -> List[str]:
        return list(self.sources.keys())

    def get(self, name: str) -> np.ndarray:
        return self.sources[name]

    def save_all(self, out_dir: Union[str, Path], prefix: str = "track") -> List[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, audio in self.sources.items():
            path = out_dir / f"{prefix}_{name}.wav"
            save_audio(path, audio, self.sample_rate)
            paths.append(path)
        return paths


class DemucsSeparator:
    def __init__(
        self,
        model_name: str = "htdemucs",
        device: Optional[str] = None,
        segment_length: Optional[float] = 7.8,
        overlap: float = 0.25,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.segment_length = segment_length
        self.overlap = overlap
        self._model = None
        self._sources_list: Optional[List[str]] = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            self._sources_list = list(self._model.sources)

    @property
    def sources(self) -> List[str]:
        self._ensure_model()
        return self._sources_list or []

    def separate_file(
        self,
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
        split_vocals: bool = False,
        clean: bool = True,
    ) -> SeparationResult:
        """
        Разделить аудиофайл на источники.
        split_vocals — дополнительно разделить вокал на lead_vocal / back_vocal.
        clean — пост-обработка (шумоподавление).
        """
        self._ensure_model()
        path = Path(path)
        # Для lead/back vocal нужен стерео вход — панорама несёт информацию
        wav, sr = load_audio(path, sr=sample_rate, mono=(not split_vocals))
        return self.separate_array(wav, sr, original_path=path, split_vocals=split_vocals, clean=clean)

    def separate_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        original_path: Optional[Path] = None,
        split_vocals: bool = False,
        clean: bool = True,
    ) -> SeparationResult:
        self._ensure_model()
        model = self._model
        model_sr = model.samplerate
        if sample_rate != model_sr:
            import librosa
            audio = librosa.resample(
                audio.T.astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=model_sr,
            )
            if audio.ndim == 2:
                audio = audio.T  # обратно в (samples, channels)
            sample_rate = model_sr

        # Подготовка тензора: (batch, channels, samples)
        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            waveform = waveform.repeat(1, 2, 1)  # дублируем моно в стерео
        else:
            # audio shape: (samples, channels) — транспонируем в (channels, samples)
            waveform = torch.from_numpy(audio.T.astype(np.float32)).unsqueeze(0)  # (1, channels, samples)
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)
            elif waveform.shape[1] > 2:
                waveform = waveform[:, :2, :]
        waveform = waveform.to(self.device)

        with torch.no_grad():
            segments = apply_model(
                model,
                waveform,
                segment=self.segment_length,
                overlap=self.overlap,
                device=self.device,
            )
        # segments: (batch, sources, channels, samples)
        out = {}
        vocals_stereo = None
        for i, name in enumerate(self.sources):
            seg = segments[0, i].cpu().numpy()  # (channels, samples)
            if name == "vocals" and split_vocals and seg.ndim == 2 and seg.shape[0] >= 2:
                vocals_stereo = seg[:2].copy()
            mono = seg.mean(axis=0) if seg.ndim == 2 else seg
            if clean:
                mono = postprocess_audio(np.ascontiguousarray(mono), sample_rate, stem_name=name)
            out[name] = np.ascontiguousarray(mono)

        if split_vocals and vocals_stereo is not None:
            lead, back = _separate_lead_back_vocal(vocals_stereo, sample_rate)
            if clean:
                lead = postprocess_audio(lead, sample_rate, stem_name="lead_vocal")
                back = postprocess_audio(back, sample_rate, stem_name="back_vocal")
            out["lead_vocal"] = np.ascontiguousarray(lead)
            out["back_vocal"] = np.ascontiguousarray(back)

        return SeparationResult(
            sources=out,
            sample_rate=sample_rate,
            original_path=original_path,
        )

    def separate_vocals_only(
        self,
        path: Union[str, Path],
        out_path: Optional[Union[str, Path]] = None,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Удобный метод: только вокал. При указании out_path — сохранить."""
        result = self.separate_file(path, sample_rate=sample_rate)
        if "vocals" not in result.sources:
            raise KeyError("Модель не содержит источник 'vocals'")
        vocals = result.get("vocals")
        if out_path:
            save_audio(out_path, vocals, result.sample_rate)
        return vocals

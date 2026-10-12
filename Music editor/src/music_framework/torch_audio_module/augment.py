"""
Аугментации на базе TorchAudio (шум, сдвиг по времени, изменение громкости и т.д.).
Дополняет Audiomentations для пайплайнов внутри PyTorch.
"""

from typing import Optional

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


class TorchAudioAugmentor:
    """
    Набор аугментаций в тензорном виде для использования в обучении моделей.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        time_mask_param: int = 80,
        freq_mask_param: int = 80,
    ):
        self.sample_rate = sample_rate
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """Добавить белый шум с заданным SNR (dB)."""
        std = waveform.std()
        if std < 1e-8:
            return waveform
        noise = torch.randn_like(waveform) * std
        snr_linear = 10 ** (snr_db / 20.0)
        noise_scale = std / (snr_linear + 1e-8)
        return waveform + noise * noise_scale

    def time_shift(self, waveform: torch.Tensor, max_shift_sec: float = 0.2) -> torch.Tensor:
        """Случайный циклический сдвиг по времени."""
        shift = int(torch.rand(1).item() * 2 * max_shift_sec * self.sample_rate)
        shift = shift - int(waveform.shape[-1] * torch.rand(1).item())
        return torch.roll(waveform, shift, dims=-1)

    def volume(self, waveform: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Изменить громкость на gain_db децибел."""
        return F.gain(waveform, gain_db)

    def random_volume(self, waveform: torch.Tensor, low_db: float = -6.0, high_db: float = 6.0) -> torch.Tensor:
        """Случайное изменение громкости в диапазоне [low_db, high_db]."""
        g = low_db + (high_db - low_db) * torch.rand(1).item()
        return F.gain(waveform, g)

    def time_stretch(
        self,
        waveform: torch.Tensor,
        rate: float,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Изменение темпа (rate > 1 — быстрее)."""
        sr = sample_rate or self.sample_rate
        return F.resample(
            F.resample(waveform, sr, int(sr * rate)),
            int(sr * rate),
            sr,
        )

    def apply_all_random(
        self,
        waveform: torch.Tensor,
        noise_snr: Optional[float] = 15.0,
        time_shift_sec: float = 0.1,
        volume_low_db: float = -3.0,
        volume_high_db: float = 3.0,
    ) -> torch.Tensor:
        """Применить случайный набор аугментаций (для обучения)."""
        out = waveform.clone()
        if noise_snr is not None:
            out = self.add_noise(out, snr_db=noise_snr)
        if time_shift_sec > 0:
            out = self.time_shift(out, max_shift_sec=time_shift_sec)
        out = self.random_volume(out, low_db=volume_low_db, high_db=volume_high_db)
        return out

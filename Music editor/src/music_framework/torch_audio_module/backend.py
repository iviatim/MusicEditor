"""
Бэкенд TorchAudio: загрузка/сохранение, спектрограммы, преобразования в тензоры.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F


class TorchAudioBackend:
    """
    Единая точка работы с аудио в формате PyTorch тензоров.
    Поддержка спектрограмм, мел-фильтров, ресемплинга.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self._mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            power=2.0,
        )
        self._resampler: Optional[T.Resample] = None

    def load(
        self,
        path: Union[str, Path],
        target_sr: Optional[int] = None,
        mono: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Загрузка файла в тензор (channels, samples).
        При необходимости ресемплинг и сведение к моно.
        """
        path = Path(path)
        waveform, sr = torchaudio.load(str(path))

        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if target_sr is not None and sr != target_sr:
            self._resampler = T.Resample(sr, target_sr)
            waveform = self._resampler(waveform)
            sr = target_sr

        return waveform, sr

    def save(self, path: Union[str, Path], waveform: torch.Tensor, sr: int) -> Path:
        """Сохранение тензора (channels, samples) в файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(path), waveform, sr)
        return path

    def to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """(C, T) -> (C, n_mels, time)."""
        return self._mel_spectrogram(waveform)

    def to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """(C, T) -> (C, freq, time)."""
        return self._spectrogram(waveform)

    def to_db(self, spec: torch.Tensor, ref: float = 1.0, amin: float = 1e-10) -> torch.Tensor:
        """Амплитуда -> dB."""
        return F.amplitude_to_DB(spec, multiplier=20.0, amin=amin, db_multiplier=1.0)

    def numpy_to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        """(samples,) или (samples, ch) -> (ch, samples)."""
        if audio.ndim == 1:
            audio = np.expand_dims(audio, 0)
        elif audio.ndim == 2 and audio.shape[1] > audio.shape[0]:
            pass
        else:
            audio = audio.T if audio.shape[0] > audio.shape[1] else audio
        return torch.from_numpy(audio).float()

    def tensor_to_numpy(self, waveform: torch.Tensor) -> np.ndarray:
        """(ch, samples) -> (samples,) или (samples, ch)."""
        x = waveform.numpy()
        if x.shape[0] == 1:
            return x.squeeze(0)
        return x.T

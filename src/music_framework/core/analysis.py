"""
Аналитические задачи и алгоритмы постобработки на базе Librosa.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import librosa
from librosa import feature as lb_feature


class AudioAnalyzer:
    """
    Анализатор аудио: спектрограммы, ритм, тональность, сегментация и др.
    """

    def __init__(self, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_and_analyze(
        self,
        path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True,
    ) -> "AnalysisResult":
        """Загрузить файл и вычислить базовые дескрипторы."""
        sr = sr or self.sr
        y, _ = librosa.load(path, sr=sr, mono=mono)
        return self.analyze(y, sr=sr)

    def analyze(self, y: np.ndarray, sr: Optional[int] = None) -> "AnalysisResult":
        """Вычислить дескрипторы по массиву сэмплов."""
        sr = sr or self.sr
        n_fft = self.n_fft
        hop = self.hop_length

        # Спектрограммы
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ритм
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if tempo.size else 0.0

        # Тональность
        chroma = lb_feature.chroma_stft(
            S=np.abs(S) ** 2, sr=sr, n_fft=n_fft, hop_length=hop
        )
        tonnetz = lb_feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        # Zero-crossing rate, RMS
        zcr = lb_feature.zero_crossing_rate(y, hop_length=hop)
        rms = lb_feature.rms(y=y, hop_length=hop)

        # Сегментация по повторениям (упрощённо)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)

        return AnalysisResult(
            sr=sr,
            duration_sec=len(y) / sr,
            tempo_bpm=float(tempo),
            spectrogram_db=S_db,
            mel_spectrogram_db=mel_db,
            chroma=chroma,
            tonnetz=tonnetz,
            zcr=zcr,
            rms=rms,
            onset_frames=onset_frames,
            beat_frames=beat_frames,
            onset_env=onset_env,
        )

    def get_beat_times(self, y: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """Времена ударов в секундах."""
        sr = sr or self.sr
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        return librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

    def get_segments_simple(
        self,
        y: np.ndarray,
        sr: Optional[int] = None,
        top_n: int = 10,
    ) -> np.ndarray:
        """Упрощённая сегментация по onset (границы в секундах)."""
        sr = sr or self.sr
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, backtrack=True
        )
        times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        return times[:top_n] if top_n else times


class AnalysisResult:
    """Результат анализа одного трека."""

    __slots__ = (
        "sr",
        "duration_sec",
        "tempo_bpm",
        "spectrogram_db",
        "mel_spectrogram_db",
        "chroma",
        "tonnetz",
        "zcr",
        "rms",
        "onset_frames",
        "beat_frames",
        "onset_env",
    )

    def __init__(
        self,
        sr: int,
        duration_sec: float,
        tempo_bpm: float,
        spectrogram_db: np.ndarray,
        mel_spectrogram_db: np.ndarray,
        chroma: np.ndarray,
        tonnetz: np.ndarray,
        zcr: np.ndarray,
        rms: np.ndarray,
        onset_frames: np.ndarray,
        beat_frames: np.ndarray,
        onset_env: np.ndarray,
    ):
        self.sr = sr
        self.duration_sec = duration_sec
        self.tempo_bpm = tempo_bpm
        self.spectrogram_db = spectrogram_db
        self.mel_spectrogram_db = mel_spectrogram_db
        self.chroma = chroma
        self.tonnetz = tonnetz
        self.zcr = zcr
        self.rms = rms
        self.onset_frames = onset_frames
        self.beat_frames = beat_frames
        self.onset_env = onset_env

    def summary(self) -> str:
        return (
            f"Duration: {self.duration_sec:.2f}s, SR: {self.sr}, "
            f"Tempo: {self.tempo_bpm:.1f} BPM"
        )

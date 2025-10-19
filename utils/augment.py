from __future__ import annotations

"""
Lightweight 1D time-series augmentations for ECG identification.

Applies a simple combination of time shift (zero-padded), amplitude scaling,
and additive Gaussian noise at a target SNR. Designed to operate on numpy
arrays prior to conversion to tensors.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ECGAugment1D:
    """Compose simple augmentations for ECG waveforms.

    Parameters
    ----------
    fs : int
        Sampling rate in Hz.
    time_shift_ms : float
        Maximum absolute time shift in milliseconds. Shift is zero-padded,
        not circular.
    amplitude_scale : float
        Maximum relative amplitude perturbation. The amplitude scale factor
        is sampled uniformly from [1 - amplitude_scale, 1 + amplitude_scale].
    noise_snr_db : float
        Target signal-to-noise ratio (in dB) for additive Gaussian noise.
        If <= 0, noise is disabled.
    apply_prob : float
        Probability (0-1) to apply the augmentation to a sample.
    """

    fs: int
    time_shift_ms: float = 0.0
    amplitude_scale: float = 0.0
    noise_snr_db: float = 0.0
    apply_prob: float = 0.0

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if not isinstance(signal, np.ndarray):
            signal = np.asarray(signal)

        if self.apply_prob <= 0.0 or np.random.rand() > self.apply_prob:
            return signal

        augmented = signal.astype(np.float32, copy=True)

        # 1) Time shift with zero padding
        if self.time_shift_ms and self.time_shift_ms > 0.0:
            max_shift_samples = int(round(self.time_shift_ms * 1e-3 * float(self.fs)))
            if max_shift_samples > 0:
                shift = int(np.random.randint(-max_shift_samples, max_shift_samples + 1))
                if shift != 0:
                    rolled = np.roll(augmented, shift)
                    if shift > 0:
                        rolled[:shift] = 0.0
                    else:
                        rolled[shift:] = 0.0
                    augmented = rolled

        # 2) Amplitude scaling
        if self.amplitude_scale and self.amplitude_scale > 0.0:
            scale = 1.0 + float(np.random.uniform(-self.amplitude_scale, self.amplitude_scale))
            augmented = augmented * np.float32(scale)

        # 3) Additive Gaussian noise at target SNR (dB)
        if self.noise_snr_db and self.noise_snr_db > 0.0:
            power = float(np.mean(augmented.astype(np.float32) ** 2))
            if power > 0.0:
                snr_linear = 10.0 ** (self.noise_snr_db / 10.0)
                noise_var = power / snr_linear
                noise_std = float(np.sqrt(noise_var))
                noise = np.random.normal(loc=0.0, scale=noise_std, size=augmented.shape).astype(np.float32)
                augmented = augmented + noise

        return augmented

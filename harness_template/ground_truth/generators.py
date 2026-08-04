"""
Layer 3: Ground Truth Generators

Create synthetic signals with known parameters for algorithm verification.
"""

import numpy as np
from scipy.signal import fftconvolve
from typing import Tuple, Optional


def generate_simple_echo_path(delay: int = 256,
                              decay: float = 0.3,
                              length: int = 8000) -> np.ndarray:
    """
    Generate a trivial echo path: single delay + attenuation.

    Use this FIRST when debugging — if the algorithm can't learn
    a single tap, it won't learn a complex path.

    Test progression:
    1. delay = k * M (partition-aligned) → should work perfectly
    2. delay = k * M + offset → tests sub-partition resolution
    3. Multiple taps → tests multi-partition convergence

    Parameters
    ----------
    delay : int
        Delay in samples
    decay : float
        Attenuation factor (0 to 1)
    length : int
        Total echo path length in samples

    Returns
    -------
    h : ndarray, shape (length,)
    """
    h = np.zeros(length)
    h[delay] = decay
    return h


def generate_realistic_echo_path(
    sampling_rate: int = 16000,
    filter_length_ms: float = 500.0,
    initial_delay_ms: float = 45.0,
    reverberation_time_ms: float = 300.0,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a realistic room impulse response.

    Components:
    1. Direct path at initial_delay
    2. Early reflections (8 taps)
    3. Late reverberation (filtered noise with exponential decay)
    4. Secondary reflections (15 smaller taps)
    5. Fade-out envelope

    Parameters
    ----------
    sampling_rate : int
        Sample rate in Hz
    filter_length_ms : float
        Total filter length in ms
    initial_delay_ms : float
        Direct path delay in ms
    reverberation_time_ms : float
        RT60 reverberation time in ms
    seed : int
        Random seed for reproducibility

    Returns
    -------
    h : ndarray
        Room impulse response
    """
    rng = np.random.RandomState(seed)
    filter_length = int(filter_length_ms * sampling_rate / 1000)
    initial_delay = int(initial_delay_ms * sampling_rate / 1000)
    rt60_samples = int(reverberation_time_ms * sampling_rate / 1000)

    h = np.zeros(filter_length)

    # Direct path
    h[initial_delay] = 0.6

    # Early reflections
    early_start = initial_delay + 1
    early_end = min(initial_delay + int(80 * sampling_rate / 1000), filter_length)
    n_early = min(8, early_end - early_start)
    if n_early > 0:
        early_positions = rng.choice(
            range(early_start, early_end),
            size=n_early,
            replace=False
        )
        early_amps = rng.uniform(0.15, 0.40, n_early)
        for pos, amp in zip(early_positions, early_amps):
            h[pos] = amp * rng.choice([-1, 1])

    # Late reverberation
    reverb_start = early_end
    reverb_length = min(rt60_samples * 2, filter_length - reverb_start)
    if reverb_length > 0:
        noise = rng.randn(reverb_length) * 0.02
        decay_rate = 3.0 * np.log(10) / rt60_samples
        envelope = np.exp(-decay_rate * np.arange(reverb_length))
        h[reverb_start:reverb_start + reverb_length] += noise * envelope

    # Secondary reflections
    sec_start = reverb_start
    sec_end = min(filter_length - int(0.1 * filter_length), filter_length)
    n_sec = min(15, sec_end - sec_start)
    if n_sec > 0:
        sec_positions = rng.choice(
            range(sec_start, sec_end),
            size=n_sec,
            replace=False
        )
        sec_amps = rng.uniform(0.05, 0.15, n_sec)
        for pos, amp in zip(sec_positions, sec_amps):
            h[pos] += amp * rng.choice([-1, 1])

    # Fade-out envelope (last 10%)
    fade_start = int(0.9 * filter_length)
    fade_length = filter_length - fade_start
    if fade_length > 0:
        fade = np.linspace(1, 0, fade_length)
        h[fade_start:] *= fade

    return h


def generate_test_signals(
    echo_path: np.ndarray,
    signal_length: int = 160000,
    sample_rate: int = 16000,
    speech_file: Optional[str] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reference and microphone signals.

    ref = clean speech (or white noise)
    mic = conv(ref, echo_path)

    Parameters
    ----------
    echo_path : ndarray
        True echo path impulse response
    signal_length : int
        Length of output signals in samples
    sample_rate : int
        Sample rate
    speech_file : str, optional
        Path to WAV file for reference. If None, uses white noise.
    seed : int
        Random seed (for noise generation)

    Returns
    -------
    ref : ndarray — Reference signal
    mic : ndarray — Microphone signal (echo)
    """
    if speech_file is not None:
        import soundfile as sf
        ref, sr = sf.read(speech_file)
        if sr != sample_rate:
            import librosa
            ref = librosa.resample(ref, orig_sr=sr, target_sr=sample_rate)
        ref = ref[:signal_length]
    else:
        rng = np.random.RandomState(seed)
        ref = rng.randn(signal_length) * 0.1

    # Microphone = convolution of reference with echo path
    mic = fftconvolve(ref, echo_path, mode='full')[:len(ref)]

    return ref, mic

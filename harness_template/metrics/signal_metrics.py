"""
Layer 2: Signal-Level Metrics

Metrics computed from time-domain signals: ERLE, SNR, output RMS.
"""

import numpy as np


def compute_erle(mic_signal: np.ndarray, output_signal: np.ndarray,
                 transient: int = 0) -> float:
    """
    Echo Return Loss Enhancement (ERLE).

    Measures how much the echo was reduced by the adaptive filter.
    Higher is better.

    ERLE = 10 * log10(Σ mic² / Σ output²)

    Parameters
    ----------
    mic_signal : ndarray
        Microphone signal (before cancellation)
    output_signal : ndarray
        Filter output (after cancellation)
    transient : int
        Number of initial samples to skip (convergence period)

    Returns
    -------
    float
        ERLE in dB
    """
    mic_eval = mic_signal[transient:]
    out_eval = output_signal[transient:]
    mic_power = np.sum(mic_eval ** 2)
    out_power = np.sum(out_eval ** 2)
    return float(10 * np.log10((mic_power + 1e-10) / (out_power + 1e-10)))


def compute_output_rms(output_signal: np.ndarray, transient: int = 0) -> float:
    """
    RMS of the output signal (after transient).

    Lower is better for echo cancellation (should approach 0).

    Parameters
    ----------
    output_signal : ndarray
        Filter output signal
    transient : int
        Samples to skip

    Returns
    -------
    float
        RMS value
    """
    out_eval = output_signal[transient:]
    return float(np.sqrt(np.mean(out_eval ** 2)))


def compute_snr(signal: np.ndarray, noise: np.ndarray,
                transient: int = 0) -> float:
    """
    Signal-to-Noise Ratio.

    SNR = 10 * log10(Σ signal² / Σ noise²)

    Parameters
    ----------
    signal : ndarray
        Clean signal
    noise : ndarray
        Noise component (or residual error)
    transient : int
        Samples to skip

    Returns
    -------
    float
        SNR in dB
    """
    s = signal[transient:]
    n = noise[transient:]
    s_power = np.sum(s ** 2)
    n_power = np.sum(n ** 2)
    return float(10 * np.log10((s_power + 1e-10) / (n_power + 1e-10)))

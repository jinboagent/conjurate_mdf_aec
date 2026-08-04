"""
Layer 2: Echo Path Comparison Metrics

Compare estimated echo path against ground truth.
"""

import numpy as np
from typing import Dict


def compute_echo_path_metrics(true_path: np.ndarray,
                              est_path: np.ndarray) -> Dict[str, float]:
    """
    Compute all echo path comparison metrics.

    Parameters
    ----------
    true_path : ndarray
        Ground truth echo path impulse response
    est_path : ndarray
        Estimated echo path from adaptive filter

    Returns
    -------
    dict with keys:
        nmse_db : float — Normalized Mean Squared Error (dB)
        correlation : float — Pearson correlation coefficient
        coherence : float — Frequency-domain coherence
        delay_error_samples : int — Error in peak delay estimation
        amplitude_error_db : float — Error in peak amplitude (dB)
        passed : bool — All metrics within targets
    """
    min_len = min(len(true_path), len(est_path))
    true = true_path[:min_len]
    est = est_path[:min_len]

    # NMSE
    nmse = float(10 * np.log10(
        np.sum((est - true) ** 2) / (np.sum(true ** 2) + 1e-10) + 1e-10
    ))

    # Correlation
    corr = float(np.corrcoef(est, true)[0, 1])

    # Coherence (frequency domain)
    H_true = np.fft.rfft(true)
    H_est = np.fft.rfft(est)
    coherence = float(
        np.abs(np.sum(H_est * np.conj(H_true))) /
        (np.sqrt(np.sum(np.abs(H_est) ** 2) * np.sum(np.abs(H_true) ** 2)) + 1e-10)
    )

    # Delay error
    true_peak = int(np.argmax(np.abs(true)))
    est_peak = int(np.argmax(np.abs(est)))
    delay_error = abs(true_peak - est_peak)

    # Amplitude error
    true_peak_val = np.abs(true[true_peak])
    est_peak_val = np.abs(est[est_peak])
    amp_error = float(20 * np.log10(
        (est_peak_val + 1e-10) / (true_peak_val + 1e-10)
    ))

    return {
        'nmse_db': nmse,
        'correlation': corr,
        'coherence': coherence,
        'delay_error_samples': delay_error,
        'amplitude_error_db': amp_error,
    }

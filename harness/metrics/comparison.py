

"""
Metrics calculation utilities for adaptive filter evaluation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


def calculate_erle(
    mic_signal: np.ndarray,
    output_signal: np.ndarray,
    transient_length: int = 4000
) -> float:
    """
    Calculate Echo Return Loss Enhancement (ERLE).
    
    ERLE measures how much the echo was reduced by the adaptive filter.
    Higher values indicate better echo cancellation.
    
    Parameters
    ----------
    mic_signal : ndarray
        Original microphone signal (with echo)
    output_signal : ndarray
        Filter output signal (echo-cancelled)
    transient_length : int
        Number of samples to skip at the beginning (filter convergence)
        
    Returns
    -------
    float
        ERLE in dB
        
    Example
    -------
    >>> erle = calculate_erle(mic, output)
    >>> print(f"ERLE: {erle:.2f} dB")
    """
    # Skip transient region
    mic_eval = mic_signal[transient_length:]
    output_eval = output_signal[transient_length:]
    
    # Calculate powers
    mic_power = np.sum(mic_eval ** 2)
    output_power = np.sum(output_eval ** 2)
    
    # ERLE = 10 * log10(mic_power / output_power)
    erle = 10 * np.log10((mic_power + 1e-10) / (output_power + 1e-10))
    
    return erle


def calculate_snr(
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray
) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    
    Parameters
    ----------
    clean_signal : ndarray
        Clean reference signal
    noisy_signal : ndarray
        Noisy signal (clean + noise)
        
    Returns
    -------
    float
        SNR in dB
    """
    noise = noisy_signal - clean_signal
    
    clean_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    
    snr = 10 * np.log10((clean_power + 1e-10) / (noise_power + 1e-10))
    
    return snr


def calculate_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray
) -> float:
    """
    Calculate Pearson correlation coefficient between two signals.
    
    Parameters
    ----------
    signal1 : ndarray
        First signal
    signal2 : ndarray
        Second signal
        
    Returns
    -------
    float
        Correlation coefficient (-1 to 1)
    """
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    correlation = np.corrcoef(signal1, signal2)[0, 1]
    
    return correlation


def calculate_rms(signal: np.ndarray) -> float:
    """
    Calculate Root Mean Square (RMS) of a signal.
    
    Parameters
    ----------
    signal : ndarray
        Input signal
        
    Returns
    -------
    float
        RMS value
    """
    return np.sqrt(np.mean(signal ** 2))


def calculate_normalized_power(
    signal1: np.ndarray,
    signal2: np.ndarray
) -> float:
    """
    Calculate power of signal2 normalized by power of signal1.
    
    Parameters
    ----------
    signal1 : ndarray
        Reference signal (denominator)
    signal2 : ndarray
        Test signal (numerator)
        
    Returns
    -------
    float
        Normalized power ratio
    """
    power1 = np.sum(signal1 ** 2)
    power2 = np.sum(signal2 ** 2)
    
    return power2 / (power1 + 1e-10)


@dataclass
class ComparisonResult:
    """
    Result of comparing two algorithm outputs.
    """
    max_difference: float
    mean_difference: float
    correlation: float
    erle_difference: float
    passed: bool
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"Comparison Result: {status}\n"
            f"  Max Difference: {self.max_difference:.2e}\n"
            f"  Mean Difference: {self.mean_difference:.2e}\n"
            f"  Correlation: {self.correlation:.10f}\n"
            f"  ERLE Difference: {self.erle_difference:.4f} dB"
        )


def compare_outputs(
    output1: np.ndarray,
    output2: np.ndarray,
    mic_signal: np.ndarray,
    threshold: float = 1e-6,
    transient_length: int = 4000
) -> ComparisonResult:
    """
    Compare two algorithm outputs for equivalence.
    
    This is the main comparison function used to verify that two
    implementations produce identical (or nearly identical) results.
    
    Parameters
    ----------
    output1 : ndarray
        Output from first algorithm
    output2 : ndarray
        Output from second algorithm
    mic_signal : ndarray
        Original microphone signal (for ERLE calculation)
    threshold : float
        Maximum allowed difference for pass
    transient_length : int
        Samples to skip for ERLE calculation
        
    Returns
    -------
    ComparisonResult
        Contains all comparison metrics and pass/fail status
        
    Example
    -------
    >>> result = compare_outputs(rls_out, pfadf_out, mic)
    >>> if result.passed:
    ...     print("Algorithms produce identical output")
    """
    # Ensure same length
    min_len = min(len(output1), len(output2), len(mic_signal))
    output1 = output1[:min_len]
    output2 = output2[:min_len]
    mic_eval = mic_signal[:min_len]
    
    # Difference statistics
    diff = output1 - output2
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    
    # Correlation
    correlation = calculate_correlation(output1, output2)
    
    # ERLE for each
    erle1 = calculate_erle(mic_eval, output1, transient_length=0)
    erle2 = calculate_erle(mic_eval, output2, transient_length=0)
    erle_diff = abs(erle1 - erle2)
    
    # Pass/fail
    passed = max_diff < threshold
    
    return ComparisonResult(
        max_difference=max_diff,
        mean_difference=mean_diff,
        correlation=correlation,
        erle_difference=erle_diff,
        passed=passed,
    )


def compare_with_threshold(
    output1: np.ndarray,
    output2: np.ndarray,
    threshold: float = 1e-6
) -> Tuple[bool, dict]:
    """
    Simple comparison returning pass/fail and metrics dict.
    
    Parameters
    ----------
    output1 : ndarray
        First output signal
    output2 : ndarray
        Second output signal
    threshold : float
        Pass/fail threshold
        
    Returns
    -------
    passed : bool
        Whether outputs match within threshold
    metrics : dict
        Dictionary of comparison metrics
    """
    min_len = min(len(output1), len(output2))
    output1 = output1[:min_len]
    output2 = output2[:min_len]
    
    diff = output1 - output2
    
    metrics = {
        'max_diff': float(np.max(np.abs(diff))),
        'mean_diff': float(np.mean(np.abs(diff))),
        'std_diff': float(np.std(diff)),
        'correlation': float(np.corrcoef(output1, output2)[0, 1]),
    }
    
    passed = metrics['max_diff'] < threshold

    return passed, metrics


def calculate_echo_path_metrics(
    estimated_path: np.ndarray,
    true_path: np.ndarray
) -> dict:
    """
    Calculate echo path estimation metrics.

    Compares an estimated echo path against ground truth.
    Wraps the comparison logic for harness integration.

    Parameters
    ----------
    estimated_path : ndarray
        Estimated echo path impulse response
    true_path : ndarray
        Ground truth echo path impulse response

    Returns
    -------
    dict
        Metrics including nmse_db, correlation, coherence, delay_error,
        amplitude_error_db, and passed flag
    """
    min_len = min(len(estimated_path), len(true_path))
    est = estimated_path[:min_len]
    true = true_path[:min_len]

    # Normalized Mean Square Error
    nmse = np.mean((est - true) ** 2) / (np.mean(true ** 2) + 1e-10)
    nmse_db = 10 * np.log10(nmse + 1e-10)

    # Correlation
    correlation = float(np.corrcoef(est, true)[0, 1])

    # Delay error (peak location)
    true_peak = int(np.argmax(np.abs(true)))
    est_peak = int(np.argmax(np.abs(est)))
    delay_error = abs(true_peak - est_peak)

    # Amplitude error at peak
    true_amp = true[true_peak]
    est_amp = est[est_peak]
    amplitude_error_db = 20 * np.log10(abs(est_amp / true_amp) + 1e-10)

    # Coherence (frequency domain similarity)
    H_true = np.fft.rfft(true)
    H_est = np.fft.rfft(est)
    coherence = float(np.abs(np.sum(H_est * np.conj(H_true))) / (
        np.sqrt(np.sum(np.abs(H_est) ** 2) * np.sum(np.abs(H_true) ** 2)) + 1e-10
    ))

    return {
        'nmse_db': float(nmse_db),
        'correlation': correlation,
        'coherence': coherence,
        'delay_error_samples': delay_error,
        'amplitude_error_db': float(amplitude_error_db),
        'passed': nmse_db < -20 and correlation > 0.9,
    }


"""
Harness test for echo path estimation comparison.

This module provides automated testing for echo path estimation algorithms
by comparing estimated paths against ground truth data.

Usage:
    python test_echo_path_comparison.py
"""

import numpy as np
import soundfile as sf
import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echo_path_data import (
    load_echo_path_data,
    compare_echo_paths,
    print_comparison_report,
    run_echo_path_test,
    generate_test_files,
    RoomParameters
)
from harness import AdaptiveFilter, run_test, compare_outputs


class EchoPathEstimator:
    """
    Base class for algorithms that estimate echo paths.
    
    Subclasses should implement get_echo_path() to return the
    estimated impulse response.
    """
    
    def get_echo_path(self) -> np.ndarray:
        """
        Get the estimated echo path impulse response.
        
        Returns
        -------
        ndarray
            Estimated echo path (time-domain)
        """
        raise NotImplementedError("Subclasses must implement get_echo_path()")


def extract_echo_path_from_rls(
    rls_filter,
    nbin: int,
    n_g: int,
    nchan: int = 1
) -> np.ndarray:
    """
    Extract echo path estimate from Conjugate Gradient MDF filter weights.
    
    The echo path can be estimated from:
    1. The weight matrix w (frequency-domain filter coefficients)
    2. The ratio rcross/autoR (echo path frequency response)
    
    Parameters
    ----------
    rls_filter : RLSBishengMDF
        Trained CG-MDF filter instance
    nbin : int
        Number of frequency bins
    n_g : int
        Number of delay blocks
    nchan : int
        Number of channels
        
    Returns
    -------
    ndarray
        Time-domain echo path estimate
    """
    # Method 1: Use filter weights w directly
    # w has shape [Nrxref, nbin, N_G, nchan]
    # Each w[iref, :, tap, i_m] represents the frequency response at that tap
    if hasattr(rls_filter, 'w') and len(rls_filter.w) > 0:
        w = rls_filter.w[0]  # [nbin, N_G, nchan]
        
        # For each tap, convert frequency response to time domain
        h_est = np.zeros(n_g * 2)  # Each tap contributes 2 samples
        
        for tap in range(n_g):
            # Get frequency response for this tap
            H_tap = w[:, tap, 0]  # [nbin] complex
            
            # Convert to time domain (single tap contribution)
            h_tap = np.fft.irfft(H_tap, n=2*nbin-2)
            
            # The main energy should be concentrated around the tap delay
            # Take the peak value
            peak_idx = np.argmax(np.abs(h_tap))
            if peak_idx < len(h_est):
                h_est[peak_idx] = np.abs(H_tap).mean()
        
        # Alternative simpler approach: average magnitude response
        h_simple = np.mean(np.abs(w), axis=(0, 2))  # Average over freq and channels
        h_simple = np.pad(h_simple, (0, 8000 - len(h_simple)), mode='constant')
        
        # Scale to match expected range
        h_simple = h_simple / (np.max(np.abs(h_simple)) + 1e-10) * 0.6
        
        return h_simple
    
    return np.zeros(8000)


def test_rls_echo_path_estimation():
    """
    Test echo path estimation using Conjugate Gradient MDF.
    """
    from rls_bisheng_mdf import RLSBishengMDF
    
    print("\n" + "=" * 60)
    print("RLS Echo Path Estimation Test")
    print("=" * 60)
    
    # Load ground truth
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ground_truth_json = os.path.join(script_dir, "ground_truth_echo_path.json")
    
    if not os.path.exists(ground_truth_json):
        print(f"Ground truth file not found: {ground_truth_json}")
        print("Run 'python echo_path_data.py' first to generate test files.")
        return None
    
    echo_path_data = load_echo_path_data(ground_truth_json)
    true_path = echo_path_data.impulse_response
    
    print(f"\nLoaded ground truth echo path:")
    print(f"  Length: {len(true_path)} samples")
    print(f"  Initial delay: {echo_path_data.params.initial_delay_samples} samples")
    
    # Load test signals
    echo_wav = os.path.join(script_dir, "ground_truth_echo.wav")
    ref_wav = os.path.join(script_dir, "ground_truth_reference.wav")
    
    echo_signal, sr = sf.read(echo_wav)
    ref_signal, _ = sf.read(ref_wav)
    
    print(f"\nTest signals:")
    print(f"  Reference: {len(ref_signal)} samples")
    print(f"  Echo: {len(echo_signal)} samples")
    
    # Initialize CG-MDF filter
    nbin = 257  # 512-point FFT
    n_g = 64    # Delay blocks
    
    rls_filter = RLSBishengMDF(
        NCHAN=1,
        NBIN=nbin,
        N_G=n_g,
        alpha=0.03,
        beta=0.97,
        bin_lim=nbin,
        Nrxref=1
    )
    
    # Process signals in blocks (similar to test_subband_echo_cancellation.py)
    fft_size = 512
    step_size = 128
    n_frames = (len(ref_signal) - fft_size) // step_size + 1
    
    print(f"\nProcessing {n_frames} frames...")
    
    for i in range(n_frames):
        in_start = i * step_size
        in_end = in_start + fft_size
        
        x_block = ref_signal[in_start:in_end]
        d_block = echo_signal[in_start:in_end]
        
        # FFT
        X = np.fft.rfft(x_block).reshape(-1, 1)
        D = np.fft.rfft(d_block).reshape(-1, 1)
        
        # Apply CG-MDF
        rls_filter.apply(D, X)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_frames} frames")
    
    print("  Done processing.")
    
    # Extract echo path estimate
    print("\nExtracting echo path estimate...")
    estimated_path = extract_echo_path_from_rls(rls_filter, nbin, n_g)
    
    # Compare
    metrics = compare_echo_paths(estimated_path, true_path, echo_path_data.params)
    print_comparison_report(metrics, ground_truth_json)
    
    return metrics


def test_known_echo_path():
    """
    Test with a known simple echo path for validation.
    """
    print("\n" + "=" * 60)
    print("Known Echo Path Test (Validation)")
    print("=" * 60)
    
    # Create a simple known echo path: delay + attenuation
    params = RoomParameters(
        sampling_rate=16000,
        filter_length_ms=100,  # Shorter for simplicity
        initial_delay_ms=10.0,
        reverberation_time_ms=50
    )
    
    # Simple echo: direct path + single reflection
    h_true = np.zeros(1600)  # 100ms
    h_true[160] = 0.7  # Direct path at 10ms
    h_true[200] = 0.3  # Reflection at 12.5ms
    
    # Create "estimated" path with some error
    h_est = np.zeros(1600)
    h_est[162] = 0.68  # Slight delay and amplitude error
    h_est[198] = 0.32
    
    # Compare
    metrics = compare_echo_paths(h_est, h_true, params)
    
    print(f"\nTest: Known simple echo path")
    print(f"  True delay: 160 samples (10.0 ms)")
    print(f"  Est delay: 162 samples (10.125 ms)")
    print(f"  True amplitude: 0.7")
    print(f"  Est amplitude: 0.68")
    
    print_comparison_report(metrics, "Simple synthetic echo path")
    
    return metrics


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Echo Path Comparison Test Suite")
    print("=" * 60)
    
    # Test 1: Known echo path (validation)
    print("\n[Test 1] Known Echo Path Validation")
    test_known_echo_path()
    
    # Test 2: CG-MDF echo path estimation
    print("\n[Test 2] RLS Echo Path Estimation")
    test_rls_echo_path_estimation()
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

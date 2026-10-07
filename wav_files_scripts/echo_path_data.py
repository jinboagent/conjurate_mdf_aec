
"""
Echo Path Data Generation and Comparison

This module generates echo path impulse responses and saves them as data files
for comparison with estimated echo paths from adaptive filters.

Integration with harness:
- Generates ground truth echo path
- Creates echo WAV files for testing
- Provides comparison metrics for echo path estimation accuracy
"""

import numpy as np
import soundfile as sf
import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from echo_path_generator import (
    RoomParameters,
    generate_time_domain_echo_path,
    generate_frequency_domain_echo_path
)


@dataclass
class EchoPathData:
    """Container for echo path data and metadata."""
    impulse_response: np.ndarray
    frequency_response: np.ndarray
    frequencies: np.ndarray
    params: RoomParameters
    seed: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'impulse_response': self.impulse_response.tolist(),
            'frequency_response_magnitude': np.abs(self.frequency_response).tolist(),
            'frequency_response_phase': np.angle(self.frequency_response).tolist(),
            'frequencies': self.frequencies.tolist(),
            'params': asdict(self.params),
            'seed': self.seed,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EchoPathData':
        """Load from dictionary."""
        params = RoomParameters(**data['params'])
        return cls(
            impulse_response=np.array(data['impulse_response']),
            frequency_response=(
                np.array(data['frequency_response_magnitude']) * 
                np.exp(1j * np.array(data['frequency_response_phase']))
            ),
            frequencies=np.array(data['frequencies']),
            params=params,
            seed=data['seed'],
            metadata=data.get('metadata', {})
        )


def save_echo_path_data(
    echo_path_data: EchoPathData,
    save_path: str
) -> None:
    """
    Save echo path data to a JSON file.
    
    Parameters
    ----------
    echo_path_data : EchoPathData
        Echo path data to save
    save_path : str
        Path to save the JSON file
    """
    data_dict = echo_path_data.to_dict()
    
    with open(save_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"Saved echo path data to: {save_path}")


def load_echo_path_data(load_path: str) -> EchoPathData:
    """
    Load echo path data from a JSON file.
    
    Parameters
    ----------
    load_path : str
        Path to the JSON file
        
    Returns
    -------
    EchoPathData
        Loaded echo path data
    """
    with open(load_path, 'r') as f:
        data_dict = json.load(f)
    
    return EchoPathData.from_dict(data_dict)


def generate_echo_signal(
    input_signal: np.ndarray,
    echo_path: np.ndarray,
    sr: int = 16000
) -> np.ndarray:
    """
    Generate echo signal by convolving input with echo path.
    
    Parameters
    ----------
    input_signal : ndarray
        Input speech signal
    echo_path : ndarray
        Echo path impulse response
    sr : int
        Sampling rate
        
    Returns
    -------
    ndarray
        Echo signal (microphone signal)
    """
    # Convolve input with echo path
    echo_signal = np.convolve(input_signal, echo_path, mode='full')
    
    # Truncate to original length
    echo_signal = echo_signal[:len(input_signal)]
    
    return echo_signal


def generate_test_files(
    input_wav_path: str,
    output_dir: str,
    params: Optional[RoomParameters] = None,
    seed: int = 42,
    prefix: str = "test"
) -> Tuple[str, str, str]:
    """
    Generate test files for echo cancellation evaluation.
    
    Creates:
    1. Echo path impulse response (JSON)
    2. Echo/microphone signal (WAV)
    3. Reference signal copy (WAV)
    
    Parameters
    ----------
    input_wav_path : str
        Path to input/clean speech WAV file
    output_dir : str
        Directory to save output files
    params : RoomParameters, optional
        Room acoustic parameters
    seed : int
        Random seed for reproducibility
    prefix : str
        Filename prefix
        
    Returns
    -------
    tuple
        (echo_path_json, echo_wav, reference_wav) paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if params is None:
        params = RoomParameters(
            sampling_rate=16000,
            filter_length_ms=500,
            initial_delay_ms=45.0,
            reverberation_time_ms=300
        )
    
    print("=" * 60)
    print("Generating Echo Path Test Files")
    print("=" * 60)
    
    # Load input speech
    print(f"\nLoading input speech: {input_wav_path}")
    speech, sr = sf.read(input_wav_path)
    
    # Ensure correct sampling rate
    if sr != params.sampling_rate:
        print(f"Resampling from {sr} Hz to {params.sampling_rate} Hz")
        import librosa
        speech = librosa.resample(speech, orig_sr=sr, target_sr=params.sampling_rate)
        sr = params.sampling_rate
    
    print(f"  Input length: {len(speech)} samples ({len(speech)/sr:.2f} s)")
    
    # Generate echo path
    print(f"\nGenerating echo path...")
    print(f"  Filter length: {params.filter_length_ms} ms ({params.filter_length_samples} samples)")
    print(f"  Initial delay: {params.initial_delay_ms} ms ({params.initial_delay_samples} samples)")
    
    h = generate_time_domain_echo_path(params, seed=seed)
    H, freqs = generate_frequency_domain_echo_path(h, params)
    
    # Create data container
    metadata = {
        'created_at': datetime.now().isoformat(),
        'input_file': os.path.basename(input_wav_path),
        'description': 'Ground truth echo path for office room environment'
    }
    
    echo_path_data = EchoPathData(
        impulse_response=h,
        frequency_response=H,
        frequencies=freqs,
        params=params,
        seed=seed,
        metadata=metadata
    )
    
    # Save echo path data
    echo_path_json = os.path.join(output_dir, f"{prefix}_echo_path.json")
    save_echo_path_data(echo_path_data, echo_path_json)
    
    # Generate echo signal
    print(f"\nGenerating echo signal...")
    echo_signal = generate_echo_signal(speech, h, sr)
    
    # Save echo signal
    echo_wav = os.path.join(output_dir, f"{prefix}_echo.wav")
    sf.write(echo_wav, echo_signal, sr)
    print(f"  Saved echo signal: {echo_wav}")
    
    # Save reference (clean speech)
    reference_wav = os.path.join(output_dir, f"{prefix}_reference.wav")
    sf.write(reference_wav, speech, sr)
    print(f"  Saved reference: {reference_wav}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Generated Files Summary")
    print(f"{'='*60}")
    print(f"Echo path:  {echo_path_json}")
    print(f"Echo signal: {echo_wav}")
    print(f"Reference:   {reference_wav}")
    print(f"\nEcho Path Statistics:")
    print(f"  Max amplitude: {np.max(np.abs(h)):.6f}")
    print(f"  Energy: {np.sum(h**2):.6f}")
    print(f"  Center of mass: {np.sum(np.arange(len(h)) * h**2) / np.sum(h**2):.1f} samples")
    
    echo_rms = np.sqrt(np.mean(echo_signal**2))
    speech_rms = np.sqrt(np.mean(speech**2))
    print(f"\nSignal Statistics:")
    print(f"  Speech RMS: {speech_rms:.6f}")
    print(f"  Echo RMS: {echo_rms:.6f}")
    print(f"  Echo/Speech ratio: {20*np.log10(echo_rms/speech_rms):.2f} dB")
    
    return echo_path_json, echo_wav, reference_wav


def compare_echo_paths(
    estimated_path: np.ndarray,
    true_path: np.ndarray,
    params: Optional[RoomParameters] = None
) -> Dict[str, float]:
    """
    Compare estimated echo path with ground truth.
    
    Parameters
    ----------
    estimated_path : ndarray
        Estimated echo path from adaptive filter
    true_path : ndarray
        Ground truth echo path
    params : RoomParameters, optional
        Room parameters for delay info
        
    Returns
    -------
    dict
        Comparison metrics
    """
    if params is None:
        params = RoomParameters()
    
    # Ensure same length
    min_len = min(len(estimated_path), len(true_path))
    estimated_path = estimated_path[:min_len]
    true_path = true_path[:min_len]
    
    # Normalized Mean Square Error
    nmse = np.mean((estimated_path - true_path)**2) / np.mean(true_path**2)
    nmse_db = 10 * np.log10(nmse + 1e-10)
    
    # Correlation
    correlation = np.corrcoef(estimated_path, true_path)[0, 1]
    
    # Find peaks (echo path delay estimation accuracy)
    true_peak_idx = np.argmax(np.abs(true_path))
    est_peak_idx = np.argmax(np.abs(estimated_path))
    delay_error = abs(true_peak_idx - est_peak_idx)
    delay_error_ms = delay_error / params.sampling_rate * 1000
    
    # Amplitude accuracy at peak
    true_peak_amp = true_path[true_peak_idx]
    est_peak_amp = estimated_path[est_peak_idx]
    amplitude_error_db = 20 * np.log10(abs(est_peak_amp / true_peak_amp) + 1e-10)
    
    # Coherence (frequency domain similarity)
    H_true = np.fft.rfft(true_path)
    H_est = np.fft.rfft(estimated_path)
    coherence = np.abs(np.sum(H_est * np.conj(H_true))) / (
        np.sqrt(np.sum(np.abs(H_est)**2) * np.sum(np.abs(H_true)**2)) + 1e-10
    )
    
    metrics = {
        'nmse': nmse,
        'nmse_db': nmse_db,
        'correlation': correlation,
        'delay_error_samples': delay_error,
        'delay_error_ms': delay_error_ms,
        'amplitude_error_db': amplitude_error_db,
        'coherence': coherence,
        'passed': nmse_db < -20 and correlation > 0.9  # Pass criteria
    }
    
    return metrics


def print_comparison_report(
    metrics: Dict[str, float],
    true_path_file: str,
    estimated_path_file: Optional[str] = None
) -> None:
    """
    Print a formatted comparison report.
    
    Parameters
    ----------
    metrics : dict
        Comparison metrics from compare_echo_paths()
    true_path_file : str
        Path to ground truth echo path file
    estimated_path_file : str, optional
        Path to estimated echo path file
    """
    print("\n" + "=" * 60)
    print("Echo Path Comparison Report")
    print("=" * 60)
    print(f"Ground truth: {true_path_file}")
    if estimated_path_file:
        print(f"Estimated:    {estimated_path_file}")
    
    print(f"\n{'='*60}")
    print("Metrics")
    print(f"{'='*60}")
    print(f"  NMSE:              {metrics['nmse_db']:8.2f} dB")
    print(f"  Correlation:       {metrics['correlation']:8.6f}")
    print(f"  Coherence:         {metrics['coherence']:8.6f}")
    print(f"  Delay error:       {metrics['delay_error_samples']:8d} samples ({metrics['delay_error_ms']:.2f} ms)")
    print(f"  Amplitude error:   {metrics['amplitude_error_db']:8.2f} dB")
    
    print(f"\n{'='*60}")
    if metrics['passed']:
        print("✓ PASS: Echo path estimation meets criteria")
        print("  (NMSE < -20 dB and Correlation > 0.9)")
    else:
        print("✗ FAIL: Echo path estimation does not meet criteria")
        print("  Required: NMSE < -20 dB and Correlation > 0.9")
    print(f"{'='*60}\n")


# Harness integration
def run_echo_path_test(
    algorithm_name: str,
    estimated_echo_path: np.ndarray,
    true_echo_path_json: str,
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Run echo path comparison test as part of the harness.
    
    This function integrates echo path comparison into the test harness,
    allowing automated evaluation of echo path estimation algorithms.
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm being tested
    estimated_echo_path : ndarray
        Echo path estimated by the algorithm
    true_echo_path_json : str
        Path to ground truth echo path JSON file
    output_dir : str
        Directory for output files
        
    Returns
    -------
    dict
        Test results including metrics and pass/fail status
    """
    print(f"\n{'='*60}")
    print(f"Echo Path Test: {algorithm_name}")
    print(f"{'='*60}")
    
    # Load ground truth
    echo_path_data = load_echo_path_data(true_echo_path_json)
    true_path = echo_path_data.impulse_response
    
    # Compare
    metrics = compare_echo_paths(estimated_echo_path, true_path, echo_path_data.params)
    
    # Print report
    print_comparison_report(metrics, true_echo_path_json)
    
    # Save results
    results = {
        'algorithm': algorithm_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'passed': metrics['passed']
    }
    
    results_file = os.path.join(output_dir, f"echo_path_test_{algorithm_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved test results: {results_file}")
    
    return results


if __name__ == "__main__":
    # Example: Generate test files from original speech
    import os
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file
    input_wav = os.path.join(script_dir, "original_speech.wav")
    
    if os.path.exists(input_wav):
        # Generate test files
        echo_path_json, echo_wav, ref_wav = generate_test_files(
            input_wav_path=input_wav,
            output_dir=script_dir,
            params=RoomParameters(
                sampling_rate=16000,
                filter_length_ms=500,
                initial_delay_ms=45.0,
                reverberation_time_ms=300
            ),
            seed=42,
            prefix="ground_truth"
        )
        
        print(f"\n{'='*60}")
        print("Next Steps:")
        print(f"{'='*60}")
        print(f"1. Run your echo cancellation algorithm on:")
        print(f"   - Reference: {ref_wav}")
        print(f"   - Echo: {echo_wav}")
        print(f"2. Extract the estimated echo path from your algorithm")
        print(f"3. Compare using: compare_echo_paths(estimated, true)")
        print(f"{'='*60}")
    else:
        print(f"Input file not found: {input_wav}")
        print("Please ensure original_speech.wav exists in the directory.")



"""
Test execution engine for adaptive filters.

Provides common test runner functionality that processes signals
through adaptive filters and collects metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .interfaces import AdaptiveFilter


@dataclass
class TestConfig:
    """Configuration for test execution."""
    fft_size: int = 256
    overlap: float = 0.75
    transient_length: int = 4000  # Samples to skip for evaluation
    threshold: float = 1e-6       # Comparison threshold
    true_echo_path: Optional[np.ndarray] = None  # Ground truth for echo path metrics

    def __post_init__(self):
        self.step_size = int(self.fft_size * (1 - self.overlap))


@dataclass
class TestResult:
    """Results from running a test on an adaptive filter."""
    algorithm_name: str
    output: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Optional[TestConfig] = None
    echo_path_metrics: Optional[Dict[str, float]] = None
    
    def __str__(self) -> str:
        lines = [f"Test Result: {self.algorithm_name}"]
        lines.append(f"  Output length: {len(self.output)} samples")
        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


def run_test(
    algorithm: AdaptiveFilter,
    ref_signal: np.ndarray,
    mic_signal: np.ndarray,
    config: Optional[TestConfig] = None,
    algorithm_name: str = "Unknown"
) -> TestResult:
    """
    Run adaptive filter test on given signals.
    
    Parameters
    ----------
    algorithm : AdaptiveFilter
        Adaptive filter instance implementing the standard interface
    ref_signal : ndarray
        Reference signal (far-end, loudspeaker)
    mic_signal : ndarray
        Microphone signal (with echo)
    config : TestConfig, optional
        Test configuration. Uses defaults if not provided.
    algorithm_name : str, optional
        Human-readable name for the algorithm
        
    Returns
    -------
    TestResult
        Contains output signal and computed metrics
        
    Example
    -------
    >>> from harness import AdaptiveFilter, run_test
    >>> filter = MyAdaptiveFilter()
    >>> result = run_test(filter, ref, mic, algorithm_name="NLMS")
    >>> print(result.metrics['erle'])
    """
    if config is None:
        config = TestConfig()
    
    # Ensure signals are same length
    min_len = min(len(ref_signal), len(mic_signal))
    ref_signal = ref_signal[:min_len]
    mic_signal = mic_signal[:min_len]
    
    # Process signal
    output = _process_signal(algorithm, ref_signal, mic_signal, config)
    
    # Calculate metrics
    metrics = _calculate_metrics(
        output=output,
        mic_signal=mic_signal,
        ref_signal=ref_signal,
        config=config
    )

    # Calculate echo path metrics if ground truth is available
    echo_path_metrics = None
    true_echo_path = getattr(config, 'true_echo_path', None)
    if true_echo_path is not None and hasattr(algorithm, 'get_echo_path'):
        estimated_path = algorithm.get_echo_path()
        if estimated_path is not None:
            from ..metrics.comparison import calculate_echo_path_metrics
            echo_path_metrics = calculate_echo_path_metrics(estimated_path, true_echo_path)

    return TestResult(
        algorithm_name=algorithm_name,
        output=output,
        metrics=metrics,
        config=config,
        echo_path_metrics=echo_path_metrics
    )


def _process_signal(
    algorithm: AdaptiveFilter,
    ref_signal: np.ndarray,
    mic_signal: np.ndarray,
    config: TestConfig
) -> np.ndarray:
    """
    Process signal through adaptive filter using overlap-save method.
    
    Parameters
    ----------
    algorithm : AdaptiveFilter
        Adaptive filter instance
    ref_signal : ndarray
        Reference signal
    mic_signal : ndarray
        Microphone signal
    config : TestConfig
        Test configuration
        
    Returns
    -------
    ndarray
        Processed output signal
    """
    n_samples = len(ref_signal)
    output = np.zeros(n_samples)
    
    fft_size = config.fft_size
    step_size = config.step_size
    
    # Number of frames we can process
    n_frames = (n_samples - fft_size) // step_size + 1
    
    for i in range(n_frames):
        # Input window for this frame
        in_start = i * step_size
        in_end = in_start + fft_size
        
        x_block = ref_signal[in_start:in_end]
        d_block = mic_signal[in_start:in_end]
        
        # Filter and update
        e_block = algorithm.filt(x_block, d_block)
        algorithm.update(e_block)
        
        # Overlap-save: valid output is at the tail of the IFFT
        # Maps to the last step_size samples of the input window
        out_start = fft_size - step_size + i * step_size
        out_end = out_start + step_size
        
        if out_end <= n_samples:
            output[out_start:out_end] = e_block[-step_size:]
    
    return output


def _calculate_metrics(
    output: np.ndarray,
    mic_signal: np.ndarray,
    ref_signal: np.ndarray,
    config: TestConfig
) -> Dict[str, float]:
    """
    Calculate performance metrics for the test result.
    
    Parameters
    ----------
    output : ndarray
        Filter output (echo-cancelled signal)
    mic_signal : ndarray
        Original microphone signal
    ref_signal : ndarray
        Reference signal
    config : TestConfig
        Test configuration
        
    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    # Skip transient region for evaluation
    transient = config.transient_length
    output_eval = output[transient:]
    mic_eval = mic_signal[transient:]
    
    # Power calculations
    mic_power = np.sum(mic_eval ** 2)
    output_power = np.sum(output_eval ** 2)
    
    # ERLE (Echo Return Loss Enhancement)
    erle = 10 * np.log10((mic_power + 1e-10) / (output_power + 1e-10))
    
    # RMS values
    mic_rms = np.sqrt(np.mean(mic_eval ** 2))
    output_rms = np.sqrt(np.mean(output_eval ** 2))
    
    # Normalized output power (relative to mic)
    normalized_power = output_power / (mic_power + 1e-10)
    
    return {
        'erle': erle,
        'mic_rms': mic_rms,
        'output_rms': output_rms,
        'mic_power': mic_power,
        'output_power': output_power,
        'normalized_power': normalized_power,
    }


def run_comparison_test(
    algorithms: list,
    ref_signal: np.ndarray,
    mic_signal: np.ndarray,
    config: Optional[TestConfig] = None
) -> Dict[str, TestResult]:
    """
    Run multiple algorithms on the same signals for comparison.
    
    Parameters
    ----------
    algorithms : list
        List of (name, algorithm_instance) tuples
    ref_signal : ndarray
        Reference signal
    mic_signal : ndarray
        Microphone signal
    config : TestConfig, optional
        Test configuration
        
    Returns
    -------
    dict
        Dictionary mapping algorithm names to TestResult objects
    """
    results = {}
    
    for name, algo in algorithms:
        print(f"Running {name}...")
        results[name] = run_test(algo, ref_signal, mic_signal, config, name)
        print(f"  ERLE: {results[name].metrics['erle']:.2f} dB")
    
    return results

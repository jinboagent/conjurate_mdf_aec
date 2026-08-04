"""
Layer 4/5: Test Runner

Processes signals through adaptive filters, collects metrics,
and produces structured results.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from core.interfaces import AdaptiveFilter
from core.config import TestConfig
from metrics.signal_metrics import compute_erle, compute_output_rms
from metrics.comparison import compute_echo_path_metrics
from ground_truth.generators import generate_test_signals


@dataclass
class TestResult:
    """Results from a single test run."""

    algorithm_name: str
    erle_db: float = 0.0
    output_rms: float = 0.0
    echo_path_metrics: Optional[Dict[str, float]] = None
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Algorithm: {self.algorithm_name}",
            f"  ERLE:       {self.erle_db:.2f} dB",
            f"  Output RMS: {self.output_rms:.6f}",
        ]
        if self.echo_path_metrics:
            ep = self.echo_path_metrics
            lines.extend([
                f"  NMSE:       {ep.get('nmse_db', 'N/A'):.2f} dB" if isinstance(ep.get('nmse_db'), (int, float)) else f"  NMSE:       N/A",
                f"  Correlation:{ep.get('correlation', 'N/A'):.4f}" if isinstance(ep.get('correlation'), (int, float)) else f"  Correlation:N/A",
                f"  Coherence:  {ep.get('coherence', 'N/A'):.4f}" if isinstance(ep.get('coherence'), (int, float)) else f"  Coherence:  N/A",
                f"  Delay err:  {ep.get('delay_error_samples', 'N/A')} samples" if isinstance(ep.get('delay_error_samples'), (int, float)) else f"  Delay err:  N/A",
                f"  Amp error:  {ep.get('amplitude_error_db', 'N/A'):.2f} dB" if isinstance(ep.get('amplitude_error_db'), (int, float)) else f"  Amp error:  N/A",
            ])
        lines.append(f"  RESULT:     {'PASS' if self.passed else 'FAIL'}")
        return '\n'.join(lines)


class TestRunner:
    """
    Runs adaptive filter tests and collects metrics.

    Usage:
        config = TestConfig(block_size=256, sample_rate=16000)
        runner = TestRunner(config)
        result = runner.run(MyFilter(), true_echo_path)
    """

    def __init__(self, config: TestConfig):
        self.config = config

    def run(self,
            algorithm: AdaptiveFilter,
            true_echo_path: Optional[np.ndarray] = None,
            ref_signal: Optional[np.ndarray] = None,
            mic_signal: Optional[np.ndarray] = None,
            algorithm_name: str = "Unknown") -> TestResult:
        """
        Run a complete test.

        If ref_signal and mic_signal are not provided, they are generated
        from the true_echo_path using the ground truth generators.

        Parameters
        ----------
        algorithm : AdaptiveFilter
            The filter to test
        true_echo_path : ndarray, optional
            Ground truth echo path for comparison
        ref_signal : ndarray, optional
            Reference signal (generated if not provided)
        mic_signal : ndarray, optional
            Microphone signal (generated if not provided)
        algorithm_name : str
            Name for reporting

        Returns
        -------
        TestResult
        """
        cfg = self.config

        # Generate signals if not provided
        if ref_signal is None or mic_signal is None:
            if true_echo_path is None:
                raise ValueError("Must provide either (ref, mic) or true_echo_path")
            ref_signal, mic_signal = generate_test_signals(
                true_echo_path, cfg.signal_length, cfg.sample_rate
            )

        # Reset algorithm state
        algorithm.reset()

        # Process signal block by block
        M = cfg.block_size
        n_blocks = min(len(ref_signal), len(mic_signal)) // M
        e_output = np.zeros(n_blocks * M)

        for i in range(n_blocks):
            x = ref_signal[i * M:(i + 1) * M]
            d = mic_signal[i * M:(i + 1) * M]
            e = algorithm.filt(x, d)
            algorithm.update(e)
            e_output[i * M:(i + 1) * M] = e

        # Compute signal-level metrics
        transient = cfg.transient
        erle = compute_erle(mic_signal, e_output, transient)
        rms = compute_output_rms(e_output, transient)

        # Compute echo path metrics if ground truth available
        ep_metrics = None
        if true_echo_path is not None and hasattr(algorithm, 'get_echo_path'):
            est_path = algorithm.get_echo_path()
            if est_path is not None:
                ep_metrics = compute_echo_path_metrics(true_echo_path, est_path)

        # Determine pass/fail
        passed = erle >= cfg.erle_target_db
        if ep_metrics and cfg.correlation_target > 0:
            passed = passed and ep_metrics.get('correlation', 0) >= cfg.correlation_target

        return TestResult(
            algorithm_name=algorithm_name,
            erle_db=erle,
            output_rms=rms,
            echo_path_metrics=ep_metrics,
            passed=passed,
            details={
                'n_blocks': n_blocks,
                'signal_length': len(ref_signal),
                'transient': transient,
            }
        )

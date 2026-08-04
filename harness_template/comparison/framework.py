"""
Layer 4: Comparison Framework

Head-to-head algorithm comparison with identical inputs and parameters.
"""

import numpy as np
from typing import List, Tuple, Dict

from core.interfaces import AdaptiveFilter
from core.runner import TestRunner, TestResult
from core.config import TestConfig
from ground_truth.generators import generate_test_signals
from reporting.tables import format_comparison_table


def compare_algorithms(
    algorithms: List[Tuple[str, AdaptiveFilter]],
    true_path: np.ndarray,
    config: TestConfig,
    ref_signal: np.ndarray = None,
    mic_signal: np.ndarray = None,
    verbose: bool = True,
) -> Dict[str, TestResult]:
    """
    Run all algorithms with identical inputs and compare results.

    Parameters
    ----------
    algorithms : list of (name, filter) tuples
        Algorithms to compare
    true_path : ndarray
        Ground truth echo path
    config : TestConfig
        Test configuration
    ref_signal : ndarray, optional
        Reference signal (generated if not provided)
    mic_signal : ndarray, optional
        Microphone signal (generated if not provided)
    verbose : bool
        Print comparison table

    Returns
    -------
    dict mapping algorithm name → TestResult
    """
    # Generate signals once (shared by all algorithms)
    if ref_signal is None or mic_signal is None:
        ref_signal, mic_signal = generate_test_signals(
            true_path, config.signal_length, config.sample_rate
        )

    runner = TestRunner(config)
    results = {}

    for name, filt in algorithms:
        result = runner.run(
            algorithm=filt,
            true_echo_path=true_path,
            ref_signal=ref_signal,
            mic_signal=mic_signal,
            algorithm_name=name,
        )
        results[name] = result

    if verbose:
        print(format_comparison_table(results, config))

    return results


def run_progressive_tests(
    algorithm: AdaptiveFilter,
    config: TestConfig,
    test_levels: List[dict] = None,
) -> List[TestResult]:
    """
    Run tests at increasing difficulty levels.

    Default levels:
    1. Partition-aligned single delay (delay = k * M)
    2. Non-aligned single delay (delay = k * M + offset)
    3. Multi-tap echo path
    4. Realistic echo path

    Parameters
    ----------
    algorithm : AdaptiveFilter
    config : TestConfig
    test_levels : list of dict, optional
        Each dict has 'name', 'echo_path' keys

    Returns
    -------
    list of TestResult, one per level
    """
    if test_levels is None:
        M = config.block_size
        test_levels = [
            {
                'name': 'Level 1: Aligned delay',
                'echo_path': _make_simple_path(M, 0, config.echo_path_length),
            },
            {
                'name': 'Level 2: Non-aligned delay',
                'echo_path': _make_simple_path(M + M // 3, 1, config.echo_path_length),
            },
            {
                'name': 'Level 3: Multi-tap',
                'echo_path': _make_multi_tap(M, config.echo_path_length),
            },
            {
                'name': 'Level 4: Realistic',
                'echo_path': _make_realistic(config.echo_path_length),
            },
        ]

    results = []
    runner = TestRunner(config)

    for level in test_levels:
        algorithm.reset()
        result = runner.run(
            algorithm=algorithm,
            true_echo_path=level['echo_path'],
            algorithm_name=level['name'],
        )
        results.append(result)
        print(result.summary())
        print()

    return results


def _make_simple_path(delay, decay_idx, length):
    """Single delay echo path."""
    h = np.zeros(length)
    decay = [0.3, 0.5, 0.4][decay_idx % 3]
    if delay < length:
        h[delay] = decay
    return h


def _make_multi_tap(M, length):
    """Multi-tap echo path."""
    h = np.zeros(length)
    h[M] = 0.4
    h[M + M // 2] = 0.25
    h[2 * M] = 0.15
    h[3 * M + M // 3] = 0.1
    return h


def _make_realistic(length):
    """Realistic echo path."""
    from ground_truth.generators import generate_realistic_echo_path
    return generate_realistic_echo_path(
        filter_length_ms=length / 16.0  # Convert samples to ms at 16kHz
    )

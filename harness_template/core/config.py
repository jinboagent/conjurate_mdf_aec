"""
Test Configuration

Defines parameters, thresholds, and targets for test runs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class TestConfig:
    """Configuration for a single test run."""

    # Signal parameters
    block_size: int = 256
    sample_rate: int = 16000
    signal_length: int = 160000  # 10 seconds at 16kHz

    # Processing parameters
    transient: int = 4000  # Samples to skip for convergence
    fft_size: Optional[int] = None  # None = 2 * block_size

    # Echo path parameters
    echo_path_length: int = 8000  # 500ms at 16kHz
    true_echo_path: Optional[np.ndarray] = None

    # Metric thresholds (pass/fail)
    erle_target_db: float = 15.0
    nmse_target_db: float = -20.0
    correlation_target: float = 0.9
    coherence_target: float = 0.8
    delay_error_target: int = 0
    amplitude_error_target_db: float = 3.0

    def __post_init__(self):
        if self.fft_size is None:
            self.fft_size = 2 * self.block_size

    @property
    def n_blocks(self) -> int:
        """Number of blocks in the signal."""
        return self.signal_length // self.block_size

    @property
    def step_size(self) -> int:
        """Step size for overlap-save processing."""
        return self.block_size


@dataclass
class MetricThresholds:
    """Pass/fail thresholds for metrics."""

    erle_min_db: float = 15.0
    nmse_max_db: float = -20.0
    correlation_min: float = 0.9
    coherence_min: float = 0.8
    delay_error_max: int = 0
    amplitude_error_max_db: float = 3.0

    def check_erle(self, erle_db: float) -> bool:
        return erle_db >= self.erle_min_db

    def check_nmse(self, nmse_db: float) -> bool:
        return nmse_db <= self.nmse_max_db

    def check_correlation(self, corr: float) -> bool:
        return corr >= self.correlation_min

    def check_coherence(self, coh: float) -> bool:
        return coh >= self.coherence_min

    def check_delay_error(self, delay_err: int) -> bool:
        return delay_err <= self.delay_error_max

    def check_amplitude_error(self, amp_err_db: float) -> bool:
        return abs(amp_err_db) <= self.amplitude_error_max_db

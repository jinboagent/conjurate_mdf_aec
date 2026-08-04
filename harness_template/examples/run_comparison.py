"""
Example: Full Comparison

Compare multiple algorithms with identical inputs.
Generates a markdown comparison table.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.config import TestConfig
from comparison.framework import compare_algorithms, run_progressive_tests
from ground_truth.generators import generate_realistic_echo_path
from reporting.tables import format_comparison_table


# ============================================================
# Step 1: Define algorithms to compare
# ============================================================

# Import your algorithms here.
# For this example, we use a simple NLMS as placeholder.

from core.interfaces import AdaptiveFilter


class SimpleNLMS(AdaptiveFilter):
    """NLMS baseline."""
    def __init__(self, filter_length, mu=0.5):
        self.filter_length = filter_length
        self.mu = mu
        self.reset()

    def filt(self, x, d):
        assert len(x) == len(d)
        M = len(x)
        e = np.zeros(M)
        for n in range(M):
            self.x_hist = np.roll(self.x_hist, 1)
            self.x_hist[0] = x[n]
            y = np.dot(self.w, self.x_hist)
            e[n] = d[n] - y
            power = np.dot(self.x_hist, self.x_hist) + 1e-10
            self.w += self.mu * e[n] * self.x_hist / power
        return e

    def update(self, e):
        pass  # Update happens in filt()

    def get_echo_path(self):
        return self.w.copy()

    def reset(self):
        self.w = np.zeros(self.filter_length)
        self.x_hist = np.zeros(self.filter_length)


class AggressiveNLMS(SimpleNLMS):
    """NLMS with higher step size."""
    def __init__(self, filter_length):
        super().__init__(filter_length, mu=0.8)


# ============================================================
# Step 2: Configure
# ============================================================

config = TestConfig(
    block_size=256,
    sample_rate=16000,
    signal_length=160000,
    transient=8000,  # More time for convergence
    erle_target_db=15.0,
    correlation_target=0.9,
)

# ============================================================
# Step 3: Generate realistic ground truth
# ============================================================

true_path = generate_realistic_echo_path(
    sampling_rate=config.sample_rate,
    filter_length_ms=config.echo_path_length / config.sample_rate * 1000,
    initial_delay_ms=45.0,
    reverberation_time_ms=300.0,
    seed=42,
)

print(f"Ground truth echo path:")
print(f"  Length: {len(true_path)} samples ({len(true_path)/config.sample_rate*1000:.0f} ms)")
print(f"  Peak at: {np.argmax(np.abs(true_path))} samples ({np.argmax(np.abs(true_path))/config.sample_rate*1000:.1f} ms)")
print(f"  Peak value: {np.max(np.abs(true_path)):.4f}")
print()

# ============================================================
# Step 4: Run comparison
# ============================================================

algorithms = [
    ("Conservative NLMS", SimpleNLMS(len(true_path), mu=0.3)),
    ("Aggressive NLMS",   AggressiveNLMS(len(true_path))),
]

print("=" * 60)
print("COMPARISON: All algorithms with identical inputs")
print("=" * 60)
print()

results = compare_algorithms(
    algorithms=algorithms,
    true_path=true_path,
    config=config,
    verbose=True,
)

# ============================================================
# Step 5: Progressive difficulty test
# ============================================================

print()
print("=" * 60)
print("PROGRESSIVE TEST: Conservative NLMS at increasing difficulty")
print("=" * 60)
print()

prog_results = run_progressive_tests(
    SimpleNLMS(len(true_path), mu=0.3),
    config,
)

# ============================================================
# Step 6: Summary
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name, r in results.items():
    status = "PASS ✓" if r.passed else "FAIL ✗"
    print(f"  {name:25s} ERLE={r.erle_db:7.2f} dB  {status}")

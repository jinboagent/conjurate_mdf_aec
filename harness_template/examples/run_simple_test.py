"""
Example: Simple Test

Run a single algorithm against a trivial echo path.
This is the FIRST thing to do when developing a new algorithm.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.config import TestConfig
from core.runner import TestRunner
from ground_truth.generators import generate_simple_echo_path


# ============================================================
# Step 1: Create a dummy algorithm for demonstration
# ============================================================
# Replace this with your actual algorithm implementation.

from core.interfaces import AdaptiveFilter


class DummyNLMS(AdaptiveFilter):
    """Simple NLMS filter for demonstration."""

    def __init__(self, N, M, mu=0.5):
        self.N = N
        self.M = M
        self.mu = mu
        self.reset()

    def filt(self, x, d):
        assert len(x) == self.M
        # Simple FIR filter output
        y = np.dot(self.w, self.x_buf)
        e = d[0] - y if len(d) > 0 else -y
        # Shift buffer and insert new sample
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = x[0]
        return np.full(self.M, e)  # Block output (simplified)

    def update(self, e):
        x_power = np.dot(self.x_buf, self.x_buf) + 1e-10
        self.w += self.mu * e[0] * self.x_buf / x_power

    def get_echo_path(self):
        return self.w.copy()

    def reset(self):
        self.w = np.zeros(self.N * self.M)
        self.x_buf = np.zeros(self.N * self.M)


# ============================================================
# Step 2: Configure the test
# ============================================================

config = TestConfig(
    block_size=256,
    sample_rate=16000,
    signal_length=160000,  # 10 seconds
    transient=4000,
    erle_target_db=10.0,  # Lower target for simple demo
)

# ============================================================
# Step 3: Create ground truth
# ============================================================

# Start with the SIMPLEST possible echo path
M = config.block_size
true_path = generate_simple_echo_path(
    delay=M,     # Exactly at partition boundary
    decay=0.3,   # Moderate attenuation
    length=config.echo_path_length,
)

print(f"True echo path: single tap at sample {M}, decay=0.3")
print(f"Peak position: {np.argmax(np.abs(true_path))}")
print()

# ============================================================
# Step 4: Run the test
# ============================================================

algorithm = DummyNLMS(N=32, M=M, mu=0.3)
runner = TestRunner(config)
result = runner.run(algorithm, true_path, algorithm_name="DummyNLMS")

# ============================================================
# Step 5: Check results
# ============================================================

print(result.summary())
print()

if result.passed:
    print("✓ Algorithm passed the simple test!")
    print("  Now try with a non-aligned delay, then realistic echo path.")
else:
    print("✗ Algorithm failed even on simple input.")
    print("  Debug the fundamentals before trying complex cases.")
    print()
    print("  Debug funnel:")
    print("  1. Check: does filt() produce non-zero output?")
    print("  2. Check: does update() change the weights?")
    print("  3. Check: do weights converge toward the true path?")
    print("  4. Print intermediate values at each stage.")

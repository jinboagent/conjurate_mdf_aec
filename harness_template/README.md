# Harness Engineering Template

> A reusable project template for AI-assisted signal processing algorithm development.
> Extracted from the pdfaf_mdf acoustic echo cancellation project.

## What Is This?

This is a **starting point** for developing and testing adaptive filtering algorithms
with a structured test harness. Copy this directory into your project and adapt it.

The harness provides:
- **Standard interfaces** — every algorithm implements `filt()/update()` or `apply()`
- **Objective metrics** — ERLE, NMSE, correlation, coherence, delay error
- **Ground truth generation** — synthetic signals with known parameters
- **Fair comparison** — head-to-head testing with identical inputs
- **Reporting** — markdown tables, pass/fail, debug process logs

## Directory Structure

```
harness_template/
├── core/
│   ├── interfaces.py    # Base classes: AdaptiveFilter, FrequencyDomainFilter, BlockWrapper
│   ├── runner.py        # Test runner: processes signals, collects results
│   └── config.py        # Test configuration: parameters, thresholds, targets
├── metrics/
│   ├── signal_metrics.py  # ERLE, SNR, output RMS
│   └── comparison.py      # Echo path comparison: NMSE, correlation, coherence
├── ground_truth/
│   ├── generators.py      # Echo path generators, test signal creators
│   └── loaders.py         # Load/save WAV files, JSON ground truth
├── comparison/
│   └── framework.py       # Head-to-head algorithm comparison
├── reporting/
│   ├── tables.py          # Markdown table formatting
│   └── templates.py       # Debug report templates
├── examples/
│   ├── run_simple_test.py     # Minimal example: single algorithm, trivial echo
│   └── run_comparison.py     # Full comparison: multiple algorithms, realistic echo
└── README.md              # This file
```

## Quick Start

### 1. Copy the template

```bash
cp -r harness_template/ your_project/harness/
```

### 2. Implement your algorithm

```python
# your_algorithm.py
from harness.core.interfaces import AdaptiveFilter

class MyAlgorithm(AdaptiveFilter):
    def __init__(self, N, M, mu):
        self.N = N
        self.M = M
        self.mu = mu
        self.reset()

    def filt(self, x, d):
        # Your filtering logic here
        return e

    def update(self, e):
        # Your weight update here
        pass

    def get_echo_path(self):
        # Extract estimated impulse response
        return h_est

    def reset(self):
        # Initialize buffers and weights
        pass
```

### 3. Run a test

```python
from harness.core.runner import TestRunner
from harness.core.config import TestConfig
from harness.ground_truth.generators import generate_simple_echo_path
from your_algorithm import MyAlgorithm

# Configure
config = TestConfig(
    block_size=256,
    sample_rate=16000,
    signal_length=160000,
    transient=4000,
)

# Ground truth
true_path = generate_simple_echo_path(delay=720, decay=0.3, length=8000)

# Run
runner = TestRunner(config)
result = runner.run(MyAlgorithm(N=64, M=256, mu=0.05), true_path)

# Check
print(result.summary())
print("PASS" if result.passed else "FAIL")
```

### 4. Compare algorithms

```python
from harness.comparison.framework import compare_algorithms

results = compare_algorithms(
    algorithms=[
        ("MyAlgorithm", MyAlgorithm(N=64, M=256, mu=0.05)),
        ("Baseline",    BaselineFilter(N=64, M=256, mu=0.1)),
    ],
    true_path=true_path,
    config=config,
)
```

## The 5-Layer Architecture

```
Layer 5: REPORTING     → Markdown tables, pass/fail, debug logs
Layer 4: COMPARISON    → Head-to-head with identical inputs
Layer 3: GROUND TRUTH  → Known signals, known parameters
Layer 2: METRICS       → ERLE, NMSE, correlation, coherence
Layer 1: INTERFACE     → filt()/update() contract
```

Each layer depends only on layers below it. You can replace any layer
without touching the others.

## Metric Targets (AEC Defaults)

| Metric | Target | Meaning |
|--------|--------|---------|
| ERLE | ≥ 15 dB | Echo reduced by factor of ~30 |
| NMSE | < -20 dB | Estimated path within 1% energy of true |
| Correlation | ≥ 0.9 | Estimated path shape matches true |
| Coherence | ≥ 0.8 | Frequency-domain match |
| Delay error | 0 samples | Correct delay estimation |
| Amplitude error | < 3 dB | Correct amplitude estimation |

## Adapting to Your Domain

This template was built for AEC but works for any adaptive filtering problem:

- **Noise cancellation**: Change ground truth to noise+signal, metrics to SNR
- **Channel equalization**: Ground truth = known channel, metrics = BER/EVM
- **System identification**: Ground truth = known system, metrics = NMSE/coherence
- **Beamforming**: Multi-channel extension, add spatial metrics

## Debug Methodology

When your algorithm fails, follow the **funnel approach**:

```
1. REPRODUCE  → Run exact failing scenario, record all metrics
2. SIMPLIFY   → Replace complex input with trivial input
3. ISOLATE    → Check intermediate values at each stage
4. HYPOTHESIZE → Form a theory about the bug
5. VERIFY     → Test with minimal change
```

See `reporting/templates.py` for the debug report template.

## AI Prompt Template

When asking an AI to develop an algorithm using this harness:

```
I have a harness engineering project for adaptive filtering.
The harness is at: harness_template/

Task: Implement [algorithm name] following the AdaptiveFilter interface.

Requirements:
1. Implement filt(x, d) → e and update(e) methods
2. Test with trivial echo path first (single delay)
3. Verify intermediate values at each stage
4. Compare against PFDAF baseline
5. Document any bugs found and fixed

Ground truth: Use generators.py to create test signals
Metrics: Use signal_metrics.py for ERLE, comparison.py for echo path
Targets: ERLE ≥ 15 dB, correlation ≥ 0.9, delay error = 0

Start with examples/run_simple_test.py as your entry point.
```

# Echo Path Generation and Comparison

This document describes how to generate ground truth echo paths, create test signals, and compare estimated echo paths using the harness.

## Overview

The echo path generation system provides:

1. **Realistic echo path generation** - Simulates office room acoustics with configurable parameters
2. **Test signal creation** - Generates echo/microphone signals from clean speech
3. **Echo path comparison** - Compares estimated echo paths against ground truth
4. **Harness integration** - Automated testing as part of the AEC evaluation pipeline

## Quick Start

### 1. Generate Test Signals

```bash
python create_test_files.py
```

This creates:
- `reference.wav` - Clean reference speech (loudspeaker signal)
- `microphone.wav` - Microphone signal with echo

### 2. Run Your AEC Algorithm

Process the generated signals with your echo cancellation algorithm:

```python
import soundfile as sf
from FD_NLMS import FD_NLMS   # (doc predates renames: RLSBishengMDF -> FD_NLMS)

# Load signals
ref, sr = sf.read('reference.wav')
mic, _ = sf.read('microphone.wav')

# Run your AEC algorithm
# ... (algorithm-specific code)

# Extract estimated echo path from your algorithm
estimated_path = get_echo_path_from_your_algorithm()
```

### 3. Compare Echo Paths

```bash
python test_echo_path_comparison.py
```

## Configuration

### Echo Path Parameters

The test signal generator uses these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 16000 Hz | Sampling frequency |
| `delay_ms` | 50 ms | Echo delay time |
| `decay` | 0.5 | Echo attenuation factor |

## Echo Path Structure

The `create_test_files.py` script generates a simple echo path:

1. **Direct path** (50ms delay) - Attenuated copy of the reference signal
2. The echo path is a single delayed + attenuated tap, suitable for initial algorithm testing

## Comparison Metrics

The echo path comparison returns these metrics:

| Metric | Description | Pass Criteria |
|--------|-------------|---------------|
| `nmse_db` | Normalized Mean Square Error (dB) | < -20 dB |
| `correlation` | Correlation coefficient | > 0.9 |
| `coherence` | Frequency domain similarity | > 0.9 |
| `delay_error_ms` | Direct path delay error | < 5 ms |
| `amplitude_error_db` | Peak amplitude error | < 3 dB |

## Harness Integration

The echo path comparison is integrated with the test harness via `test_echo_path_comparison.py`. The harness computes correlation, coherence, NMSE, delay error, and amplitude error between estimated and true echo paths.

## Example Workflow

```python
import numpy as np
import soundfile as sf
from FD_NLMS import FD_NLMS   # (doc predates renames: RLSBishengMDF -> FD_NLMS)

# 1. Load test signals
ref, sr = sf.read('reference.wav')
mic, _ = sf.read('microphone.wav')

# 2. Run AEC algorithm
rls = RLSBishengMDF(NCHAN=1, NBIN=257, N_G=64, alpha=0.03, beta=0.97, bin_lim=257, Nrxref=1)

# Process in blocks
fft_size, step_size = 512, 128
for i in range(0, len(ref) - fft_size, step_size):
    X = np.fft.rfft(ref[i:i+fft_size]).reshape(-1, 1)
    D = np.fft.rfft(mic[i:i+fft_size]).reshape(-1, 1)
    rls.apply(D, X)

# 3. Extract estimated echo path
estimated_path = np.mean(np.abs(rls.w[0]), axis=(0, 2))
```

## Files

| File | Purpose |
|------|---------|
| `create_test_files.py` | Generate test audio with echo |
| `test_echo_path_comparison.py` | Echo path estimation verification |
| `reference.wav` | Reference/loudspeaker signal |
| `microphone.wav` | Microphone signal with echo |

## References

- Room impulse response modeling: Schroeder (1965)
- Echo path estimation: Sayed (2003) "Fundamentals of Adaptive Filtering"
- Conjugate Gradient MDF: Valin (2007) ICASSP; "Analysis of Conjugate Gradient Algorithms for Adaptive Filtering"

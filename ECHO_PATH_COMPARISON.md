# Echo Path Generation and Comparison

This document describes how to generate ground truth echo paths, create test signals, and compare estimated echo paths using the harness.

## Overview

The echo path generation system provides:

1. **Realistic echo path generation** - Simulates office room acoustics with configurable parameters
2. **Test signal creation** - Generates echo/microphone signals from clean speech
3. **Echo path comparison** - Compares estimated echo paths against ground truth
4. **Harness integration** - Automated testing as part of the AEC evaluation pipeline

## Quick Start

### 1. Generate Ground Truth Echo Path

```bash
cd frequency_domain_adaptive_filters
python echo_path_data.py
```

This creates:
- `ground_truth_echo_path.json` - Ground truth impulse response
- `ground_truth_echo.wav` - Echo/microphone signal
- `ground_truth_reference.wav` - Reference/clean speech signal

### 2. Run Your AEC Algorithm

Process the generated signals with your echo cancellation algorithm:

```python
import soundfile as sf
from rls_bisheng_mdf import RLSBishengMDF

# Load signals
ref, sr = sf.read('ground_truth_reference.wav')
echo, _ = sf.read('ground_truth_echo.wav')

# Run your AEC algorithm
# ... (algorithm-specific code)

# Extract estimated echo path from your algorithm
estimated_path = get_echo_path_from_your_algorithm()
```

### 3. Compare Echo Paths

```bash
python test_echo_path_comparison.py
```

Or programmatically:

```python
from echo_path_data import load_echo_path_data, compare_echo_paths

# Load ground truth
data = load_echo_path_data('ground_truth_echo_path.json')
true_path = data.impulse_response

# Compare
metrics = compare_echo_paths(estimated_path, true_path, data.params)

# Check results
if metrics['passed']:
    print("Echo path estimation meets criteria")
```

## Configuration

### Room Parameters

The echo path generator uses these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 16000 Hz | Sampling frequency |
| `filter_length_ms` | 500 ms | Echo path length |
| `initial_delay_ms` | 45.0 ms | Direct path delay (sound travel time) |
| `reverberation_time_ms` | 300 ms | RT60 equivalent (decay time) |

### Custom Echo Path Generation

```python
from echo_path_data import generate_test_files, RoomParameters

# Custom office room parameters
params = RoomParameters(
    sampling_rate=16000,
    filter_length_ms=500,
    initial_delay_ms=45.0,
    reverberation_time_ms=300
)

# Generate test files
echo_path_json, echo_wav, ref_wav = generate_test_files(
    input_wav_path='original_speech.wav',
    output_dir='.',
    params=params,
    seed=42,  # For reproducibility
    prefix='my_test'
)
```

## Echo Path Structure

The generated echo path models:

1. **Direct path** (45ms delay) - Sound traveling from loudspeaker to microphone
2. **Early reflections** (50-100ms) - Discrete echoes from walls, desk, ceiling
3. **Late reverberation** (100-500ms) - Dense exponential decay tail

```
Amplitude
    ^
    |     Direct    Early        Late
    |      Path    Reflections  Reverberation
    |        |\      | | |      ~~~~~~~~~~
    |        | \     | | |     /
    |        |  \    | | |    /
    |________|___\___|_|_|___/___________________> Time
    0       45ms   100ms     500ms
```

## Comparison Metrics

The `compare_echo_paths()` function returns these metrics:

| Metric | Description | Pass Criteria |
|--------|-------------|---------------|
| `nmse_db` | Normalized Mean Square Error (dB) | < -20 dB |
| `correlation` | Correlation coefficient | > 0.9 |
| `coherence` | Frequency domain similarity | > 0.9 |
| `delay_error_ms` | Direct path delay error | < 5 ms |
| `amplitude_error_db` | Peak amplitude error | < 3 dB |

## Harness Integration

The echo path comparison is integrated with the test harness:

```python
from harness import run_echo_path_test

# Run test and get results
results = run_echo_path_test(
    algorithm_name='RLS_Bisheng_MDF',
    estimated_echo_path=estimated_path,
    true_echo_path_json='ground_truth_echo_path.json',
    output_dir='.'
)

# Check pass/fail
if results['passed']:
    print("Test PASSED")
else:
    print("Test FAILED")
    print(f"  NMSE: {results['metrics']['nmse_db']:.2f} dB")
    print(f"  Correlation: {results['metrics']['correlation']:.4f}")
```

## Example Workflow

```python
import numpy as np
import soundfile as sf
from echo_path_data import load_echo_path_data, compare_echo_paths
from rls_bisheng_mdf import RLSBishengMDF

# 1. Load ground truth
data = load_echo_path_data('ground_truth_echo_path.json')
true_path = data.impulse_response

# 2. Load test signals
ref, sr = sf.read('ground_truth_reference.wav')
echo, _ = sf.read('ground_truth_echo.wav')

# 3. Run AEC algorithm
rls = RLSBishengMDF(NCHAN=1, NBIN=257, N_G=64, alpha=0.03, beta=0.97, bin_lim=257, Nrxref=1)

# Process in blocks
fft_size, step_size = 512, 128
for i in range(0, len(ref) - fft_size, step_size):
    X = np.fft.rfft(ref[i:i+fft_size]).reshape(-1, 1)
    D = np.fft.rfft(echo[i:i+fft_size]).reshape(-1, 1)
    rls.apply(D, X)

# 4. Extract estimated echo path
estimated_path = np.mean(np.abs(rls.w[0]), axis=(0, 2))
estimated_path = np.pad(estimated_path, (0, 8000 - len(estimated_path)))
estimated_path = estimated_path / (np.max(estimated_path) + 1e-10) * 0.6

# 5. Compare
metrics = compare_echo_paths(estimated_path, true_path, data.params)
print(f"NMSE: {metrics['nmse_db']:.2f} dB")
print(f"Correlation: {metrics['correlation']:.4f}")
print(f"Passed: {metrics['passed']}")
```

## Files

| File | Purpose |
|------|---------|
| `echo_path_generator.py` | Echo path generation algorithms |
| `echo_path_data.py` | Data generation and comparison utilities |
| `echo_path_examples.py` | Usage examples for different scenarios |
| `test_echo_path_comparison.py` | Automated test suite |
| `ground_truth_echo_path.json` | Ground truth echo path data |
| `ground_truth_echo.wav` | Generated echo signal |
| `ground_truth_reference.wav` | Reference speech signal |

## References

- Room impulse response modeling: Schroeder (1965)
- Echo path estimation: Sayed (2003) "Fundamentals of Adaptive Filtering"
- RLS Bisheng MDF: Valin (2007) ICASSP

# Harness Engineering — AI-Assisted Algorithm Development Framework

## What is Harness Engineering?

Harness Engineering is a structured methodology for developing, testing, and debugging signal processing algorithms with AI assistance. It was extracted from the patterns used in this project (pdfaf_mdf) and can be applied to any adaptive filtering, audio processing, or DSP algorithm development project.

The core idea: **build a test harness around your algorithm first, then iterate on the algorithm inside the harness.** The harness provides objective metrics, reproducible tests, and automated comparison — so the AI (and you) always know if a change helped or hurt.

---

## The 5-Layer Harness Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5: REPORTING                                                  │
│  Markdown reports, pass/fail tables, plots, process documentation    │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: COMPARISON                                                 │
│  Head-to-head algorithm comparison with identical inputs/params      │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: GROUND TRUTH                                               │
│  Known signals, known parameters, known expected outputs             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: METRICS                                                    │
│  ERLE, NMSE, correlation, coherence, delay error, amplitude error    │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: ALGORITHM INTERFACE                                        │
│  Standardized filt()/update() or apply() contract                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Algorithm Interface

Every algorithm must implement a standard interface so the harness can test them interchangeably.

### Option A: Block-based interface (for time-domain processing)

```python
class AdaptiveFilter:
    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        x: reference signal block (far-end, loudspeaker)
        d: desired signal block (microphone with echo)
        returns: error signal (echo-cancelled output)
        """
        pass

    def update(self, e: np.ndarray) -> None:
        """Update filter coefficients based on error signal."""
        pass

    def get_echo_path(self) -> Optional[np.ndarray]:
        """Return estimated echo path impulse response (optional)."""
        return None

    def reset(self) -> None:
        """Reset filter state to initial conditions."""
        pass
```

### Option B: Frequency-domain interface (for frame-based processing)

```python
class FrequencyDomainFilter:
    def apply(self, Y: np.ndarray, Y_rx: np.ndarray) -> np.ndarray:
        """
        Y: microphone input [NBIN, NCHAN] in frequency domain
        Y_rx: reference signal [NBIN, Nrxref] in frequency domain
        returns: echo cancelled output [NBIN, NCHAN]
        """
        pass
```

### Wrapper pattern (bridging time-domain and frequency-domain)

```python
class BlockWrapper:
    """Wraps a frequency-domain filter with time-domain overlap-save."""
    def __init__(self, filter, fft_size, overlap):
        self.filter = filter
        self.fft_size = fft_size
        self.step_size = int(fft_size * (1 - overlap))
        self.x_old = np.zeros(self.step_size)

    def process_block(self, x, d):
        # Overlap-save: concatenate old + new
        x_now = np.concatenate([self.x_old, x])
        X = np.fft.rfft(x_now)
        D = np.fft.rfft(d)

        # Apply frequency-domain filter
        E = self.filter.apply(D.reshape(-1,1), X.reshape(-1,1))

        # Extract valid output
        e_time = np.fft.irfft(E[:, 0], n=self.fft_size)
        self.x_old = x[-self.step_size:]
        return e_time[-self.step_size:]
```

---

## Layer 2: Metrics

Standardized metrics computed identically for all algorithms.

```python
def compute_metrics(e_output, mic_signal, true_echo_path, est_echo_path, transient=4000):
    """Compute all evaluation metrics."""

    # Signal-level metrics
    mic_eval = mic_signal[transient:]
    out_eval = e_output[transient:]
    mic_power = np.sum(mic_eval ** 2)
    out_power = np.sum(out_eval ** 2)

    erle = 10 * np.log10((mic_power + 1e-10) / (out_power + 1e-10))

    # Echo path metrics (if ground truth available)
    ep_metrics = None
    if true_echo_path is not None and est_echo_path is not None:
        min_len = min(len(true_echo_path), len(est_echo_path))
        true = true_echo_path[:min_len]
        est = est_echo_path[:min_len]

        nmse = 10 * np.log10(np.sum((est - true)**2) / (np.sum(true**2) + 1e-10) + 1e-10)
        corr = float(np.corrcoef(est, true)[0, 1])

        H_true = np.fft.rfft(true)
        H_est = np.fft.rfft(est)
        coherence = float(np.abs(np.sum(H_est * np.conj(H_true))) /
                         (np.sqrt(np.sum(np.abs(H_est)**2) * np.sum(np.abs(H_true)**2)) + 1e-10))

        true_peak = int(np.argmax(np.abs(true)))
        est_peak = int(np.argmax(np.abs(est)))
        delay_error = abs(true_peak - est_peak)

        amp_error = 20 * np.log10(np.abs(true[true_peak] / (est[est_peak] + 1e-10)) + 1e-10)

        ep_metrics = {
            'nmse_db': nmse,
            'correlation': corr,
            'coherence': coherence,
            'delay_error_samples': delay_error,
            'amplitude_error_db': amp_error,
        }

    return {
        'erle_db': erle,
        'output_rms': np.sqrt(np.mean(out_eval**2)),
        'echo_path': ep_metrics,
    }
```

### Metric Targets

| Metric | Target | Meaning |
|---|---|---|
| ERLE | ≥ 15 dB | Echo reduced by factor of ~30 |
| NMSE | < -20 dB | Estimated path within 1% energy of true |
| Correlation | ≥ 0.9 | Estimated path shape matches true |
| Coherence | ≥ 0.8 | Frequency-domain match |
| Delay error | 0 samples | Correct delay estimation |
| Amplitude error | < 3 dB | Correct amplitude estimation |

---

## Layer 3: Ground Truth

Generate known signals with known parameters so you can verify the algorithm finds the right answer.

### Pattern: Synthetic echo path + convolution

```python
def generate_ground_truth(speech_file, seed=42):
    """Generate ground truth echo path and microphone signal."""

    # Load clean speech
    ref, sr = sf.read(speech_file)

    # Generate echo path with known parameters
    params = RoomParameters(
        sampling_rate=sr,
        filter_length_ms=500,      # 8000 samples
        initial_delay_ms=45.0,     # 720 samples
        reverberation_time_ms=300  # RT60
    )
    true_path = generate_time_domain_echo_path(params, seed=seed)

    # Create microphone = conv(reference, echo_path)
    mic = fftconvolve(ref, true_path, mode='full')[:len(ref)]

    return ref, mic, true_path, params
```

### Pattern: Simple test cases for debugging

```python
def simple_echo_path(delay, decay, length=8000):
    """Trivial echo path: single delay + attenuation."""
    h = np.zeros(length)
    h[delay] = decay
    return h

# Test cases ordered by difficulty:
# 1. Partition-aligned delay (delay = k*M)     → should work perfectly
# 2. Non-aligned delay (delay = k*M + offset)   → tests sub-partition resolution
# 3. Multi-tap echo path                        → tests multi-partition convergence
# 4. Realistic echo path (generator)            → tests real-world performance
```

---

## Layer 4: Comparison Framework

### Fair comparison test pattern

```python
def compare_algorithms(ref, mic, true_path, algorithms, params):
    """
    Run all algorithms with identical inputs and parameters.

    algorithms: list of (name, filter_instance) tuples
    """
    results = {}

    for name, filt in algorithms:
        # Reset filter state
        filt.reset()

        # Process signal
        e_output = np.zeros(len(ref))
        M = params['block_size']

        for i in range(len(ref) // M):
            x = ref[i*M:(i+1)*M]
            d = mic[i*M:(i+1)*M]
            e = filt.filt(x, d)
            filt.update(e)
            e_output[i*M:(i+1)*M] = e

        # Extract echo path
        est_path = filt.get_echo_path() if hasattr(filt, 'get_echo_path') else None

        # Compute metrics
        metrics = compute_metrics(e_output, mic, true_path, est_path)
        results[name] = metrics

    # Print comparison table
    print_comparison_table(results)
    return results
```

### Key principle: identical configuration

```python
# Both algorithms must use:
# - Same FFT size (M-point or 2M-point — must match!)
# - Same number of partitions (N_G)
# - Same learning rate (mu/alpha)
# - Same forgetting factor (beta)
# - Same input data (ref, mic)
# - Same block processing (step size, overlap)
```

---

## Layer 5: Reporting

### Process documentation pattern

After every debug session, write a markdown report:

```markdown
# Debug Report: [Issue Title]

## Problem
- What failed? (metric, error message, unexpected behavior)
- What was expected?

## Investigation
- Step 1: Simplest possible test case
- Step 2: Isolate the failing component
- Step 3: Check intermediate values

## Root Cause
- What was the actual bug?
- Why did it happen?

## Fix
- What was changed?
- Why does this fix work?

## Verification
- Test results before and after
- Pass/fail for each metric
```

### Debug methodology (the funnel approach)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: REPRODUCE                                                   │
│  Run the exact failing scenario. Confirm the failure is consistent.  │
│  Record all metrics.                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Step 2: SIMPLIFY                                                    │
│  Replace complex input with trivial input:                           │
│    Complex echo path → single delay (h = 0.3 * δ[n-256])           │
│    Real speech → white noise                                         │
│    N_G=64 → N_G=4                                                    │
│  If it works with simple input, the bug is in complexity handling.  │
│  If it fails with simple input, the bug is fundamental.             │
├─────────────────────────────────────────────────────────────────────┤
│  Step 3: ISOLATE                                                     │
│  Check intermediate values at each stage:                            │
│    - Buffer contents (are frames stored correctly?)                  │
│    - FFT output (correct frequency bins?)                            │
│    - Correlation values (Rtoe, rcross match expected?)               │
│    - Weight values (converging to known answer?)                     │
│    - Filter output (echo estimate correct?)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Step 4: HYPOTHESIZE                                                 │
│  Form a theory about the bug. Common categories:                     │
│    - Convention mismatch (FFT size, buffer ordering, modulation)     │
│    - Numerical issue (division by zero, overflow, conditioning)      │
│    - Model limitation (can't represent the true system)              │
│    - Parameter issue (step size too large/small, wrong beta)         │
├─────────────────────────────────────────────────────────────────────┤
│  Step 5: VERIFY                                                      │
│  Test the hypothesis with a minimal change.                          │
│  If the fix works → document and move on.                            │
│  If not → back to Step 3 with new information.                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Applying This to a New Algorithm Project

### Step-by-step guide

1. **Define the algorithm interface** (Layer 1)
   - What inputs does the algorithm take?
   - What outputs does it produce?
   - What internal state does it maintain?

2. **Create ground truth** (Layer 3)
   - Generate synthetic signals with known parameters
   - Start with trivial cases, then realistic cases
   - Save ground truth to disk for reproducibility

3. **Implement metrics** (Layer 2)
   - Define what "correct" means quantitatively
   - Set pass/fail thresholds
   - Create comparison tables

4. **Build comparison tests** (Layer 4)
   - Compare against a known-working baseline
   - Use identical inputs and parameters
   - Test at multiple difficulty levels

5. **Document everything** (Layer 5)
   - Write debug reports for every issue
   - Maintain an architecture document
   - Track design decisions and their rationale

### Checklist for AI-assisted development

```
Before implementing:
  □ Algorithm interface defined (inputs, outputs, state)
  □ Ground truth generation working
  □ Metrics defined with pass/fail thresholds
  □ At least one reference implementation to compare against

During implementation:
  □ Test with trivial input first (single delay, white noise)
  □ Verify intermediate values at each stage
  □ Compare against reference at every step
  □ Document every bug found and fixed

After implementation:
  □ Fair comparison test with identical parameters
  □ All metrics pass thresholds
  □ Debug report written
  □ Architecture document updated
```

---

## Anti-Patterns to Avoid

| Anti-Pattern | What Happens | Do This Instead |
|---|---|---|
| Testing only on real data | Can't tell if the answer is correct | Start with synthetic ground truth |
| Changing multiple things at once | Can't isolate what helped | Change one thing, test, repeat |
| Using different FFT sizes | Algorithms aren't comparable | Document and match FFT conventions |
| Ignoring intermediate values | Bug could be anywhere | Print/check at every stage |
| No metrics | "It looks better" isn't science | Define quantitative targets |
| Skipping simple tests | Complex failures are hard to debug | Always start with trivial input |
| Not documenting bugs | Same bug comes back | Write it down immediately |

---

## File Templates

### New algorithm template

```python
class MyAlgorithm:
    """
    Description of the algorithm.

    Parameters
    ----------
    N : int — number of partitions
    M : int — partition size (block size)
    mu : float — learning rate
    beta : float — forgetting factor
    """

    def __init__(self, N, M, mu, beta=0.97):
        self.N = N
        self.M = M
        self.mu = mu
        self.beta = beta
        self.reset()

    def filt(self, x, d):
        """Process one block. Returns error signal."""
        assert len(x) == self.M
        # ... filtering logic ...
        return e

    def update(self, e):
        """Update weights based on error."""
        # ... adaptation logic ...
        pass

    def get_echo_path(self):
        """Extract estimated impulse response."""
        # ... weight-to-impulse-response conversion ...
        return h_est

    def reset(self):
        """Reset all internal state."""
        # ... initialize buffers, weights ...
        pass
```

### New test template

```python
"""
Test: [Algorithm Name] — [What is being tested]
"""
import numpy as np
from echo_path_generator import generate_time_domain_echo_path
from my_algorithm import MyAlgorithm
from scipy.signal import fftconvolve

# Ground truth
ref = np.random.randn(160000)  # or load WAV
true_path = generate_time_domain_echo_path(seed=42)
mic = fftconvolve(ref, true_path, mode='full')[:len(ref)]

# Algorithm
filt = MyAlgorithm(N=64, M=256, mu=0.05)

# Process
e_out = np.zeros(len(ref))
for i in range(len(ref) // 256):
    e = filt.filt(ref[i*256:(i+1)*256], mic[i*256:(i+1)*256])
    filt.update(e)
    e_out[i*256:(i+1)*256] = e

# Evaluate
transient = 4 * len(true_path)
erle = 10 * np.log10((np.sum(mic[transient:]**2) + 1e-10) /
                      (np.sum(e_out[transient:]**2) + 1e-10))
print(f"ERLE: {erle:.2f} dB")
print(f"PASS: {erle > 15}")
```

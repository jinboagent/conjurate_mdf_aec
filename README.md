# pdfaf_mdf — Partitioned-Block Frequency-Domain Adaptive Filters for Acoustic Echo Cancellation

A collection of frequency-domain adaptive filtering algorithms for **Acoustic Echo Cancellation (AEC)**, implemented in pure Python/NumPy. Includes a reusable test harness with two-level verification (signal + model) for fair algorithm comparison.

## Algorithms

### Frequency-Domain Adaptive Filters

| Algorithm | File | Adaptation | Use Case |
|-----------|------|------------|----------|
| **PFADF MDF CG** | `pfadf_mdf_cg.py` | Normalized gradient | Stable + sub-partition delay resolution |
| **Conjugate Gradient MDF** | `conjugate_mdf.py` | Conjugate Gradient (Toeplitz) | Toeplitz-matrix CG solve |
| **PFDAF-CG** | `pfdaf_cg.py` | Conjugate Gradient | Fast convergence |

## Architecture

All algorithms use **2M-point FFTs with overlap-save** for sub-partition delay resolution.

```
Reference (x) ──→ [Loudspeaker] ──→ [Room h] ──→ [Mic] ──→ d
       │                                                    │
       └──→ [Adaptive Filter H] ──→ echo_est (y) ──→ (-) ←─┘
                                          │
                                      error (e) ──→ output
                                          │
                                          └──→ [Weight Update]
```

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for full architecture details, and [DATAFLOW.md](DATAFLOW.md) for signal flow diagrams.

## Benchmark Results

### PFADF MDF CG (N=64, M=256, real speech, 8000-sample echo path)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| ERLE | 7.18 dB | >15 dB | ⚠ Limited by signal length |
| Correlation | **0.978** | >0.9 | ✓ Pass |
| Coherence | **0.978** | — | Excellent |
| Delay error | **0 samples** | — | ✓ Perfect |
| Amplitude error | **-0.68 dB** | <3 dB | ✓ Pass |
| NMSE | -13.66 dB | <-20 dB | ⚠ Needs more blocks |

### Identity Test (white noise, d=x, mu=0.5, 1000 blocks)

| Metric | Value |
|--------|-------|
| ERLE | **23.77 dB** |
| Signal-proportional ε ERLE | **~70 dB** (ε = mean(e²)/mean(x²)) |

### Fair Comparison: Conjugate Gradient MDF vs PFADF MDF CG (both 2M-point FFT)

| Metric | Conjugate Gradient MDF | PFADF MDF CG |
|--------|----------------|--------------|
| ERLE | **3.21 dB** | 0.59 dB |
| Delay error | 16 ms | **0 ms** |
| Correlation | 0.40 | **0.92** |

### Stability

| Parameter | Safe Range | Divergence |
|-----------|------------|------------|
| mu (N=64) | ≤ 2.0 | ≥ 3.0 |
| beta (forgetting) | 0.97 | — |
| Input power init (Rtoa) | 2×M | Near-zero → NaN |

## Requirements

- Python 3.6+
- numpy
- scipy
- librosa (for test audio generation)

## Usage

```python
from pfadf_mdf_cg import pfadf_mdf_cg
import numpy as np
import soundfile as sf

# Load signals
ref, sr = sf.read('reference.wav')   # Far-end (loudspeaker)
mic, _  = sf.read('microphone.wav')  # Near-end (mic with echo)

# Echo cancellation
output = pfadf_mdf_cg(ref, mic, N=64, M=256, mu=0.03, beta=0.97)

sf.write('output.wav', output, sr)
```

## Test Harness

The project includes a structured test harness with **two-level verification**:

| Layer | What it measures | Metrics |
|-------|-----------------|---------|
| **Signal-level** | Did the filter cancel echo? | ERLE (dB) |
| **Model-level** | Did the filter learn the true echo path? | NMSE, correlation, coherence, delay error, amplitude error |

### Harness Structure

```
harness/
├── core/
│   ├── interfaces.py    # AdaptiveFilter ABC (filt/update/get_echo_path)
│   ├── runner.py        # run_test(), TestConfig, TestResult
│   └── report.py        # TestReport, ComparisonReport
├── metrics/
│   └── comparison.py    # ERLE, NMSE, correlation, coherence, echo path metrics
└── adapters/
    └── __init__.py      # Algorithm adapters for harness interface
```

See [HARNESS_ENGINEERING.md](HARNESS_ENGINEERING.md) for the full methodology.

## Documentation

- [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) — Algorithm family tree, design decisions, common pitfalls
- [DATAFLOW.md](DATAFLOW.md) — Visual signal flow diagrams, buffer structures, algorithm internals
- [RLS_DEBUG_PROCESS.md](docs/RLS_DEBUG_PROCESS.md) — Complete Conjugate Gradient MDF debug journey (5 phases, 10+ bugs fixed)
- [REGULARIZATION_RESEARCH.md](REGULARIZATION_RESEARCH.md) — Deep dive on regularization factor ε (theory, literature, experiments)
- [ECHO_PATH_COMPARISON.md](ECHO_PATH_COMPARISON.md) — Echo path comparison methodology and metrics
- [HARNESS_ENGINEERING.md](HARNESS_ENGINEERING.md) — 5-layer harness framework for AI-assisted algorithm development

### Test Scripts

| Script | Purpose |
|--------|---------|
| `test_subband_echo_cancellation.py` | End-to-end NLMS vs CG-MDF echo cancellation |
| `test_echo_path_comparison.py` | Echo path estimation verification |
| `visualize_weight_convergence.py` | Weight convergence visualization suite |
| `create_test_files.py` | Generate test audio from LibriSpeech samples |

## Key Design Decisions

1. **2M-point FFT (not M-point)**: Required for sub-partition delay resolution. M-point FFTs can only model delays at exact multiples of M.
2. **Normalized gradient over Toeplitz CG**: Theoretically inferior but practically stable. Toeplitz-based CG creates ill-conditioned systems that single CG steps can't solve.
3. **Signal-proportional regularization**: ε = mean(e²)/mean(x²) outperforms fixed ε by 20-40 dB on identity tests.
4. **Two-level verification**: Signal-level ERLE and model-level echo path metrics catch different classes of bugs.

## References

- Valin, J.-M. (2007). "On Adjusting the Learning Rate in Frequency Domain Echo Cancellation With Double-Talk." ICASSP 2007.
- Sayed, A.H. (2003). "Fundamentals of Adaptive Filtering." Wiley.
- García, L. et al. (2006). "The Conjugate Gradient Partitioned Block Frequency-Domain Adaptive Filter for Multichannel Acoustic Echo Cancellation." EUSIPCO 2006.
- Lee, K.-A., Gan, W.-S., Kuo, S.M. "Subband Adaptive Filtering: Theory and Implementation."
- Haykin, S. "Adaptive Filter Theory."

## Author

Jinboli

## License

MIT — see [LICENSE](LICENSE) for details.

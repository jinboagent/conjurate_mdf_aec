# pdfaf_mdf — Partitioned-Block Frequency-Domain Adaptive Filters for Acoustic Echo Cancellation

A collection of frequency-domain adaptive filtering algorithms for **Acoustic Echo Cancellation (AEC)**, implemented in pure Python/NumPy. Includes a reusable test harness with two-level verification (signal + model) for fair algorithm comparison.

## Algorithms

### Frequency-Domain Adaptive Filters

| Algorithm | File | Adaptation | Use Case |
|-----------|------|------------|----------|
| **PFADF MDF CG** | `pfadf_mdf_cg.py` | Normalized gradient | Stable + sub-partition delay resolution |
| **CONJUGATE_MDF (canonical `[0;e]` MDF)** | `conjugate_mdf.py` | Normalized gradient + β-averaging, excitation gate, G-projection | The reference implementation — cliff-free at any delay |
| **FD_NLMS** | `conjugate_mdf.py` (same module) | Instantaneous normalized gradient | The classic FDAF baseline (≡ CONJUGATE_MDF at β=0) |
| **PBFDAF-CG (AES 2006)** | `pfdaf_cg.py` | Conjugate-gradient direction on the memory-averaged gradient | Fastest convergence on longer signals |

All four share the same overlap-save frame contract: full-frame FFTs in,
zero-headed `[0;e]` a-priori error out — the `[0;e]` criterion (grade only
the new block) is what removes the delay-mod-M "criterion cliff"
(see [DATAFLOW.md §7](docs/DATAFLOW.md)).

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

### Official suite (test_subband_echo_cancellation.py, canonical 50 ms pair, FFT 512 / hop 128, N_G=8)

| key | algorithm | ERLE | Correlation | Status |
|-----|-----------|------|-------------|--------|
| `nlms` | time-domain NLMS | 22.27 dB | 0.997 | PASS |
| `fdnlms` | FD_NLMS (`[0;e]` + G) | 23.77 dB | 0.998 | PASS |
| `cgmdf` | CONJUGATE_MDF (β=0 ≡ fdnlms) | 23.77 dB | 0.998 | PASS |
| `pfcg` | PBFDAF-CG (AES 2006, γ=0.4) | **26.11 dB** | **0.999** | PASS |

### PBFDAF-CG vs FD_NLMS across pairs (γ=0.4, Hestenes-Stiefel)

| pair | PBFDAF-CG | FD_NLMS | Δ |
|------|-----------|---------|---|
| canonical (5 s) | 26.11 dB | 23.77 dB | +2.3 |
| noise + lowpass (5 s) | 48.06 dB | 43.20 dB | +4.9 |
| speech + lowpass (30 s) | 31.44 dB | 21.67 dB | **+9.8** |
| canonical tiled ×4 (20 s) | 27.63 dB | 17.69 dB | **+9.9** |

The CG advantage **grows with signal length** — the averaged-gradient +
conjugate-direction machinery keeps adapting where plain NLMS stalls
(the paper's faster-convergence claim, confirmed). β-method ranking:
Hestenes-Stiefel ≥ Dai-Yuan > Polak-Ribière ≫ Fletcher-Reeves (FR
diverges on speech — norm-only β cannot sense the changing quadratic).
Details: `results/2026-10-05_pfdaf-cg/FINDINGS.md`.

### Known limits

- **Dense reverb**: coverage first (N_G must exceed the direct delay),
  then per-bin FD convergence is slow — full-tap time-domain NLMS is
  the workhorse there (`results/2026-10-01_hop-ablation/`).
- **CG iterations**: k_max > 1 diverges on the per-bin model; keep 1.

### Historical (early development, superseded)

<details>
<summary>PFADF MDF CG first runs, identity test, early fair comparison</summary>

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| ERLE | 7.18 dB | >15 dB | ⚠ Limited by signal length |
| Correlation | **0.978** | >0.9 | ✓ Pass |
| Delay error | **0 samples** | — | ✓ Perfect |
| NMSE | -13.66 dB | <-20 dB | ⚠ Needs more blocks |

Identity test (white noise, d=x, mu=0.5): ERLE 23.77 dB, signal-proportional
ε ERLE ~70 dB. Early fair comparison (CG-MDF 3.21 dB vs PFADF 0.59 dB) —
both since superseded by the canonical rewrite.

</details>

## Requirements

- Python 3.12 (project venv: `.venv`)
- numpy, scipy, soundfile, matplotlib, librosa, tqdm

## Usage

### Official test suite (all algorithms, one command each)

```bash
.venv/Scripts/python test_subband_echo_cancellation.py cgmdf    # or nlms | fdnlms | pfcg
```

### Parameterized A/B driver

```bash
.venv/Scripts/python run_fdnlms.py canonical --algo pfcg --gamma 0.4 --beta-method hestenes-stiefel
.venv/Scripts/python run_fdnlms.py speechlp --algo fdnlms --per-second
# pairs: canonical | noise | speechlp | reverb_conference ; --delay D for synthetic pure delays
```

### Python API

```python
from conjugate_mdf import CONJUGATE_MDF, FD_NLMS
from pfdaf_cg import PFDAF_CG
import numpy as np
import soundfile as sf

ref, sr = sf.read('audio/reference.wav')   # Far-end (loudspeaker)
mic, _  = sf.read('audio/microphone.wav')  # Near-end (mic with echo)

FFT, HOP = 512, 128
f = PFDAF_CG(NCHAN=1, NBIN=FFT // 2 + 1, N_G=8, hop=HOP, gamma=0.4)
out = np.zeros(len(ref))
for i in range((len(ref) - FFT) // HOP):
    s = i * HOP
    E = f.apply(np.fft.rfft(mic[s:s+FFT]).reshape(-1, 1),
                np.fft.rfft(ref[s:s+FFT]).reshape(-1, 1))
    out[s+FFT-HOP:s+FFT] = np.fft.irfft(E[:, 0], n=FFT)[FFT-HOP:]

sf.write('output.wav', out, sr)
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
| `test_subband_echo_cancellation.py` | Official suite — keys `nlms` / `fdnlms` / `cgmdf` / `pfcg` |
| `run_fdnlms.py` | Parameterized CLI driver (pair, algo, γ/β/k_max knobs, per-second ERLE) |
| `test_echo_path_comparison.py` | Echo path estimation verification (shared harness) |
| `visualize_weight_convergence.py` | Weight convergence visualization suite |
| `create_test_files.py` | Generate the canonical test pair (50 ms delay, 0.4 gain) |
| `create_noise_echo.py` | Parameterized noise+lowpass echo generator |

## Key Design Decisions

1. **`[0;e]` criterion (grade only the new block)**: the overlap-save head
   cannot serve non-hop-aligned delays — grading it causes the criterion
   cliff (~10 dB ceiling at sub-hop delays). All current engines use the
   zero-headed error; see DATAFLOW §7 for the geometry and the ablation.
2. **Overlap-save tail-keep, no windows**: raw frames beat WOLA/windowed
   variants on every test pair (taper tax: −1.2 dB speech, −25 dB flat
   spectra, μ=1 instability on colored speech).
3. **Excitation gate, not level threshold**: adapt only while
   P_ref > gate_rel·P_mic (ratio) AND above a running-peak floor — a pure
   level detector cannot separate quiet speech from a reverb pause.
4. **Two-level verification**: signal-level ERLE and model-level echo path
   metrics catch different classes of bugs.

## References

- García Morales, L., Beracoechea, J.A., Torres-Guijarro, S., Casajús-Quirós, F.J. (2006). "Conjugate Gradient Techniques for Multichannel Acoustic Echo Cancellation in Frequency Domain." AES 120th Convention, Paper 6713.
- Boray, G., Srinath, M.D. (1992). "Conjugate Gradient Techniques for Adaptive Filtering." IEEE Trans. CAS-I 39(1).
- Lee, K.-A., Gan, W.-S., Kuo, S.M. "Subband Adaptive Filtering: Theory and Implementation."
- Ferrara, E.R. (1980). "Fast Implementations of LMS Adaptive Filters." IEEE Trans. ASSP.
- Mansour, D., Gray, A.H. (1982). "Unconstrained Frequency-Domain Adaptive Filter." IEEE Trans. ASSP.
- Sayed, A.H. (2003). "Fundamentals of Adaptive Filtering." Wiley.
- Haykin, S. "Adaptive Filter Theory."

## Author

Jinboli

## License

MIT — see [LICENSE](LICENSE) for details.

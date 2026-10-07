# Linear Acoustic Echo Cancellation — Partitioned & Filterbank Frequency-Domain Adaptive Filters

A collection of frequency-domain adaptive filtering algorithms for **Acoustic
Echo Cancellation (AEC)**, implemented in pure Python/NumPy. Two processing
machines are compared on identical signals, metrics, and ground-truth echo
paths: the classic **overlap-save FDAF** family and a **4×-oversampled WOLA
filterbank** family whose per-bin RLS solver is the current champion
(**44.68 dB ERLE** on the official pair).

**Start here:** [docs/OVERVIEW.md](docs/OVERVIEW.md) — theory, the two
machines, the full engine scoreboard, the tangled gradient questions, and the
configuration reference.

## Results at a glance

Official suite — `test_subband_echo_cancellation.py`, canonical pair
(50 ms delay, 0.40 gain, LibriSpeech speech, 5 s), metric = ERLE after a 1 s
transient + echo-estimate correlation + learned-path check:

| key | engine | machine | ERLE | corr | status |
|-----|--------|---------|------|------|--------|
| `nlms` | time-domain NLMS (per-sample) | — | 22.27 dB | 0.997 | PASS |
| `fdnlms` | FD_NLMS (OLS, [0;e] + G) | OLS 512/128 | 23.77 dB | 0.998 | PASS |
| `cgmdf` | FD_NLMS **β=0** (≡ fdnlms; β=0.3 → 24.47 dB via run_fdnlms) | OLS 512/128 | 23.77 dB | 0.998 | PASS |
| `pfcg` | PBFDAF-CG (AES 2006, γ=0.4) | OLS | 26.11 dB | 0.999 | PASS |
| `fbtoe` | **WOLA FB-Toeplitz + per-bin RLS** | WOLA 1024/256 | **44.68 dB** | **1.0000** | PASS |

### The WOLA solver ladder (same filterbank, three solvers)

ERLE on three signals — canonical speech, 30 s lowpassed speech, and a
stationary white-noise probe with the echo at delay 512 (4 s):

| solver | canonical | speechlp | d512@4s | role in the story |
|--------|-----------|----------|---------|-------------------|
| correlation gradient (best: β=1, δ=8) | 9.35 dB | 15.27 dB | 26.57 dB | FAIL — the windowed system's target wanders every frame |
| error gradient (Chang-Willson CG, hybrid) | 29.78 dB | 35.46 dB | 29.65 dB | PASS — unbiased direction, matrix demoted to stride |
| **per-bin RLS** | **44.68 dB** | **46.62 dB** | **69.20 dB** | champion — exact Newton step via a maintained inverse |

Why the ladder: the three solvers give the autocorrelation matrix three
different jobs. **Correlation** lets it *define the target* (`T·w = rcross`)
— but `rcross` is re-estimated from noisy speech every frame and cond(T) ≈ 220
amplifies that noise, so the target itself wanders (damping cannot fix a
moving destination). **Error** makes the direction independent of the matrix
(the instantaneous error gradient has *zero expectation at the true path*)
and uses T only to size the stride. **RLS** maintains the exact inverse
P = R⁻¹ (Sherman-Morrison, negligible at 8 taps) with a fading loading and an
excitation gate against covariance wind-up — direction and stride both exact.
Details: [docs/autocorr_matrix_methods.md](docs/autocorr_matrix_methods.md)
and ![the four jobs](docs/autocorr_matrix_roles_en.png).

**Correlation tuning cannot escape the failure** — old class defaults vs the
full parameter sweep (β, δ, β-method, k_max, reset, geometry all swept;
d512 probe: RLS run ungated — stationary always-on excitation):

| configuration | canonical | speechlp | d512@4s |
|---|---|---|---|
| correlation, old defaults (β=0.97, δ=0.1) | 6.39 | 6.89 | 17.25 |
| correlation, retuned (β=0.999/1.0, δ=1.0) | 8.92 / 9.13 | 14.2 / 15.1 | 24.7 / 24.9 |
| correlation, best of sweep (β=1, δ=8) | 9.35 | 15.27 | 26.57 |
| error gradient (champion CG) | 29.78 | 35.46 | 29.65 |
| RLS | 44.68 | 46.62 | 69.20 |

Every knob combined is worth +3 dB — the direction itself points at a moving
target; no step size fixes that.

Geometry stacks on top of the solver: at hop 128 (8× overlap) the same engines
reach **RLS 55.30 dB / CG 36.92 dB** on canonical (scratch-verified; wrapper
still at 1024/256 for benchmark continuity). Dense reverb needs coverage
first: reverb30 (800 ms RIR) requires n_g ≥ 32, where RLS also leads
(26.31 vs 23.85).

## The two processing machines

Both machines take the same signals in and produce the same residual stream;
they differ in **what the FFT is** and therefore in how the criterion must be
built.

### Overlap-Save — the FFT is a computational tool

Weights are **full frame spectra** (one 512-point filter per partition); the
frame-wise spectral product is a *circular* convolution whose wrapped head
must never be graded — the zero-headed **[0;e]** criterion grades only the
fresh block, and the G projection keeps each partition's impulse response
causal. Grading the whole frame instead causes the **criterion cliff**
(5.94 dB vs 23.77 dB measured):

![criterion cliff](docs/criterion_cliff.png)
![overlap-save head](docs/overlap_save_head.png)

### WOLA — the FFT *is* the filterbank

The windowed frame FFT **is** a 513-band filterbank; each bin runs its own
tiny complex FIR **across subband ticks** (`Ŷ_b[m] = Σ_j w_b[j]·X_b[m−j]`).
The per-bin error is a scalar formed in the subband domain — a circular-alias
head is *structurally impossible*, so no [0;e] and no G projection exist.
Perfect reconstruction comes from the sqrt-Hann analysis/synthesis pair
(q²-COLA). Note the naming trap: WOLA's synthesis overlap-add is
*reconstruction*, not the fast-convolution OLA.

![WOLA has no time folding](docs/wola_folding.png)

Full comparison (isomorphism of the multiplication structures, the three
real difference axes, oversampling conventions):
[docs/wola_vs_overlapsave.md](docs/wola_vs_overlapsave.md) (English:
[docs/wola_vs_overlapsave_en.md](docs/wola_vs_overlapsave_en.md)).

## Algorithms

| engine | file | machine | solver | note |
|--------|------|---------|--------|------|
| **FD_NLMS** | `FD_NLMS.py` | OLS 512/128 | instantaneous NLMS; `beta>0` = gradient averaging (the former CONJUGATE_MDF, merged bit-exact); `rho` proportionate; `preemph` in-class whitening | canonical [0;e] engine |
| **PBFDAF-CG** | `pfdaf_cg.py` | OLS | γ-averaged error gradient + per-bin Gram + CG (k_max=1 in streaming) | fastest OLS on long signals |
| **FB-Toeplitz** | `conjugate_fb_toeplitz.py` | WOLA 1024/256 | `solver='rls'` (champion) or `'cg'` with `gradient='error'`/`'correlation'` | the filterbank family |

Removed engines (measurements archived in
`results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md` and
`results/2026-10-06_conjugate-toeplitz/`): full-frame correlation Toeplitz
(the 5.94 dB cliff counterexample) and the WOLA-NLMS sibling (10.66 dB; the
NLMS baseline role is covered by FD_NLMS).

PBFDAF-CG vs FD_NLMS across pairs (γ=0.4, Hestenes-Stiefel) — the CG advantage
**grows with signal length**:

| pair | PBFDAF-CG | FD_NLMS | Δ |
|------|-----------|---------|---|
| canonical (5 s) | 26.11 dB | 23.77 dB | +2.3 |
| noise + lowpass (5 s) | 48.06 dB | 43.20 dB | +4.9 |
| speech + lowpass (30 s) | 31.44 dB | 21.67 dB | **+9.8** |

## Test data and the ground truth

All test pairs are synthetic by construction — the echo path h is built
explicitly and `mic = conv(ref, h)`, so every run can verify the engine
**learned the physics**, not just got quiet:

- reference choice: LibriSpeech speech (canonical — colored, realistic),
  white noise (the conditioning-friendly ceiling oracle), 30 s variants
  (the data-quantity axis), three documented rooms;
- echo tiers: single delayed tap (0.4 @ 50 ms) → Butterworth-lowpassed tap →
  a three-part synthetic room RIR (direct path + 8 early reflections +
  lowpass-colored RT60 tail + secondary reflections; scenarios office /
  conference / huddle, seeds 42/123/456, normalized to a passive room:
  energy ≤ 1 and |H(f)| ≤ 1);
- success = ERLE ≥ 15 dB **and** echo-estimate correlation ≥ 0.9 **and** the
  learned path matching the ground truth (NMSE < −20 dB, peak position and
  amplitude). Example: canonical true `0.40 @ 800` vs learned
  `0.3941 @ delay 800`.

Full details: [docs/OVERVIEW.md §6](docs/OVERVIEW.md#6-test-data-reference-choice-echo-reverb-and-the-ground-truth).

## Usage

```bash
.venv/Scripts/python test_subband_echo_cancellation.py fbtoe   # champion
.venv/Scripts/python test_subband_echo_cancellation.py fdnlms  # OLS baseline (nlms | cgmdf | pfcg too)
.venv/Scripts/python run_fdnlms.py canonical --algo cgmdf --beta 0.3
.venv/Scripts/python run_fdnlms.py speechlp --algo fdnlms --preemph 0.95
.venv/Scripts/python run_fdnlms.py reverb30 --algo pfcg --delta 6
# pairs: canonical | noise | speechlp | reverb_conference | reverb30 ; --delay D = synthetic pure delay
```

Python API (any engine obeys the same frame contract):

```python
from FD_NLMS import FD_NLMS
from pfdaf_cg import PFDAF_CG
import numpy as np, soundfile as sf

ref, sr = sf.read('audio/reference.wav')
mic, _  = sf.read('audio/microphone.wav')
FFT, HOP = 512, 128
f = PFDAF_CG(NCHAN=1, NBIN=FFT // 2 + 1, N_G=8, hop=HOP, gamma=0.4)
out = np.zeros(len(ref))
for i in range((len(ref) - FFT) // HOP):
    s = i * HOP
    E = f.apply(np.fft.rfft(mic[s:s+FFT]).reshape(-1, 1),
                np.fft.rfft(ref[s:s+FFT]).reshape(-1, 1))
    out[s+FFT-HOP:s+FFT] = np.fft.irfft(E[:, 0], n=FFT)[FFT-HOP:]
```

## Harness

Two-level verification — signal-level (did it cancel?) and model-level (did
it learn the true path?): NMSE, correlation, coherence, delay error,
amplitude error.

```
harness/
├── core/       interfaces.py (AdaptiveFilter ABC), runner.py (run_test/TestConfig), report.py
├── metrics/    comparison.py (ERLE/corr/echo-path), spectrum.py (band ERLE, spectrograms)
└── adapters/   frame-contract adapters (CGMDFAdapter = FD_NLMS β, FDNLMSAdapter)
```

Generators: `wav_files_scripts/echo_path_generator.py` (room model) +
`harness_template/ground_truth/generators.py` (room wrappers, driven by
`create_test_files.py` / `create_noise_echo.py` / `create_reverb_signals.py`).
Methodology: [HARNESS_ENGINEERING.md](HARNESS_ENGINEERING.md) and
[docs/OVERVIEW.md §6–7](docs/OVERVIEW.md).

## Documentation index

| doc | content |
|-----|---------|
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | **front door** — theory, two machines, scoreboard, tangled questions, test data, harness, config |
| [docs/wola_vs_overlapsave.md](docs/wola_vs_overlapsave.md) / [EN](docs/wola_vs_overlapsave_en.md) | the two machines in depth (the naming traps, the three real difference axes, [0;e]'s family split) |
| [docs/autocorr_matrix_methods.md](docs/autocorr_matrix_methods.md) / [EN](docs/autocorr_matrix_methods_en.md) | the autocorrelation matrix's four jobs (correlation / error / PFCG / RLS) + update rules side by side |
| [docs/fold_and_oversampling_notes.md](docs/fold_and_oversampling_notes.md) / [EN](docs/fold_and_oversampling_notes_en.md) | the reference rig's fold (polyphase) trick; three kinds of "folding" disambiguated; oversampling conventions |
| [DATAFLOW.md](docs/DATAFLOW.md) | signal-flow diagrams, buffer structures, the criterion cliff geometry (§7), benchmarks (§10), the WOLA family (§11) |
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | family tree, design decisions |
| [docs/RLS_DEBUG_PROCESS.md](docs/RLS_DEBUG_PROCESS.md) | the original MDF debug chronicle (5 phases) |
| [TODO.md](TODO.md) | current roadmap — done cycles, bottlenecks, ranked next steps |

Figures (generated locally; `*.png` is gitignored): criterion cliff
(`docs/criterion_cliff.png`, `docs/cliff_geometry.png`), overlap-save head
(`docs/overlap_save_head*.png`), WOLA folding proof (`docs/wola_folding.png`),
fold trick (`docs/fold_trick.png`), the four matrix jobs
(`docs/autocorr_matrix_roles_en.png`), window placement and gate geometry
(`docs/window_placement.png`, `docs/gate_dense_matrix.png`).

## Key design decisions

1. **Grade only what you emit.** On OLS that is the [0;e] zero-headed error
   (the circular head cannot serve sub-hop delays — grading it is the
   criterion cliff). On WOLA the per-bin scalar error is formed in the
   subband domain, so the headless property is structural, not a rule.
2. **The gradient must be unbiased; the matrix may only size the stride.**
   Directions derived from windowed statistics (correlation mode) chase a
   target that wanders with the estimation noise — measured catastrophic on
   speech. The instantaneous error gradient has zero expectation at the true
   path; the autocorrelation matrix then only shapes the step (or, in RLS, is
   maintained as an exact inverse).
3. **Excitation gate, not a level threshold** — adapt only while
   P_ref > gate_rel·P_mic and above a running-peak floor; a level detector
   cannot separate quiet speech from a reverb pause. For RLS the gate is the
   covariance wind-up guard, not an option.
4. **Two-level verification** — signal-level ERLE and model-level echo-path
   metrics catch different failure classes; the ground-truth path is part of
   the pass criteria.
5. **Coverage before speed on reverb** — N_G (OLS) or n_g·hop (WOLA) must
   exceed the path length first; then the solver decides how deep it gets.

## Requirements

- Python 3.12 (project venv: `.venv`)
- numpy, scipy, soundfile, matplotlib, librosa, tqdm

## References

- García Morales, L., et al. (2006). "Conjugate Gradient Techniques for
  Multichannel Acoustic Echo Cancellation in Frequency Domain." AES 120th,
  Paper 6713. (PFCG)
- Chang, K.-H., Willson, A.N. (2000). "Analysis of Conjugate Gradient
  Algorithms for Adaptive Filtering." IEEE Trans. SP. (the FB-Toeplitz CG
  solver's lineage)
- PAES/Eneman et al. — PBFDAF error-gradient averaging (the φ update).
- Haykin, S. "Adaptive Filter Theory." ch. 9 (RLS).
- Paleologu, C., Benesty, J. et al. — VFF-RLS for tracking (roadmap).
- Crochiere & Rabiner (1983), *Multirate Digital Signal Processing* (WOLA /
  polyphase); Allen & Rabiner (1977) STFT.
- Ferrara (1980) FDAF; Mansour & Gray (1982) unconstrained FDAF; Soo & Pang
  (1990) MDF; Shynk (1992) survey.

## Author

Jinboli

## License

MIT — see [LICENSE](LICENSE) for details.

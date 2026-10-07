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

### The WOLA family: three solvers, one filterbank

The same WOLA front end (analysis → per-bin error → synthesis) runs with
three interchangeable solvers. Signals below: **canonical** = 5 s LibriSpeech
speech (official pair), **speechlp** = 30 s lowpassed speech, **d512@4s** = a
stationary white-noise probe with the echo at delay 512.

#### 1. Correlation gradient — the windowed-system solver (FAIL)

Direction = the residual of the windowed normal equations,
`g = rcross − T·w` with T = toeplitz(autoR) accumulated over a β window: it
solves `T·w = rcross`, i.e. the matrix **defines the target**. On streaming
speech `rcross` is re-estimated from recent noisy data every frame and
cond(T) ≈ 220 (4× geometry) amplifies that noise — the target itself wanders,
and damping cannot fix a moving destination. Tuning trajectory (all knobs
swept: β, δ, β-method, k_max, reset, geometry):

| configuration | canonical | speechlp | d512@4s |
|---|---|---|---|
| old defaults (β=0.97, δ=0.1) | 6.39 | 6.89 | 17.25 |
| retuned (β=0.999/1.0, δ=1.0) | 8.92 / 9.13 | 14.2 / 15.1 | 24.7 / 24.9 |
| **best of sweep (β=1, δ=8)** | **9.35** | **15.27** | **26.57** |
| *with longer audio* (tiled 20 s, best config) | *14.46* | — | *38.08 @ 64 s* |

Longer audio DOES help it — but only in its one honest regime: stationary +
on-grid (d512 reaches 38.08 dB at 64 s, beating the CG hybrid there). On
speech it stays ~30 dB below its own fixed point (the frozen exact solve of
the same system is 45.4 dB): a trajectory failure, not a system failure.
Kept as the class default purely as a teaching counterexample.

#### 2. Error gradient — the CG hybrid (PASS, 29.78)

Direction = the γ-averaged **instantaneous** error gradient
`φ ← γφ + (1−γ)·conj(rx)·E` (Chang-Willson CG1 machinery: Rayleigh line
search, conjugate-direction memory, η=0.999 damping). The direction is
**unbiased** — its expectation is exactly zero at the true path, at any
delay, any window — and T is demoted to the line-search denominator with a
current-power floor (T only sizes the stride; a wrong T costs speed, never
accuracy):

| configuration | canonical | speechlp | d512@4s |
|---|---|---|---|
| 1024/256 (wrapper default) | 29.78 | 35.46 | 29.65 |
| **1024/128 (hop halved)** | **36.92 (+7.1)** | **40.84 (+5.4)** | 31.85 |
| 2048/128 | 35.68 | 41.20 | 36.54 |

| longer audio (1024/256) | canonical | d512 probe |
|---|---|---|
| 5 s → 20 s → 80 s (tiled) | 29.78 → 33.85 → **37.08** | — |
| 4 s → 16 s → 64 s → 256 s (d512) | — | 29.75 → 32.98 → 36.91 → **42.18** |

The time-slope is diagnostic: the d512 climb accelerates toward the
10 dB/decade averaging limit (5.4 → 8.7 dB/dec measured) — pure slowness,
**no misadjustment floor**. But it is ~6 orders of magnitude slower than RLS
to any given depth: at 80 s it is still 8 dB below its own 45.4 dB fixed
point. Robustness (δ=1 floor, k_max=1) traded for speed is this family's
structural cost — every acceleration knob measured (δ<1, k_max>1, fading
loading) makes it worse or diverges.

#### 3. Per-bin RLS — the champion (44.68)

Each bin maintains the exact inverse Gram P = R⁻¹ by Sherman-Morrison rank-1
(O(n_g²) per bin — negligible at 8 taps), P(0) = I/δ fading loading, λ=0.999
forgetting; the update `w += K·ξ` is an exact Newton step — best direction
and best stride at once:

| configuration | canonical | speechlp | d512 probe (ungated) |
|---|---|---|---|
| 1024/256 (official wrapper) | **44.68** | **46.62** | 69.20 @4 s |
| 1024/128 (scratch) | 55.30 | 57.44 | — |
| 2048/128 (scratch) | 64.77 | 65.21 | — |

| longer audio (d512 probe, ungated) | 4 s | 16 s | 64 s | 256 s |
|---|---|---|---|---|
| ERLE (dB) | 69.18 | 75.61 | 81.77 | **87.86** |

RLS runs exactly at the 10.3 dB/decade averaging limit — gated RLS is already
**at** the windowed-LS fixed point by 5 s on canonical (what "champion"
means in practice). Two cautions: WITHOUT the excitation gate it wind-ups
during speech pauses (P ← P/λ inflation; 44.68 → 26.5) — the gate is the
anti-wind-up, not an option; and dense reverb needs coverage first
(reverb30: n_g ≥ 32, where RLS leads 26.31 vs 23.85).

Why this ordering — the three solvers give the autocorrelation matrix three
different jobs (define-the-target / stride-only / exact-inverse); the
form is innocent (‖R−T‖/‖R‖ = 0.37%, frozen T-solve = 45.4 dB ≈ RLS 46.7).
Details and the four update rules side by side:
[docs/autocorr_matrix_methods.md](docs/autocorr_matrix_methods.md) and
![the four jobs](docs/autocorr_matrix_roles_en.png).

### Geometry: hop and nfft are first-order knobs

Shrinking the hop at fixed nfft improves TWO things at once — the adaptation
rate (steps/second ∝ 1/hop) and the subband lag-grid density (the canonical
800-sample delay = 6.25 lags at hop 128 vs 3.125 at hop 256, halving the
fractional-lag interpolation burden). Measured on the CG error-hybrid
(canonical / speechlp / d512@4s), oversampling ratio = nfft/hop
(standard convention: nbin/hop = half that):

| nfft/hop | overlap | canonical | speechlp | d512@4s |
|----------|---------|-----------|----------|---------|
| 1024/256 *(wrapper default)* | 4× | 29.71 | 35.42 | 29.65 |
| **1024/128** | **8×** | **36.92 (+7.1)** | **40.84 (+5.4)** | **31.85** |
| 2048/128 | 16× | 35.68 | 41.20 | 36.54 |
| 1024/64 | 16× | 6.57 | 6.67 | 23.27 |
| 256/128 | 2× | 19.67 | 19.62 | 67.12 |

The table has edges in both directions: hop 64 over-collinearizes the lag
regressors (16× ratio → the per-bin system degenerates, 6.6 dB), and a short
window (256/128) starves speech while being near-RLS-fast on the stationary
probe (67.12). **1024/128 is the sweet spot**, and the gain STACKS with the
solver — solver quality and geometry are two independent axes:

| geometry | CG error hybrid | RLS |
|---|---|---|
| 1024/256 (wrapper default) | 29.78 | 44.68 |
| 1024/128 | 36.92 | **55.30** |
| 2048/128 | 35.68 | **64.77** |

(speechlp at hop 128: CG 40.84, RLS 57.44; 2048/128 → 65.21.)

**More time also helps** — and the *slope* diagnoses the engine. On the
stationary d512 probe the CG hybrid climbs 29.75 → 36.91 → 42.18 dB at
4/64/256 s with a slope accelerating toward the 10 dB/decade averaging
limit: pure slowness, **no misadjustment floor** (RLS runs exactly at
10.3 dB/decade: 69.2 → 81.8 → 87.9 at 4/64/256 s). On tiled canonical the
CG hybrid gains +3.3 dB per 4× data (29.78 → 33.85 → 37.08 at 5/20/80 s)
but would need decades more to reach its own 45.4 dB fixed point, while
gated RLS is already there at 5 s — the practical meaning of "the champion
sits at the fixed point". Long-run caution for RLS: WITHOUT the gate it
wind-ups in speech pauses (44.68 → 26.5) — the gate is the anti-wind-up,
not an option.

Price of hop 128: latency nfft−hop (56 ms at 1024/128, 112 ms at 2048/128)
and compute ×2 per hop halving. Dense reverb needs coverage first:
reverb30 (800 ms RIR) requires n_g ≥ 32 (RLS 26.31 vs CG 23.85 at n_g=32).

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
(5.94 dB vs 23.77 dB measured). Buffer handling per frame (FFT 512 / hop 128):

```
x[hop] ──► keep 384 old + 128 new samples ──► rfft ──► X        (regressor)
est_b  = Σ_p W[b,p]·X_b(m−p)              p = 0…N_G−1    (partition lags)
R      = D − est  ──► e_tail = irfft(R)[−hop:]            (discard the head)
E_upd  = rfft([0; e_tail])                                (zero-headed error)
W     += μ·conj(X_p)·E_upd / (X2 + ε)      →  G projection (support ≤ hop)
output: the fresh e_tail
```

![criterion cliff](docs/criterion_cliff.png)
![overlap-save head](docs/overlap_save_head.png)
![head discard](docs/overlap_save_head_discard.png)
![window placement](docs/window_placement.png)

### WOLA — the FFT *is* the filterbank

The windowed frame FFT **is** a 513-band filterbank; each bin runs its own
tiny complex FIR **across subband ticks** (`Ŷ_b[m] = Σ_j w_b[j]·X_b[m−j]`,
m = subband tick = one hop of audio, j = subband lag = one tap). The per-bin
error is a scalar formed in the subband domain — a circular-alias head is
*structurally impossible*, so no [0;e] and no G projection exist. Perfect
reconstruction comes from the sqrt-Hann analysis/synthesis pair (q²-COLA).
Note the naming trap: WOLA's synthesis overlap-add is *reconstruction*, not
the fast-convolution OLA.

The full data flow (`conjugate_fb_toeplitz.py`, one hop per tick):

```
x[hop], d[hop]
  │  analysis: X = rfft(q · [history ‖ new hop])      q = sqrt-Hann
  ▼        (nfft 1024 → 513 bins; subband rate = fs/hop)
buf_X shift register ──► rx[:, j] = X(m−j)            (subband lags j = 0…n_g−1)
  ▼
per-bin a priori error   E_b[m] = D_b[m] − Σ_j w_b[j]·X_b[m−j]   ← SCALAR, no head
  ▼
┌─ solver (three interchangeable members) ──────────────────────────────────┐
│ RLS:      K = P·conj(u)/(λ+uᴴPu);  w += K·E;  P ← (P−K·uᴴP)/λ            │
│           (P = R⁻¹ maintained by Sherman-Morrison; P(0)=I/δ fading       │
│            loading; gate freezes P and w — the anti-wind-up)             │
│ CG error: φ ← γφ + (1−γ)·conj(rx)·E;                                        │
│           α = 0.999⟨g,v⟩/(⟨v,Tv⟩ + δ·P_inst‖v‖²)  — T = toeplitz(autoR)  │
│           only sizes the stride (the direction is unbiased at the truth) │
│ CG corr:  g = rcross − T·w  → solves the windowed system T·w = rcross    │
│           (defines its own target; fails on speech — target wanders)     │
└───────────────────────────────────────────────────────────────────────────┘
  ▼
synthesis: acc += (2/k)·q·irfft(E);  emit the oldest hop   (k = nfft/hop)
```

![WOLA has no time folding](docs/wola_folding.png)
![window placement](docs/window_placement.png)

Full comparison (isomorphism of the multiplication structures, the three
real difference axes, oversampling conventions):
[docs/wola_vs_overlapsave.md](docs/wola_vs_overlapsave.md) (English:
[docs/wola_vs_overlapsave_en.md](docs/wola_vs_overlapsave_en.md)).

#### The reference rig's variant: polyphase fold before a short FFT

The original rig (`polyphase_dft_fb_analysis.m`) runs the same WOLA idea
with a **longer prototype than the FFT** (Lp = 1024, nfft = 512): because
the demodulation waveform e^(−j2πkd/512) has period nfft, the two halves of
the windowed frame (delays d and d+512) carry identical modulation phase
and can be **added before a single 512-FFT** — the "fold". This is exact
(1e-14), halves the FFT cost and the channel count (257 coarse bins), and
decouples filter length from FFT size; our engine sits at the OS=1 special
case where fold is the identity (1024 window, 1024 rfft, 513 fine bins).
![fold trick](docs/fold_trick.png)
Full notes: [docs/fold_and_oversampling_notes.md](docs/fold_and_oversampling_notes.md)
(English: [docs/fold_and_oversampling_notes_en.md](docs/fold_and_oversampling_notes_en.md));
rig principle: ![rig buffer + fold](docs/osfb_analysis_buffer.png).

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

# Linear Acoustic Echo Cancellation — Theory, Code, Configuration, Results

> The front-door technical document. Generated 2026-10-06; all numbers were
> re-run on that date. Deep dives:
> [wola_vs_overlapsave.md](wola_vs_overlapsave.md) (the two machines, CN/EN),
> [autocorr_matrix_methods.md](autocorr_matrix_methods.md) (the autocorrelation
> matrix's four jobs), [fold_and_oversampling_notes.md](fold_and_oversampling_notes.md)
> (the rig's fold trick + oversampling conventions), DATAFLOW.md (signal-flow
> diagrams), PROJECT_ARCHITECTURE.md (module map).

---

## 1. The problem, and what we built

A loudspeaker plays a far-end signal `x[n]`; the room's echo path `h[n]`
(canonical test pair: a pure 800-sample / 50 ms delay with gain 0.40) makes
the microphone pick up `d[n] = (h * x)[n]`. The canceller adaptively builds
`ĥ` and subtracts the estimated echo; success is measured after skipping the
first 1 s of convergence:

- **ERLE** = 10·log10(⟨d²⟩/⟨e²⟩) — echo suppression depth (target ≥ 15 dB);
- **correlation** of the estimated echo with the true echo (target ≥ 0.9);
- the **learned path** must match the true one in amplitude and position.

Everything in this repo is **linear** AEC (no NLP/residual suppressor — out
of scope by decision). We built and compared **seven engines across two
machines** (§3), which reduced to five official suite keys after the failed
and redundant ones were removed (§4).

### Why frequency domain at all

A 2048-tap time-domain NLMS costs 2048 MACs per sample. The frequency-domain
trick: overlap frames of 512 samples, FFT once, and the echo estimate becomes
a **pointwise product of spectra** partitioned across `N_G` frame lags —
O(N log N) instead of O(N²), at the price of one complexity: the frame-wise
FFT turns linear convolution into **circular** convolution, which creates a
wrapped "head" of aliased samples. Handling that head correctly is the
criterion cliff story (§2).

## 2. Theory: the criterion cliff and the [0;e] fix

With weights stored as **full frame spectra** (the overlap-save machine,
§3.1), each frame's error is formed by an irfft of a same-frame spectral
product. The circular wrap contaminates the first `hop−1` samples of every
frame; the middle samples are clean but were already emitted (scoring them
again double-counts); only the last `hop` samples are clean **and** fresh.
Hence the canonical criterion:

```
E = rfft([0; e_tail])          # zero-headed error — grade ONLY the fresh block
```

Score the whole frame instead and the adaptation fits the garbage head: the
measured cliff is **5.94 dB vs 23.77 dB** on the same pair (the deleted
full-frame counterexample engine; delay-mod-hop dependent). This is the
criterion layer. Independently, the **path layer** constrains what the
weights can represent (the G projection zeroes each partition's time-domain
taps beyond `hop` so the spectral products stay exact linear convolutions),
and the **solver layer** chooses the update. The three layers are
independent — the WOLA machine (§3.2) needs neither [0;e] nor G because its
head is structurally absent.

## 3. Two machines (buffer processing)

### 3.1 Overlap-Save (OLS) — the FFT is a *computational tool*

```
x[hop] in ──► slide 512-frame (keep 384 old + 128 new) ──► rfft ──► X
          ┌────────────────────────────────────────────┐
          │ ŷ_b = Σ_p W[b,p]·X_b(m−p)   (partitioned    │   weights: FULL spectrum
          │     spectral product per bin, p = 0…N_G−1)  │   columns W[:,p] — one
          └────────────────────────────────────────────┘   512-pt filter each
          ▼
e_frame = irfft(D − ŷ)  →  emit the LAST hop (head = circular wrap, discarded)
gradient: E = rfft([0; e_tail])  →  update W  →  G projection (support ≤ hop)
```

Machines of this family: FD_NLMS (with the β option), PFCG. Weights are
frequency-domain columns; bins are coupled by the G projection. Causal
latency = nfft − hop = 384 samples (24 ms) at 512/128.

### 3.2 WOLA filterbank — the FFT *is* the filterbank

```
x[hop] in ──► q · [1023-old ‖ 128-new] ──► rfft ──► X_b[m]  (b = 0…512)
          ┌──────────────────────────────────────────────┐
          │ per-bin SCALAR FIR across subband TICKS:      │   weights: w[b, j]
          │ Ŷ_b[m] = Σ_j w_b[j]·X_b[m−j],  j = 0…n_g−1    │   = n_g taps of ONE bin
          └──────────────────────────────────────────────┘   (bins never mix)
          ▼
E_b[m] = D_b[m] − Ŷ_b[m]         ← a scalar, scored DIRECTLY (no time frame,
                                     no head — circularity has nowhere to occur)
          ▼
synthesis: q · irfft(E) overlap-added (Σq² = k/2 COLA ⇒ perfect reconstruction)
```

Geometry: nfft 1024 / hop 256 = **4× oversampled** in the overlap sense
(k = nfft/hop; the standard filterbank oversampling ratio is nbin/hop ≈ 2×),
513 bins, each bin's subband sampled at fs/hop = 62.5 Hz. `n_g·hop` = 2048
samples of path coverage. Delay grid = hop (delays must be interpolated
across taps when d mod hop ≠ 0 — the off-grid floor, a soft −2.5 dB at 4×).
Causal latency = nfft − hop = 768 samples (48 ms). Window q = sqrt-Hann on
analysis and synthesis; the q²-COLA overlap sum is the constant k/2,
normalized away by `_syn = 2/k`.

Full comparison (including the three axes that actually differ once you
notice both machines do per-bin cross-lag convolution):
[wola_vs_overlapsave.md](wola_vs_overlapsave.md).

## 4. The engines, their results, and why performance differs

### 4.1 Scoreboard (canonical pair, official metric, re-run 2026-10-06)

| # | engine | how to run | machine / geometry | criterion | direction | matrix's job | ERLE | verdict |
|---|--------|-----------|--------------------|-----------|-----------|--------------|------|---------|
| 1 | NLMS | `test_subband nlms` | time domain, per-sample | — | instantaneous error | X² normalization | 22.27 | PASS |
| 2 | FD_NLMS | `fdnlms` (run_fdnlms / test_subband) | OLS 512/128 | [0;e] | instantaneous error | X² + G projection | 23.77 | PASS |
| 3 | FD_NLMS β=0.3 *(former CONJUGATE_MDF, merged)* | `run_fdnlms --algo cgmdf --beta 0.3` | OLS 512/128 | [0;e] | β-averaged error | smoothed Pn + G | **24.47** | PASS |
| 4 | PFCG | `pfcg` (pfdaf_cg.py) | OLS (paper geometry) | [0;e] | γ-averaged error (unbiased) | per-bin Gram: stride + k_max model | 26.11 | PASS |
| 5 | FB-Toeplitz **correlation** | `fbtoe, gradient='correlation'` | WOLA 1024/256 | per-bin scalar | windowed residual g = rcross − T·w | **defines the target** + stride | ≤ **9.35** (best tuning) | FAIL |
| 6 | FB-Toeplitz **error** | `fbtoe, gradient='error'` | WOLA 1024/256 (4×) | per-bin scalar | γ-averaged error (unbiased) | stride only, T + current-power floor | **29.78** | PASS |
| 7 | FB-Toeplitz **RLS** ★ | `fbtoe, solver='rls'` | WOLA 1024/256 (4×) | per-bin scalar | exact Newton K·ξ | everything: P = R⁻¹ exact inverse | **44.68** | PASS (champion) |

Removed engines: full-frame correlation (the criterion-cliff counterexample,
5.94 dB — deleted, code at git `2f23d9d`); WOLA-NLMS (10.66 dB FAIL — NLMS
baseline covered by FD_NLMS; deleted). Measurements archived in
`results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md`.

### 4.2 Why the performance ladder looks like this

The scoreboard is a controlled descent of the **autocorrelation matrix's job
description** (details: [autocorr_matrix_methods.md](autocorr_matrix_methods.md)):

1. **NLMS → FD_NLMS (+1.5 dB):** the same update, accelerated. FFT
   partitioning changes cost, not convergence — the small gain is geometry
   (normalization per bin).
2. **β averaging (+0.7 dB):** the gradient accumulator `gacc = β·gacc +
   conj(X)·E_upd` with smoothed normalizer trades a little speed for
   mic-noise robustness. β=0 is bit-identical to FD_NLMS.
3. **PFCG (+2.3 dB over FD_NLMS):** the direction is still the (γ-averaged,
   hence unbiased) error gradient, but a per-bin Gram R now supplies the
   curvature — the stride fits the local valley shape, and k_max model
   iterations exist (streaming still prefers 1).
4. **correlation gradient FAILS (9.35 best, all tunings swept):** here the
   matrix *defines the target*: the direction is the residual
   `g = rcross − T·w`, whose zero is `T⁻¹·rcross`. But `rcross` is re-estimated
   from recent noisy data every frame, and cond(T) ≈ 220 at 4× amplifies that
   noise into a target that **wanders** — damping buys only +0.4 dB because
   smaller steps converge to the same moving point. The fixed point itself is
   fine (frozen exact solve = 45.4 dB!); the *trajectory* is the failure.
5. **error hybrid (+20 dB over correlation):** swap the direction to the
   instantaneous error gradient, whose expectation is **exactly zero at the
   true path** (there the residual is uncorrelated with the regressor —
   unbiased at any delay, any window). T is demoted to the line-search
   denominator: it now only measures the local valley width (stride). A wrong
   T costs speed, never accuracy.
6. **RLS (+15 dB over the hybrid):** maintain P = R⁻¹ exactly
   (Sherman-Morrison rank-1, O(n_g²) per bin — negligible at n_g = 8). The
   Newton step K·ξ is simultaneously the best direction and the best stride;
   P(0) = I/δ provides a fading loading that stabilizes the low-excitation
   stretches where exact re-solve methods swing; the excitation gate freezes
   P and w in pauses (without it: covariance wind-up, 44.68 → 26.5).

Geometry multiplies on top: hop 256→128 (8× overlap) raised RLS to **55.30**
and the CG hybrid to 36.92 (scratch-verified; not wired into the official
wrapper yet). Dense reverb needs coverage first: reverb30 (800 ms RIR)
requires n_g ≥ 32, where RLS also leads (26.31 vs 23.85).

## 5. The two tangled questions

### 5.1 correlation vs error — "direction unbiased" in one formula

Both directions are noisy per frame. The difference is **where their noise
centers**:

- error gradient: at the true path `h`, the residual E equals mic noise η,
  uncorrelated with the regressor, so **E[conj(rx)·E] = 0** — the expected
  update vanishes exactly at the truth. Noise fluctuates *around* the right
  target.
- correlation gradient: its zero is the solution of `T·w = rcross` assembled
  from this window's estimates; 1% of b-noise displaces that zero by up to
  **cond × 1% ≈ 220%** in the worst direction. The target itself jumps —
  the direction doesn't fluctuate around the truth, it chases a moving
  surrogate.

### 5.2 The T matrix's variations — form vs job

Three matrix forms appear across the engines: the **Toeplitz projection**
T = toeplitz(autoR) (fb-toeplitz), the **full Gram** R (PFCG), and the
**maintained inverse** P = R⁻¹ (RLS). The dissection showed the form is
innocent: ‖R−T‖/‖R‖ = 0.37% on real speech, condition numbers nearly equal,
and the frozen exact solve of even the T system reaches 45.4 dB. What
matters is only the job: define-the-target (fails), stride-only (works),
exact-inverse (wins). See [autocorr_matrix_methods.md](autocorr_matrix_methods.md)
for the four update rules side by side.

## 6. Test data: reference choice, echo, reverb, and the ground truth

Every test pair is **synthetic by construction**: we choose the reference
signal, we build the echo path h explicitly, and the microphone is exactly
`mic = conv(ref, h)` — so the answer is known before the algorithm starts.

### 6.1 Choosing the reference signal

| pair | reference | length | why this signal |
|---|---|---|---|
| canonical | LibriSpeech libri2 sample (via librosa), 16 kHz | 5 s | real colored speech with pauses — the realistic middle ground; THE official pair |
| noise | white Gaussian noise | 5 s | flat spectrum = the conditioning-friendly extreme; its converged ERLE is the engine's ceiling oracle |
| speechlp | same speech tiled ×3 | 30 s | the data-quantity axis (longer = deeper convergence) plus a colored path |
| reverb_office / conference / huddle | original_speech.wav | 5 s | three documented rooms (RT60 300/500/150 ms), seeds 42/123/456 |
| reverb30 (= conf30, +30n noise variant) | speech / noise | 30 s | reverb + length combined — the hardest family |

The spectrum of the reference is a first-order knob: flat excitation
excites every bin equally (well-conditioned per-bin problems), speech
concentrates energy and starves the rest — the same engine scores 48 dB on
noise but 21–31 dB on speech.

### 6.2 Generating the echo — three tiers

**Tier 1 — pure delay + gain** (`create_test_files.py`,
`create_simple_echo.py`): a single tap, `h[800] = 0.4` (50 ms, −8 dB):
`mic[800:] = 0.4 · x[:-800]`. If an engine cannot learn one tap, no
amount of sophistication matters — this is the first debug run.

**Tier 2 — filtered echo** (`create_noise_echo.py`): a 2nd-order Butterworth
lowpass (cutoff 0.3·fs/2) placed at the 800-sample tap — the path becomes
frequency-dependent but still sparse (noise pair: DC gain 0.1; speechlp:
gain 0.4). This is the pair family that exposed the time-domain/FDAF
convergence gap on colored excitation.

**Tier 3 — room reverberation** (`wav_files_scripts/echo_path_generator.py`,
wrapped by `harness_template/ground_truth/generators.py::
generate_room_echo_signals`): a three-part synthetic RIR from
`RoomParameters`:

1. **direct path** — 0.6 at the initial delay (45/60/20 ms by scenario);
2. **8 early reflections** within the first 50 ms — discrete taps at
   0.1–0.4 of the direct amplitude, random sign (wall/desk bounces);
3. **late reverberation** — Gaussian noise **lowpass-colored at 0.3·fs/2**
   (rooms absorb highs), scaled by the RT60 exponential envelope
   `exp(−t·6.9/RT60)` (−60 dB at RT60) × 0.15, with a 10 ms fade-in;
4. **15 secondary reflections** — 0.05–0.2 of direct, random sign, each
   Hann-spread over ±3 samples (diffuse);
5. a final **10 % fade-out** to avoid truncation artifacts.

Scenarios: office (500 ms filter / 45 ms / RT60 300, seed 42), conference
(800 ms / 60 ms / RT60 500, seed 123), huddle (300 ms / 20 ms / RT60 150,
seed 456). `normalize_rir=True` then scales the RIR to unit energy **and**
max |H(f)| ≤ 1 — a passive room: the microphone can never be louder than
the reference. Finally `mic = fftconvolve(ref, h)` (trimmed to length).

### 6.3 The ground-truth path as the success marker

Because h is constructed, not merely assumed, every run can check whether
the engine **learned the physics**:

1. after the run, the learned weights are folded back to the time domain
   (per machine: irfft of an OLS weight column; per-partition `irfft[:step]`
   blocks for WOLA — `get_echo_path()` in the adapters / engine wrappers);
2. `calculate_echo_path_metrics` (harness/metrics/comparison.py) compares
   against h_true: **NMSE**(dB), **correlation**, **delay error** (peak
   location), **amplitude error**(dB) at the peak, and frequency-domain
   **coherence** — `passed` = NMSE < −20 dB **and** correlation > 0.9;
3. the suite prints the learned peak next to the truth, e.g. canonical
   true `0.40 @ 800` vs learned `0.3941 @ lag 6, tap 32 = delay 800`;
   the lowpass pair's oracle run learns `0.0365 @ 802` — the true tap to
   the sample.

Why this matters: ERLE alone can be gamed — an engine that shrinks its
output without learning the path still "suppresses" power. The
ground-truth check separates *learned the room* from *got quiet by luck*,
which is why it is part of the official pass criteria alongside ERLE and
correlation.

## 7. The harness

Two layers share one philosophy — *same signals, same metric, same block
loop, swap only the engine*:

**`test_subband_echo_cancellation.py`** — the official suite. Owns the
metric (ERLE after a 1 s transient, echo-estimate correlation, band ERLE,
learned-path print), the per-engine wrapper classes (which translate the
time-block engine API to frame spectra), and the WOLA latency alignment
(block i's output written at input block i − (nfft/hop − 1)). Keys:
`nlms / fdnlms / cgmdf / pfcg / fbtoe`.

**`harness/` + `test_echo_path_comparison.py`** — the shared comparison
rig (templated in `harness_template/`):

- `harness/core/runner.py` — `TestConfig` (fft_size, overlap 0.75,
  transient_length, `true_echo_path`) + `run_test(algorithm, ref, mic,
  config)`: one block loop driving any `AdaptiveFilter` to a `TestResult`;
- `harness/adapters/` — the frame-contract adapters: `CGMDFAdapter`
  (FD_NLMS with β), `FDNLMSAdapter`, each providing `filt()/update()/
  get_echo_path()`;
- `harness/metrics/comparison.py` — ERLE, correlation, echo-path metrics;
  `spectrum.py` — band ERLE and spectrograms;
- `harness_template/ground_truth/generators.py` — the generators of §6
  (`generate_simple_echo_path`, `generate_room_echo_signals`, ...),
  composing `wav_files_scripts/echo_path_generator.py`'s room model.

Pair-audio lives in `audio/` (regenerable by the `create_*.py` drivers);
experiment records live in `results/<date>_<label>/FINDINGS.md` — both
gitignored by design. The process view (how to add an engine to the
suite, the debug funnel) is HARNESS_ENGINEERING.md; DATAFLOW.md §8–§9
carry the same flows as diagrams.

## 8. Code map

| File | Contents | Role |
|---|---|---|
| `FD_NLMS.py` *(renamed from conjugate_mdf.py 2026-10-06)* | `FD_NLMS` (+ `beta` option absorbing the former CONJUGATE_MDF) | canonical OLS engine, suite keys `fdnlms` / `cgmdf` |
| `pfdaf_cg.py` | `PFDAF_CG` | OLS + per-bin Gram + CG; key `pfcg` |
| `conjugate_fb_toeplitz.py` | `CONJUGATE_FB_TOEPLITZ` (+ `fb_toeplitz_aec` driver) | WOLA filterbank; solvers `rls` (champion) / `cg`; gradients `error` / `correlation`; key `fbtoe` |
| `test_subband_echo_cancellation.py` | official suite | keys: nlms / fdnlms / cgmdf / pfcg / fbtoe; plots + learned-path metrics |
| `run_fdnlms.py` | CLI A/B driver | pairs, algos, all knobs, per-second ERLE |
| `harness/adapters/__init__.py` | block adapters | CGMDFAdapter (FD_NLMS β) / FDNLMSAdapter for the shared harness |
| `neural_train/` | TF/JAX training contracts | numpy reference now FD_NLMS(β) |
| deleted | `conjugate_toeplitz.py` (cliff counterexample), `conjugate_full.py` (WOLA-NLMS) | recoverable from git history / FINDINGS |

Suite driver relationship: `test_subband_echo_cancellation.py` owns the
official metric and the WOLA latency alignment; `run_fdnlms.py` owns the
pair loader (`canonical` / `noise` / `speechlp` / `reverb_conference` /
`reverb30`) and the CLI; both drive engines through the same `apply()` or
`process()` time-block contract.

## 9. Configuration reference

Shared by all engines — the **excitation gate** (adapt only while the
reference is identifiable): `gate_rel=0.3` (P_ref vs P_mic), `gate_hold=2`
frames, `gate_decay=1.5 s` running peak, `gate_floor=1e-6`. `gate_rel=None`
disables. Never disable for RLS (anti-wind-up).

### FD_NLMS (keys fdnlms / cgmdf)

| knob | default | meaning |
|---|---|---|
| `mu` | 1.0 | NLMS step (0 < mu ≤ 2) |
| `N_G` | 8 | partitions; coverage ≈ (N_G)·hop samples |
| `beta` | 0.0 | gradient averaging; 0 = instantaneous; 0.05–0.3 for mic noise; 0.3 = the former CONJUGATE_MDF |
| `rho` | 0.0 | proportionate step (sparse paths; keep ≤ 0.5) |
| `preemph` | 0.0 | in-class whitening (0.9–0.97 for speech; output de-emphasized) |
| `constraint` | True | G projection (False measured +2.4 dB — the wrap taps help) |
| `eps` | 1e-10 | normalizer floor |

### PFCG (key pfcg)

| knob | default | meaning |
|---|---|---|
| `gamma` | 0.4 | gradient-memory average (0 = pure FD_NLMS descent) |
| `k_max` | 1 | CG iterations per frame (>1 diverges in streaming) |
| `beta_method` | hestenes-stiefel | HS ≥ DY > PR ≫ FR (FR diverges) |
| `delta` | 1.0 (CLI) | Tikhonov stride floor; 4–8 on colored dense reverb |
| `constrain` | 'full' | G projection every frame |

### FB-Toeplitz (key fbtoe) — the WOLA family

| knob | default | meaning |
|---|---|---|
| `nfft / hop` | 1024 / 256 | 4× overlap; hop = lag grid + update rate; hop 128 = +10–20 dB (scratch), hop 64 collapses |
| `n_g` | 8 | taps per bin; coverage = n_g·hop (reverb30 needs 32) |
| `solver` | 'cg' | `'rls'` = champion; `'cg'` = Chang & Willson one-step |
| `delta` | 0.1 (rls) / 1.0 (cg) | rls: P(0) = I/δ loading scale (sweet 0.1–0.3); cg: stride floor (1.0; <1 diverges on speech) |
| `lam` | 0.999 | rls forgetting (62 ms window; insensitive 0.995–0.9995) |
| `gradient` | 'correlation' | `'error'` for the CG hybrid — the class default is the TEACHING counterexample |
| `gamma` | 0.1 | error-gradient averaging (flat 0.05–0.2) |
| `beta` | 0.97 | correlation window (β=1.0 best; the correlation mode's only honest setting) |
| `k_max` / `internal` | 1 / 'true' | CG2 machinery (keep 1 in streaming) |
| `gate_rel` | 0.3 | freeze P and w when closed (anti-wind-up — mandatory for RLS) |

Recommended recipes: `FD_NLMS(nfft 512/128, constraint=True)` for the OLS
baseline; `CONJUGATE_FB_TOEPLITZ(nfft=1024, hop=256, solver='rls',
delta=0.1, lam=0.999)` for maximum ERLE.

## 10. How to run

```bash
.venv/Scripts/python test_subband_echo_cancellation.py fbtoe   # champion (44.68 dB)
.venv/Scripts/python test_subband_echo_cancellation.py fdnlms  # OLS baseline
.venv/Scripts/python run_fdnlms.py canonical --algo cgmdf --beta 0.3
.venv/Scripts/python run_fdnlms.py speechlp --algo fdnlms --preemph 0.95
.venv/Scripts/python run_fdnlms.py reverb30 --algo pfcg --delta 6
```

Outputs: residual WAV, spectrograms, per-band ERLE, learned-path metrics
(true peak position/amplitude) into the working directory; experiment
records live under `results/<date>_<label>/FINDINGS.md`.

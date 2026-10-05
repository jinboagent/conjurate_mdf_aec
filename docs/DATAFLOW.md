# Data Flow — Frequency Domain Adaptive Filters

> Visual guide to signal flow, buffer structures, and algorithm internals.
> Read this alongside `PROJECT_ARCHITECTURE.md` (what things are) and
> `HARNESS_ENGINEERING.md` (how to test them).

---

## Table of Contents
1. [End-to-End AEC Signal Flow](#1-end-to-end-aec-signal-flow)
2. [Overlap-Save Block Processing](#2-overlap-save-block-processing)
3. [Partitioned Buffer Structure](#3-partitioned-buffer-structure)
4. [PFDAF Data Flow (LMS Baseline)](#4-pfdaf-data-flow-lms-baseline)
5. [Canonical CONJUGATE_MDF Data Flow](#5-canonical-conjugate_mdf-data-flow)
6. [Weight Update Strategies](#6-weight-update-strategies)
7. [Sub-Hop Delays: Representation vs the Criterion Cliff](#7-sub-hop-delays-representation-vs-the-criterion-cliff)
8. [Echo Path Generation and Verification](#8-echo-path-generation-and-verification)
9. [Test Harness Flow](#9-test-harness-flow)
10. [Benchmark: 50 ms delay + lowpass pair](#10-benchmark-50-ms-delay--lowpass-pair-reference-target--20-db)

---

## 1. End-to-End AEC Signal Flow

```
                    PHYSICAL WORLD                          DIGITAL PROCESSING
                    ──────────────                          ──────────────────

 ┌────────────┐    ┌──────────────┐    ┌────────────┐     ┌──────────────────────────┐
 │            │    │              │    │            │     │                          │
 │ Loudspeaker│───→│ Room Acoustics│───→│ Microphone │────→│  ADC → Digital Signal    │
 │            │    │  (echo path   │    │            │     │                          │
 │  plays x   │    │   h[n])      │    │  picks up  │     │  d[n] = h[n] * x[n]     │
 │            │    │              │    │  d = h*x   │     │       (echo signal)      │
 └─────┬──────┘    └──────────────┘    └────────────┘     └───────────┬──────────────┘
       │                                                              │
       │  x[n] (reference)                                           │  d[n] (microphone)
       │                                                              │
       └──────────────────────┬───────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Adaptive Filter    │
                    │                    │
                    │  H(z) estimates    │
                    │  the echo path h   │
                    │                    │
                    │  y[n] = ĥ[n]*x[n] │──── echo estimate
                    │                    │
                    │  e[n] = d[n]-y[n]  │──── error (output)
                    │                    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Weight Update      │
                    │                    │
                    │  Uses x[n], e[n]   │
                    │  to improve H(z)   │
                    │  every block        │
                    └────────────────────┘

 GOAL: e[n] → 0  (all echo removed, only near-end speech remains)
```

### Signal Naming Convention

| Symbol | Name | Description |
|--------|------|-------------|
| `x[n]` | Reference | Far-end signal played through loudspeaker |
| `d[n]` | Desired/Mic | Microphone signal containing echo |
| `y[n]` | Echo estimate | Adaptive filter's estimate of the echo |
| `e[n]` | Error/Output | `d[n] - y[n]` — what remains after cancellation |
| `h[n]` | True echo path | Room impulse response (what we want to learn) |
| `ĥ[n]` | Estimated path | What the adaptive filter has learned |
| `H[k]` | Frequency weights | Filter coefficients in frequency domain |
| `X[k]` | Reference FFT | Frequency-domain reference signal |

---

## 2. Overlap-Save Block Processing

All partitioned-block algorithms use **overlap-save** to convert between
time-domain signals and frequency-domain processing.

### The geometry dial: L = R × M (configurable)

Two lengths define every block scheme in this project:

| Symbol | Meaning | Classic choice | This harness (current code) |
|--------|---------|----------------|------------------------------|
| `M` | block size = **hop** (frame advance) | 256 | **128** |
| `L` | frame length = FFT size | `2M` (50% overlap) | **512 = 4M** (75% overlap) |
| `R = L/M` | overlap factor | 2 | **4** |

The frame length is **any integer multiple of the hop**: `L = R·M`.
The classic textbook scheme uses `R = 2`; the current harness wrapper
(`CGMDF(fft_size=512, step=128)` in `test_subband_echo_cancellation.py`)
runs `R = 4`. Both are just settings of the same two knobs
(`fft_size`, `step`) — nothing in `CONJUGATE_MDF` itself knows `L` or `M`;
it only sees pre-transformed frames (§5).

> **Notation note:** this document uses `N_G` for the number of frequency
> partitions (filter length in frames) and `R` for the overlap factor, so a
> filter covers `(N_G + R − 2)·M` samples of delay in total (§7).

```
TIME DOMAIN                    FREQUENCY DOMAIN              TIME DOMAIN
───────────                    ────────────────              ───────────

Input signal x[n]:
┌───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │  ... blocks of M samples (the hop)
└───┴───┴───┴───┴───┴───┘

Frame t (R = 2 shown — the classic scheme):

                    ┌───────────────────────┐
  x_old (block t-1) │   x (block t)         │
  ┌───────────────┐ │ ┌───────────────┐     │
  │ M samples     │ │ │ M samples     │     │
  └───────┬───────┘ │ └───────┬───────┘     │
          └────┬────┘         │             │
               ▼              │             │
          ┌─────────────────┐ │             │
          │  L = 2M samples │◄┘             │
          │  [x_old | x]    │               │
          └────────┬────────┘              │
                   │                        │
                   ▼                        │
            ┌────────────┐                 │
            │  L-pt FFT  │  → X[k] (L/2+1 bins)
            └────┬───────┘                 │
                 ▼                         │
            Filtering Y = H·X (per bin)    │
                 ▼                         │
            ┌────────────┐                 │
            │  L-pt IFFT │                 │
            └────┬───────┘                 │
                 ▼                         │
      keep the LAST M samples ─────────────┘
      (the first L−M are circular garbage)
```

With `R = 2` the previous block is the whole history in the window; with
`R = 4` (the harness) the window reaches 3 blocks back — the regressor for
one weight spans `R` blocks, which is what makes the partition bookkeeping
in §7 subtle.

WHY an L ≥ 2M-point FFT?
─────────────────────────
Linear convolution of an M-tap filter with M new samples needs 2M−1 points.
With an L-point circular convolution, the first L−M output samples contain
wrap-around artifacts → discard them; the last M are the valid linear-convolution
result. Any `R ≥ 2` keeps this property (bigger R only adds overlap).

### Step-by-step example (M=4, 2M=8)

```
x_old = [a, b, c, d]     (previous block)
x     = [e, f, g, h]     (current block)

Concatenate: [a, b, c, d, e, f, g, h]  (8 samples)
          ↓
       FFT(8) → X[k], k=0..4  (5 frequency bins = M+1)
          ↓
       Y[k] = H[k] * X[k]    (element-wise multiply)
          ↓
       IFFT(8) → y[n], n=0..7  (8 time samples)
          ↓
       y_valid = y[4:8]  (last 4 samples — the valid ones)

Discarded (circular artifacts):  y[0], y[1], y[2], y[3]
Kept (valid linear convolution):  y[4], y[5], y[6], y[7]
```

---

## 3. Partitioned Buffer Structure

For long echo paths, a single block isn't enough. We split the filter into
**N partitions**, each covering M samples. Total coverage: N × M samples.

```
                    PARTITIONED BUFFER X_buf[N, M+1]
                    ─────────────────────────────────

  Time →

  Partition 0 (newest)     Partition 1              Partition N-1 (oldest)
  ┌──────────────────┐    ┌──────────────────┐     ┌──────────────────┐
  │ X_buf[0, :]      │    │ X_buf[1, :]      │     │ X_buf[N-1, :]    │
  │ FFT of current    │    │ FFT of previous   │     │ FFT of N-1       │
  │ block [x_old|x]  │    │ block             │ ... │ blocks ago       │
  │                  │    │                  │     │                  │
  │ covers delay:    │    │ covers delay:    │     │ covers delay:    │
  │ [0, M-1]        │    │ [M, 2M-1]       │     │ [(N-1)M, NM-1]  │
  └──────────────────┘    └──────────────────┘     └──────────────────┘

  Buffer update (every new block):
  ┌──────────────────────────────────────────────────────────┐
  │  X_buf[N-1] = X_buf[N-2]    (shift everything down)      │
  │  X_buf[N-2] = X_buf[N-3]                                 │
  │  ...                                                      │
  │  X_buf[1]   = X_buf[0]                                   │
  │  X_buf[0]   = FFT([x_old | x_new])   (insert new frame)  │
  └──────────────────────────────────────────────────────────┘

  In numpy:
    self.X[1:] = self.X[:-1]     # shift
    self.X[0]  = X               # insert

  Total echo path coverage (classic R = 2 geometry):
  N partitions × M samples/partition = N×M samples

  General L = R·M geometry (§2): consecutive lag frames are offset by M,
  but each lag's weight spans R blocks, so the representable delay range is
  (N_G + R − 2)·M samples  →  R=2: N_G·M (classic);
  R=4 harness: (N_G+2)·M, e.g. N_G=8, M=128 → 1280 samples = 80 ms.

  Example: N=64, M=256 → covers 64×256 = 16384 samples = 1024ms at 16kHz
```

### How partitions map to echo path delays

```
TRUE ECHO PATH h[n] (time domain):

  Amplitude
  │
  │        ╱╲
  │       ╱  ╲    ← direct path at sample 720 (45ms)
  │      ╱    ╲
  │     ╱      ╲  ╱╲
  │    ╱        ╲╱  ╲    ← early reflections
  │───╱──────────────╲───────╱╲╱╲───
  │                              ← late reverberation (decaying noise)
  └──────────────────────────────────────→ Time (samples)
  0    256   512   768  1024  1280  ...  8000
       │     │     │     │     │
       ▼     ▼     ▼     ▼     ▼
       P1    P2    P3    P4    P5       ← Partition assignments

  Partition 0: h[0:256]     → mostly silence (before direct path)
  Partition 1: h[256:512]   → silence
  Partition 2: h[512:768]   → CONTAINS DIRECT PATH PEAK at h[720]
  Partition 3: h[768:1024]  → early reflections
  Partition 4: h[1024:1280] → more reflections
  ...

  Each partition's weight H[p,k] models the frequency content
  of h[n] in the delay range [p×M, (p+1)×M-1].
```

---

## 4. PFDAF Data Flow (LMS Baseline)

`pfadf_mdf_cg.py` — the simplest partitioned-block algorithm in this project. Good reference for
understanding the structure before tackling the CG-MDF filter.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PFDAF filt(x, d) — FILTERING STEP                                         │
│                                                                             │
│  INPUT: x (M samples), d (M samples)                                       │
│                                                                             │
│  ①  x_now = concat(x_old, x)           # [2M] overlap-save input           │
│     X = rfft(x_now)                     # [M+1] frequency bins             │
│                                                                             │
│  ②  X_buf[1:] = X_buf[:-1]             # shift buffer                     │
│     X_buf[0] = X                        # insert new frame                  │
│     x_old = x                           # save for next block               │
│                                                                             │
│  ③  Y = Σ_p H[p] * X_buf[p]            # echo estimate in freq domain      │
│     y = irfft(Y)[M:]                    # extract last M (valid) samples    │
│                                                                             │
│  ④  e = d - y                           # error = desired - estimate       │
│                                                                             │
│  OUTPUT: e (M samples)                                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  PFDAF update(e) — WEIGHT UPDATE STEP                                      │
│                                                                             │
│  INPUT: e (M samples)                                                      │
│                                                                             │
│  ①  e_fft[M:] = e * hanning(M)         # windowed error in second half     │
│     E = rfft(e_fft)                     # [M+1] frequency bins             │
│                                                                             │
│  ②  X2 = Σ_p |X_buf[p]|²               # total power across partitions     │
│     G = μ * E / (X2 + ε)                # normalized gradient              │
│                                                                             │
│  ③  H += conj(X_buf) * G                # weight update (all partitions)   │
│                                                                             │
│  ④  TIME-DOMAIN CONSTRAINT (partial):                                      │
│     h = irfft(H[p_idx])                  # convert one partition to time   │
│     h[M:] = 0                            # zero the tail (causality)       │
│     H[p_idx] = rfft(h)                   # back to frequency domain        │
│     p_idx = (p_idx + 1) % N              # cycle to next partition         │
│                                                                             │
│  WHY THE CONSTRAINT?                                                        │
│  Without it, frequency-domain weights can drift into non-causal solutions. │
│  Zeroing h[M:] forces each partition to be an M-tap causal filter.         │
│  Partial constraint (one partition per block) saves computation.            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Computation per block

```
Operation                    Complexity
─────────                    ──────────
1 × 2M-point FFT             O(2M log 2M)
1 × 2M-point IFFT            O(2M log 2M)
N complex multiplications    O(N × (M+1))
1 error FFT                  O(2M log 2M)
N weight updates             O(N × (M+1))
1 partition constraint       O(2M log 2M) (2 FFTs)
─────────────────────────────────────────
Total:                       O((4 + 2N) × M log M)
```

---

> **Historical note (2026-10-01):** the legacy full-frame Toeplitz-CG
> criterion that this section previously described was REMOVED from
> `conjugate_mdf.py` (recoverable from git `2f23d9d`; records in
> results/2026-10-01_class-simplify/). The flow below is the CURRENT
> canonical class. A short summary of the removed criterion is in §5.5.

## 5. Canonical CONJUGATE_MDF Data Flow

`conjugate_mdf.py` — the classic partitioned-block MDF (Lee & Chang):
`[0;e]` criterion + normalized gradient + G-projection. Works on
pre-transformed frequency-domain inputs (no internal overlap-save);
`FD_NLMS` in the same module is the identical geometry with an
instantaneous gradient (`beta=0` reproduces it exactly, 1.6e-16).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONJUGATE_MDF.apply(Y, Y_rx) — ONE FRAME                                   │
│                                                                             │
│  INPUT: Y [NBIN, NCHAN] (mic), Y_rx [NBIN, Nrxref] (reference)              │
│         (already FULL L-sample frame FFTs from the external wrapper)        │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 1: REGRESSOR BUFFER (reference frames only; mic is not buffered)     │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  buf_Y_rx = roll(buf_Y_rx, -1, axis=1)     # shift lag buffer              │
│  buf_Y_rx[:, -1, :] = Y_rx                   # insert new frame at end     │
│  rx_flipped = flip(buf_Y_rx, axis=1)         # index j = frame lag          │
│                                                                             │
│  Buffer layout: [NBIN, N_G, Nrxref]                                         │
│    axis 0: frequency bins (NBIN = L/2+1)                                    │
│    axis 1: time partitions (0 = oldest … N_G-1 = newest after flip)        │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 2: A PRIORI ESTIMATE + THE [0;e] ERROR                                │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  est     = Σ_j w_last[j] (.) rx_flipped[j]   # estimate with OLD weights    │
│  R       = Y - est                           # full-frame residual spectrum │
│  e_tail  = irfft(R)[L-hop:]                  # residual of the NEW block    │
│  E_zh    = rfft([ zeros(L-hop) ; e_tail ])   # zero-headed error [0;e]      │
│  E_upd   = R  if full_frame_error else E_zh  # ablation flag (§7.4)         │
│                                                                             │
│  ─── THE CRITERION: the update grades ONLY the new block. The overlap-save │
│      head (circular wrap) never biases the weights — this is the cliff fix.│
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 3: REFERENCE-EXCITATION GATE                                          │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  adapt only while  P_ref > gate_rel·P_mic  AND  P_ref > gate_floor·P_run   │
│  (P_run = decaying running peak; gate_hold frames of hysteresis).           │
│  Covers reverb ring-down, digital silence, double talk. Ratio-based —       │
│  a pure level threshold cannot separate quiet speech from a reverb pause.   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 4: GRADIENT AVERAGING + NORMALIZER (per reference channel)            │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  gacc = β·gacc + conj(rx_flipped) (.) E_upd   # decay-first averaging       │
│  Pn   = β·Pn + (1-β)·Σ_j |rx_flipped|²        # smoothed per-bin power      │
│  (gate closed: gacc only decays — the stale gradient fades out)             │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 5: NORMALIZED UPDATE + G-PROJECTION (per bin, gate open)              │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  w[k] = w_last[k] + μ·(1-β)·gacc[k] / (Pn[k] + 1e-10)                       │
│  (silent bins Pn ≤ 1e-12: weight suspended at w_last)                       │
│  G-PROJECTION (Lee & Chang eq.10, unconditional):                           │
│    Wt = irfft(w);  Wt[hop:] = 0;  w = rfft(Wt)                              │
│  — each partition is a hop-tap causal filter; keeps the frequency           │
│    products exact linear convolutions.                                      │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 6: OUTPUT + STATE ADVANCE                                             │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  output = E_zh        # zero-headed A PRIORI error spectrum                 │
│  self.e  = e_tail     # the hop time samples the wrapper keeps              │
│  w_last  = w                                                # next frame    │
│  OUTPUT: rfft([0; e]) — IDENTICAL frame geometry to FD_NLMS                 │
│  (same buffers, same in/out; only the weight update differs).               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key difference from PFDAF

```
              PFDAF (LMS)                    CONJUGATE_MDF (canonical)
              ───────────                    ─────────────────────────
Input         Time-domain blocks             Frequency-domain frames
              (handles own FFT)              (external FFT)

Adaptation    Normalized gradient:           Normalized gradient with
              H += μ·conj(X)·E/(|X|²+ε)     β-averaged gradient + gate:
                                             w += μ(1-β)·gacc/(Pn+ε)
                                             (β=0 ⇒ exactly FD_NLMS)

Error         e*window, full frame           [0;e] — new block only
                                             (no criterion cliff, §7)

Constraint    Partial (one partition/frame)  Full G-projection every
                                             frame (all partitions)

Memory        Instantaneous power            β-averaged gradient + power
              (forgets after 1 block)        (exponential window)

Stability     Always stable                  Gate freezes adaptation when
                                             the reference is not identifiable
```

### 5.5 Historical: the removed legacy criterion (summary)

The pre-2026-10-01 class accumulated full-frame correlations
(`autoR`, `rcross` — both conjugated on the regressor side, the
`y = Σ w_j·X(t−j)` convention) and took per-bin CG steps on the
Toeplitz system `toeplitz(autoR)·w = rcross` with a positive-curvature
guard. Its criterion graded the WHOLE frame (head included), which is
exactly the cliff condition of §7.3 — the reason it was removed. The
full-frame mode is preserved as the `full_frame_error=True` ablation
flag (§7.4); the complete Toeplitz-CG machinery is in git `2f23d9d`.

---

## 6. Weight Update Strategies

### Strategy 1: Normalized LMS (PFDAF)

```
                    ┌─────────────┐
  E[k] ────────────→│             │
                    │  Normalize  │     G[k] = μ · E[k] / (X²[k] + ε)
  X²[k] ───────────→│  by power   │
                    │             │
                    └──────┬──────┘
                           │
  conj(X[k,p]) ───────────→│ × │────→ ΔH[k,p] = conj(X[k,p]) · G[k]
                           │
                           ▼
                    H[k,p] += ΔH[k,p]

  Characteristics:
  ─────────────────
  • Each partition updated independently
  • Step size inversely proportional to signal power
  • ε prevents division by zero
  • Simple, stable, but slow convergence
```

### Strategy 2: Toeplitz CG (the REMOVED legacy criterion — historical)

> Removed from `conjugate_mdf.py` on 2026-10-01 (git `2f23d9d`): its
> full-frame criterion is the cliff condition (§7.3). Kept here as the
> algorithm survey; the current CONJUGATE_MDF uses Strategy 1's update
> with β-averaging (§5).

```
                    ┌─────────────────────────────────────┐
                    │  Build Toeplitz matrix T from autoR  │
                    │                                      │
                    │  ┌                    ┐              │
                    │  │ R[0]  R[1]  R[2]   │              │
                    │  │ R[1]  R[0]  R[1]   │   (symmetric)│
                    │  │ R[2]  R[1]  R[0]   │              │
                    │  └                    ┘              │
                    │                                      │
                    │  This captures inter-partition        │
                    │  correlation — how partition p        │
                    │  relates to partition q               │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Compute gradient:                    │
                    │  g = rcross - T @ w_last             │
                    │                                      │
                    │  rcross = E[X·conj(Y)] (where to go) │
                    │  T @ w  = E[X·conj(X)]·w (where we   │
                    │           are in the solution space)  │
                    │  g = residual (correction needed)     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  CG step:                            │
                    │  α = (g^H · g) / (g^H · T · g)      │
                    │  w = w_last + clip(α, -1, 1) · g    │
                    │                                      │
                    │  Takes a step along the gradient,    │
                    │  scaled by the curvature of T        │
                    └─────────────────────────────────────┘

  Characteristics:
  ─────────────────
  • All partitions coupled through T
  • Uses accumulated statistics (autoR, rcross)
  • Faster convergence but can be unstable for large N_G
  • Toeplitz structure allows O(N²) solve (vs O(N³) for general matrix)
```

### Strategy 3: Conjugate Gradient (PFDAF-CG, `pfdaf_cg.py` — NOT WIRED)

> Status: `pfdaf_cg.py` crashes at init (missing `d_old`/`D` state) and is
> not used by any test; kept for reference only.

```
                    ┌─────────────────────────────────────┐
                    │  Maintain conjugate direction v:     │
                    │                                      │
                    │  v[k] = -g[k] + β · v[k-1]          │
                    │                                      │
                    │  where β = Hestenes-Stiefel:         │
                    │  β = ((g_k - g_{k-1})^H · g_k) /    │
                    │      (v_{k-1}^H · (g_k - g_{k-1}))  │
                    │                                      │
                    │  This avoids the "zigzag" of pure    │
                    │  gradient descent by choosing         │
                    │  conjugate (T-orthogonal) directions  │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  Step size:                          │
                    │  α = -(g^H · v) / (v^H · R · v)     │
                    │                                      │
                    │  H += α · v                          │
                    │                                      │
                    │  Converges in at most N steps        │
                    │  (vs many iterations for LMS)        │
                    └─────────────────────────────────────┘
```

---

## 7. Sub-Hop Delays: Representation vs the Criterion Cliff

> This section was rewritten 2026-09-24 after the white-noise oracle debug
> (`results/2026-09-24_noise-oracle-debug/`). The old text claimed "the delay
> is NOT representable" after switching to a 2M FFT — that is wrong in
> general: **representation is exact; the full-frame criterion is the
> blocker.** Figure: `docs/criterion_cliff.png` (reproduce with
> `scratch_cliff_figure.py`; single-delay probes: `scratch_delay_mechanism.py`).

### 7.1 What an L-point FFT weight can represent (exact, any delay)

Each partition weight `w[·, j]` is one complex number per bin — i.e. the DFT
of an **L-tap real FIR** `h_j`. Tap `m` of lag `j` paints input sample at
window position `n − m` (circularly) of frame `t−j`:

```
  frame t output sample n (absolute t·M + n)
  ◄──────────────────────────── L ─────────────────────────────►
                                                    ◄── tail (KEPT) ──►
  n: 0 ·········································· L−M ················· L
      ┌─────────────────────────────────────────────┬───────────────────┐
      │            head (overlap-save discards)     │   valid output    │
      └─────────────────────────────────────────────┴───────────────────┘
                    tap (j, m) contributes:
                      m ≤ n   →  delay d = j·M + m        (clean, causal)
                      m > n   →  delay d = j·M + m − L    (circular WRAP)

  So a pure delay d = j·M + r (0 ≤ r < L−M) is represented EXACTLY by a
  phase ramp in partition j — zeros in the head of the time-domain weight:

     h_j = δ[m − r]   ⇔   w[b, j] = exp(−j·2π·b·r / L)
```

Measured: delay 800 = 6·128+32, weight `w[b,6] = exp(−j2πb·32/512)` cancels
to **288.8 dB** on the kept tail — machine precision. Delay range covered:
`[0, (N_G+R−2)·M]` (§3). **Sub-partition delays are perfectly representable.**

### 7.2 Problem 1 (the real M-point issue): inter-block circular wrap

With an M-point FFT and hop M (no overlap-save head), each block convolves
*circularly*: a delay `r` inside a block wraps the first `r` output samples
of every block to input from the END of the same block. Sub-block delays
alias into wrong output samples — genuinely unmodelable. This is why
`L ≥ 2M` overlap-save exists at all (§2), and that fix is correct.

### 7.3 Problem 2 (the CLIFF): the full-frame criterion vs r ≠ 0

`CONJUGATE_MDF` fits the **whole L-sample frame spectrum** — head included
— even though only the last M samples are ever kept as output. For a delay
with `r = d mod M ≠ 0` the true weight's response in the head contains
circular-wrap spill:

```
  true delay 800 (L=512, M=128, r=32), one frame:

  n:   0     32                                  384              512
       │wrap │                                     └── KEPT tail ──┘
       ▼     ▼
  (6,32) → 288✗│──────── delay-800 echo ✓ ────────│──── ✓ ─────────┤
  (7,416)→ ───│── ✓ ────── delay-800 echo ────────│── 1312 ✗ ──────┤
              └── fixes [0,32) but corrupts [416,512): REMAINDER CHAIN

  every patch for the head wraps garbage into a new region; the chain
  closes ONLY for r = 0 (then tap m=0 paints the whole frame cleanly).
```

**Minimal worked example (L=8, M=2, delay d=3; `scratch_cliff_toy.py`):**
tap (j=1,m=1) is the true tap, tap (j=2,m=7) is its wrap-patch. Which
absolute input sample each tap reads per output n (t=0; target: abs = n−3):

```
        n=    0      1      2      3      4      5      6      7
  (1,1)  +5✗   -2✓    -1✓    0✓    +1✓    +2✓    +3✓    +4✓      (✗ = FUTURE)
  (2,7)  -3✓   -2✓    -1✓    0✓    +1✓    +2✓    +3✓   -4✗      (✗ = d=+11)
  LS  →  h1[1] = 0.495, h2[7] = 0.505  (the HALF-SPLIT: correct taps share
        amplitude; boundary samples n=0, n=7 keep ~half garbage each)
  per-sample residual: n=0: -3.1 dB, n=1..6: -46..-59 dB, n=7: -2.9 dB
  frame ERLE: 9.0 dB  — the cliff, in miniature.
```

The full-frame LS cannot give one tap amplitude 1 and the other 0: sample
n=0 needs only the patch, sample n=7 needs only the true tap, samples in
between need their sum = 1 → the compromise (0.5, 0.5) leaves every boundary
sample half-wrong. With L=512 the boundary segment is `[0,r)` (and the
patches' far edges), same compromise, ~10 dB.

The LS must compromise → systematic bias. Measured on the white-noise
oracle (batch per-bin LS = the class's β→1 fixed point):

| delay | r = d mod M | full-frame LS tail ERLE | ramp (tail-only) |
|-------|-------------|--------------------------|------------------|
| 640, 768, 1152… | 0 | **~295–298 dB** | ~300 dB |
| 800   | 32  | 10.3 dB (N_G = 8/12/16/24 — flat) | **288.8 dB** |
| 960   | 64  | 6.1 dB | ~300 dB |
| 736   | 96  | 5.7 dB | ~300 dB |

**The cliff depends only on `d mod M` — not on L, not on R, not on N_G.**
L = 512 (R=4) and L = 256 (R=2) give the SAME numbers at every delay
(docs/criterion_cliff.png, both panels). Increasing the overlap cannot fix
it; it only costs more frames per second.

**Criterion-only A/B (scratch_classic_criterion.py, delay 800, identical
pair-window geometry and regressors):** classic `[0;e]` criterion (error on
the new block only — what standard FDAF/FBLMS/MDF computes): optimum ≥
288.7 dB. Full-frame per-bin criterion (what `CONJUGATE_MDF`'s correlations
encode): 10.3 dB. Same data, same frames, same weights available — only the
objective differs.

### 7.4 Fixes

| Route | What | Status |
|---|---|---|
| **CONJUGATE_MDF** (current; canonical-only since 2026-10-01) | the classic `[0;e]` MDF — exact valid-region gradient, FD-NLMS normalization (`mu`, `beta`=gradient averaging), G-constraint built in, reference-excitation gate. No mode switch: `hop` is a plain required parameter; the legacy full-frame criterion was removed at user request (recoverable from git `2f23d9d`) and survives only as the `full_frame_error=True` ablation flag (added 2026-10-01, output geometry unchanged) | **implemented & PASS**: canonical pair **23.77 dB / corr 0.9982**; noise oracle 43.2 dB with the true path learned exactly; cliff check 45.2/44.3 dB at delays 640/800 (no gap); `beta=0` ≡ FD_NLMS **exactly** (1.6e-16) |
| Table 2 / PAES constrained criterion | full Table-2 machinery (circulant D_T from truncated rg + G̃-wrapped products, kmax>1, Polak-Ribière) — the deeper CG structure on top of the same valid-region criterion | future work on top of hop mode |
| Classic FDAF-NLMS ([0;e] error) | `FD_NLMS` in conjugate_mdf.py — same interface/geometry, simplest classic form | **verified no-cliff**: delay 800 reaches the 49 dB oracle ceiling (converged); the `full_frame_error=True` ablation of the SAME class cliffs — canonical pair 5.40 dB vs 23.77, synthetic delay-800 13.0 vs 44.3 (results/2026-10-01_hop-ablation/) |
| Unconstrained FDAF (no G-projection; Mansour & Gray UFLMS) | `--no-constraint` (FD_NLMS) — skip the per-partition irfft/rfft projection | **no cliff, and +2.4 dB on the canonical pair (26.16 vs 23.77)**: each partition keeps its wrap taps, which add sub-hop delay freedom; weights are no longer hop-tap causal filters |
| Sample-domain engine | NLMS / time-domain CG: no windowing → no cliff | NLMS reaches the 49 dB oracle ceiling |
| Hop-aligned data | delays that are multiples of M | exact even now (~300 dB) |

### 7.5 Configuration cheat-sheet (aligned with the code)

```
  wrapper knobs (test_subband_echo_cancellation.CGMDF):
      fft_size = L   (any R·M; current 512 = 4×128, classic 2M = 256)
      step     = M   (hop; current 128)
      n_g      = N_G (partitions; delay coverage (N_G+R−2)·M)

  CONJUGATE_MDF itself is agnostic: it sees NBIN = L/2+1 bins and N_G
  lagged frames; the geometry lives entirely in the wrapper's FFT/hop.
```

---

## 8. Echo Path Generation and Verification

Two generators exist; do not mix them up:

| generator | path shape | output pair | used by |
|---|---|---|---|
| `create_test_files.py` | **single tap**: 50 ms delay (800 samples), decay 0.4 | `audio/reference.wav` + `audio/microphone.wav` (~10 s speech) | `test_subband_echo_cancellation.py` (the official pair, §10) |
| `harness_template/ground_truth/generators.py` | **room model**: direct + early reflections + late reverb (RoomParameters below) | `ground_truth_reference.wav` + `ground_truth_microphone.wav` + true path | `test_echo_path_comparison.py` |

### Room-path generation flow (`generate_room_echo_signals`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  test_echo_path_comparison.py → ground_truth/generators.py               │
│                                                                          │
│  RoomParameters:                                                         │
│    sampling_rate  = 16000 Hz                                            │
│    filter_length  = 500 ms (8000 samples)                               │
│    initial_delay  = 45 ms  (720 samples)                                │
│    RT60           = 300 ms                                              │
│                                                                          │
│  GENERATION STEPS:                                                       │
│                                                                          │
│  1. Direct path                                                          │
│     h[720] = 0.6                                                        │
│     │                                                                    │
│     ▼                                                                    │
│  2. Early reflections (8 taps in [721, 1520])                           │
│     Amplitudes: 0.15 to 0.40 (random)                                  │
│     │                                                                    │
│     ▼                                                                    │
│  3. Late reverberation (filtered noise with exponential decay)           │
│     envelope[n] = A · exp(-3·ln(10)·(n-720) / (RT60·fs))               │
│     │                                                                    │
│     ▼                                                                    │
│  4. Secondary reflections (15 smaller taps)                             │
│     │                                                                    │
│     ▼                                                                    │
│  5. Fade-out envelope (last 10% of filter)                              │
│     │                                                                    │
│     ▼                                                                    │
│  true_path[0:8000]                                                      │
│                                                                          │
│  PLOT:                                                                   │
│  │       ╱╲                                                              │
│  │      ╱  ╲    ← direct path (720)                                      │
│  │     ╱    ╲                                                            │
│  │    ╱      ╲ ╱╲                                                        │
│  │───╱────────╲╱──╲──╱╲╱╲──╱╲─────                                      │
│  │                                ← late reverb (noise × decay)          │
│  └──────────────────────────────→ n                                       │
│  0   720     1520              8000                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Verification Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  VERIFICATION PIPELINE                                                   │
│                                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐            │
│  │ true_path   │   │ ref (speech) │   │ adaptive filter    │            │
│  │ (8000 samp) │   │ (160k samp)  │   │ (PFDAF or CG-MDF)  │            │
│  └──────┬──────┘   └──────┬───────┘   └─────────┬──────────┘            │
│         │                 │                      │                       │
│         │    ┌────────────▼───────────┐          │                       │
│         └───→│ mic = conv(ref, true)  │          │                       │
│              └────────────┬───────────┘          │                       │
│                           │                      │                       │
│                           │   ref ───────────────→│                       │
│                           │   mic ───────────────→│                       │
│                           │                      │                       │
│                           │              ┌───────▼───────┐               │
│                           │              │ e = filt(x,d) │               │
│                           │              │ filter.update  │               │
│                           │              │ (N frames)     │               │
│                           │              └───────┬───────┘               │
│                           │                      │                       │
│                           │              ┌───────▼───────┐               │
│                           │              │ est_path =     │               │
│                           │              │ get_echo_path()│               │
│                           │              └───────┬───────┘               │
│                           │                      │                       │
│  ┌────────────────────────┼──────────────────────┼──────────────────┐    │
│  │  METRICS               │                      │                   │    │
│  │                        ▼                      ▼                   │    │
│  │  ┌─────────────────────────────────────────────────────┐          │    │
│  │  │ SIGNAL LEVEL:                                        │          │    │
│  │  │   ERLE = 10·log10(Σmic² / Σe²)                      │          │    │
│  │  │   Target: ≥ 15 dB                                   │          │    │
│  │  └─────────────────────────────────────────────────────┘          │    │
│  │  ┌─────────────────────────────────────────────────────┐          │    │
│  │  │ ECHO PATH LEVEL:              true_path   est_path  │          │    │
│  │  │   NMSE       = 10·log10(Σ(est-true)² / Σtrue²)     │          │    │
│  │  │   Correlation = corrcoef(true, est)                  │          │    │
│  │  │   Coherence   = |Σ(H_est·conj(H_true))| / norms     │          │    │
│  │  │   Delay error = |argmax|est| - argmax|true||         │          │    │
│  │  │   Amp error   = 20·log10(|est_peak|/|true_peak|)    │          │    │
│  │  └─────────────────────────────────────────────────────┘          │    │
│  └───────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  PASS/FAIL:                                                              │
│    ERLE ≥ 15 dB       → echo cancellation adequate                       │
│    Correlation ≥ 0.9  → echo path shape correct                          │
│    Delay error = 0    → correct delay estimation                        │
│    Amplitude error < 3 dB → correct amplitude                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Test Harness Flow

### Block Wrapper Pattern

```
  TIME DOMAIN                    FREQUENCY DOMAIN               TIME DOMAIN
  ───────────                    ────────────────               ───────────

  ref[n] ──→ ┌──────────────────────────────────────────────┐
             │  CGMDF / FDNLMS (test_subband_echo_cancellation) │
             │  (identical wrapper; CLI keys 'cgmdf'/'fdnlms')  │
             │                                              │
             │  ┌─────────────┐    ┌─────────────────┐      │
  x_block ──→│  │  rfft(x)    │───→│                 │      │
  (fft_size) │  │  → X[k]     │    │  CONJUGATE_MDF  │      │
             │  └─────────────┘    │  .apply(D, X)   │      │
  d_block ──→│  ┌─────────────┐    │  (or FD_NLMS —  │      │
  (fft_size) │  │  rfft(d)    │───→│  same geometry) │      │
             │  │  → D[k]     │    │                 │      │
             │  └─────────────┘    └────────┬────────┘      │
             │                              │                │
             │                     ┌────────▼────────┐       │
             │                     │  irfft(E)        │      │
             │                     │  → e_time        │      │
             │                     │  take last       │      │
             │                     │  step samples    │      │
             │                     └────────┬────────┘       │
             └──────────────────────────────┼────────────────┘
                                            │
  e_block ──────────────────────────────────┘
  (step)

  CURRENT CODE GEOMETRY (configurable — §2, §7.5):
  ─────────────────────────
  fft_size = 512, step = 128  →  R = 4 (75% overlap)

  FRAME ADVANCE:
  ──────────────
  step = hop M (independent of fft_size; only constraint L = R·M)

  Each frame:
    Input:  fft_size = 512 samples (with 75% overlap from previous)
    Output: step = 128 valid samples

  Total frames for the 10 s canonical pair at 16 kHz
  (the wrapper loop advances by `step`, head = fft_size − step):
    n_frames = (160000 − 384) // 128 = 1247 frames
```

### Debug Funnel (When Things Go Wrong)

```
  ┌─────────────────────────────────────────────────────────┐
  │  Level 0: FULL SYSTEM                                    │
  │  Real speech, realistic echo path, N_G=64               │
  │  ERLE = -8 dB ← FAILURE                                 │
  │                                                          │
  │  ┌───────────────────────────────────────────────┐       │
  │  │  Level 1: SIMPLE ECHO PATH                     │       │
  │  │  h[n] = 0.3·δ[n-256] (single tap at P1)      │       │
  │  │  ERLE = 5.68 dB ← WORKS!                     │       │
  │  │                                                │       │
  │  │  ┌──────────────────────────────────────┐      │       │
  │  │  │  Level 2: NON-ALIGNED DELAY            │      │       │
  │  │  │  h[n] = 0.3·δ[n-720] (between P2,P3) │      │       │
  │  │  │  ERLE = -2 dB ← FAILS                  │      │       │
  │  │  │                                         │      │       │
  │  │  │  ┌──────────────────────────────┐       │      │       │
  │  │  │  │  Level 3: INSPECT INTERNALS   │       │      │       │
  │  │  │  │  Print Wiener solution at     │       │      │       │
  │  │  │  │  bin k=10:                    │       │      │       │
  │  │  │  │  w_opt puts energy at WRONG   │       │      │       │
  │  │  │  │  partitions → MODEL LIMITATION │       │      │       │
  │  │  │  └──────────────┬───────────────┘       │      │       │
  │  │  │                 │                        │      │       │
  │  │  │  ┌──────────────▼───────────────┐        │      │       │
  │  │  │  │  ROOT CAUSE (§7):            │        │      │       │
  │  │  │  │  full-frame criterion        │        │      │       │
  │  │  │  │  contaminates the fit when   │        │      │       │
  │  │  │  │  r = delay mod M ≠ 0         │        │      │       │
  │  │  │  │                              │        │      │       │
  │  │  │  │  FIX (implemented, §7.4):    │        │      │       │
  │  │  │  │  the [0;e] criterion —       │        │      │       │
  │  │  │  │  grade ONLY the new block    │        │      │       │
  │  │  │  │  (CONJUGATE_MDF / FD_NLMS);  │        │      │       │
  │  │  │  │  or sample-domain engine     │        │      │       │
  │  │  │  │  (NOT "use a bigger FFT" —  │        │      │       │
  │  │  │  │  the cliff is R-independent) │        │      │       │
  │  │  │  └──────────────────────────────┘        │      │       │
  │  │  └──────────────────────────────────────────┘      │       │
  │  └─────────────────────────────────────────────────────┘       │
  └─────────────────────────────────────────────────────────────────┘

  ALWAYS simplify first:
    Complex path → single tap → partition-aligned → check internals
```

---

## 10. Benchmark: 50 ms delay + lowpass pair (reference target ≥ 20 dB)

The canonical correctness pair: white-noise reference, mic = butter(2,0.3)
lowpass at 50 ms delay, gain 0.1 (`audio/noise_*.wav`,
`create_noise_echo.py`; true path peak 0.0365 @ sample 802). Both engines,
standard buffer processing (FFT 512 / hop 128, n_g=8, `[0;e]` +
G-constraint + excitation gate, official metric):

| configuration | ERLE | learned path |
|---|---|---|
| FD_NLMS raw frames | **43.2 dB** | **0.0365 @ 802 — exact** |
| CG-MDF hop (β=0.3) raw frames | **43.3 dB** | **0.0365 @ 802 — exact** |
| CG-MDF hop (β=0) | 43.2 dB | ≡ FD_NLMS digit-for-digit |

Reading: the target is exceeded by > 20 dB and the true path is learned
exactly (amplitude and position) — the overlap-save tail-keep buffer
processing is fully correct; raw frames are the recommended configuration
for this class.

Reproduce: `.venv/Scripts/python scratch_lowpass_compare.py`;
record: `results/2026-10-01_lowpass-benchmark/`.

**Same recipe on 30 s SPEECH** (`audio/speechlp_*.wav`: original_speech
tiled ×3, butter(2,0.3)@50 ms, gain 0.4; true peak 0.1459 @ 802):
FD_NLMS **21.7 dB** overall; CG-MDF hop **21.7 dB at β=0** (≡ FD_NLMS)
and **20.3 dB at β=0.05** — voiced seconds reach 47–57 dB, last-5-s
≈ 42 dB, and both learn the true path to 0.6 % amplitude / 1 sample
position. β guidance on speech: use β ≤ 0.05 for maximum performance
(the averaging knob trades average ERLE for mic-noise robustness);
per-second dips are speech pauses (mic = pure echo ⇒ noise-dominated
ratios), not failures. Reproduce: `scratch_speechlp_bench.py`;
generator: `create_noise_echo.py 50 0.4 0.3 audio/speech30.wav 30
speechlp`; record: `results/2026-10-01_speechlp-benchmark/`.

---

## Quick Reference: File → Algorithm → Data Flow Section

| File | Algorithm | See Section |
|------|-----------|-------------|
| `pfadf_mdf_cg.py` | PFADF MDF CG | [§4](#4-pfdaf-data-flow-lms-baseline) |
| `conjugate_mdf.py` | CONJUGATE_MDF (canonical `[0;e]`) + FD_NLMS | [§5](#5-canonical-conjugate_mdf-data-flow), [§7](#7-sub-hop-delays-representation-vs-the-criterion-cliff) |
| `pfdaf_cg.py` | PFDAF-CG (broken init, not wired) | [§6 Strategy 3](#6-weight-update-strategies) |
| `test_subband_echo_cancellation.py` | Block wrapper (keys: nlms / fdnlms / cgmdf) | [§9](#9-test-harness-flow) |
| `run_fdnlms.py` | CLI driver for FD_NLMS / CONJUGATE_MDF A/B runs | [§10](#10-benchmark-50-ms-delay--lowpass-pair-reference-target--20-db) |
| `test_echo_path_comparison.py` | Shared-harness comparison on the ground-truth room path | [§8](#8-echo-path-generation-and-verification) |

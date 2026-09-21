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
5. [Conjugate Gradient MDF Data Flow](#5-conjugate-gradient-mdf-data-flow)
6. [Weight Update Strategies](#6-weight-update-strategies)
7. [The 2M-Point FFT](#7-the-2m-point-fft)
8. [Echo Path Generation and Verification](#8-echo-path-generation-and-verification)
9. [Test Harness Flow](#9-test-harness-flow)

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

```
TIME DOMAIN                    FREQUENCY DOMAIN              TIME DOMAIN
───────────                    ────────────────              ───────────

Input signal x[n]:
┌───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │  ... blocks of M samples
└───┴───┴───┴───┴───┴───┘

Block n=2 processing:
                    ┌───────────────────────┐
  x_old (block 1)   │   x (block 2)         │
  ┌───────────────┐ │ ┌───────────────┐     │
  │ M samples     │ │ │ M samples     │     │
  └───────┬───────┘ │ └───────┬───────┘     │
          │         │         │             │
          └────┬────┘         │             │
               │              │             │
               ▼              │             │
          ┌─────────────────┐ │             │
          │  2M samples     │◄┘             │
          │  [x_old | x]    │              │
          └────────┬────────┘              │
                   │                        │
                   ▼                        │
            ┌──────────────┐               │
            │  2M-pt FFT   │               │
            │  X = rfft()  │               │
            │  (M+1 bins)  │               │
            └──────┬───────┘               │
                   │                        │
                   ▼                        │
            ┌──────────────┐               │
            │  Filtering   │               │
            │  Y = H * X   │               │
            └──────┬───────┘               │
                   │                        │
                   ▼                        │
            ┌──────────────┐               │
            │  2M-pt IFFT  │               │
            │  y = irfft() │               │
            └──────┬───────┘               │
                   │                        │
                   ▼                        │
            ┌──────────────────────┐       │
            │  Extract last M:     │       │
            │  y_valid = y[M:]     │       │
            │  (first M are        │       │
            │   circular garbage)  │       │
            └──────────┬───────────┘       │
                       │                    │
                       ▼                    │
                 y_valid (M samples) ──────┘

WHY 2M-point FFT?
─────────────────
Linear convolution of M-tap filter with M new samples needs 2M-1 points.
Using 2M-point FFT gives exact linear convolution via circular convolution.
The first M output samples contain wrap-around artifacts → discard them.
The last M samples are the valid linear convolution result.
```

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

  Total echo path coverage:
    N partitions × M samples/partition = N×M samples

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

## 5. Conjugate Gradient MDF Data Flow

`conjugate_mdf.py` — frequency-domain conjugate gradient with Toeplitz matrix adaptation.
Works on pre-transformed frequency-domain inputs (no internal overlap-save).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Conjugate Gradient MDF apply(Y, Y_rx) — COMPLETE FRAME PROCESSING         │
│                                                                             │
│  INPUT: Y [NBIN, NCHAN] (mic), Y_rx [NBIN, Nrxref] (reference)           │
│         (already in frequency domain from external FFT)                     │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 1: BUFFER UPDATE                                                     │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  buf_Y_rx = roll(buf_Y_rx, -1, axis=1)     # shift reference buffer       │
│  buf_Y_rx[:, -1, :] = Y_rx                   # insert new frame at end     │
│                                                                             │
│  buf_Y = roll(buf_Y, -1, axis=1)            # shift mic buffer            │
│  buf_Y[:, -1, :] = Y                          # insert new frame at end    │
│                                                                             │
│  Buffer layout: [NBIN, N_G, NCHAN/Nrxref]                                 │
│    axis 0: frequency bins                                                   │
│    axis 1: time partitions (0=newest, N_G-1=oldest)                        │
│    axis 2: channels                                                         │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 2: FILTERING (with w_last, previous iteration's weights)             │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  rx_flipped = flip(buf_Y_rx, axis=1)       # reverse: oldest first        │
│  micest = Σ_p w_last[p] * rx_flipped[p]    # echo estimate per bin        │
│  output = Y - micest                        # error signal                 │
│  e = output.copy()                                                         │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 3: CORRELATION UPDATE (accumulate statistics)                        │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  For each reference channel iref:                                           │
│    new_R1 = rx_flipped * conj(Y_rx)         # autocorrelation contribution │
│    autoR[iref] += α * new_R1                 # accumulate                   │
│                                                                             │
│    new_cross = rx_flipped * conj(Y)          # cross-correlation           │
│    rcross[iref] += α * new_cross             # accumulate                  │
│                                                                             │
│    autoR[iref] *= β                          # forgetting factor            │
│    rcross[iref] *= β                         # forgetting factor            │
│                                                                             │
│  autoR[k,p] ≈ Σ_n β^(N-n) * X[n] * conj(X[n-p])   (autocorrelation)      │
│  rcross[k,p] ≈ Σ_n β^(N-n) * X[n] * conj(Y[n-p])   (cross-correlation)   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 4: CG ADAPTATION (per frequency bin)                                 │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  For each frequency bin k:                                                  │
│    r_vec = autoR[k, :]                       # autocorrelation vector      │
│    T = toeplitz(r_vec)                       # build Toeplitz matrix       │
│                                                                             │
│    For each microphone channel:                                             │
│      g = rcross[k, :] - T @ w_last[k, :]    # gradient (residual)         │
│      p = g                                    # search direction            │
│      α = (p^H · g) / (p^H · T · p)          # CG step size               │
│      α = clip(α, -1, 1)                      # stability clamp             │
│      w[k, :] = w_last[k, :] + α * p          # weight update              │
│                                                                             │
│  ═══════════════════════════════════════════════════════════                │
│  STEP 5: RE-FILTER (with updated weights)                                  │
│  ═══════════════════════════════════════════════════════════                │
│                                                                             │
│  micest = Σ_p w[p] * rx_flipped[p]          # new echo estimate           │
│  output = Y - micest                         # updated error               │
│  w_last = w.copy()                           # save for next frame         │
│                                                                             │
│  OUTPUT: output [NBIN, NCHAN]                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key difference from PFDAF

```
              PFDAF (LMS)                    Conjugate Gradient MDF
              ───────────                    ──────────────────────
Input         Time-domain blocks             Frequency-domain frames
              (handles own FFT)              (external FFT)

Adaptation    Normalized gradient:           Toeplitz CG:
              H += μ·conj(X)·E/(|X|²+ε)     w += α·T⁻¹·(rcross - T·w)

Memory        Instantaneous power            Accumulated correlations
              (forgets after 1 block)        (exponential window via β)

Convergence   ~1/μ blocks                    ~N_G frames (faster)
Speed         O(N·M)                         O(N_G²·NBIN) per frame

Stability     Always stable                  Needs regularization
                                             (Toeplitz can be ill-conditioned)
```

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

### Strategy 2: Toeplitz CG (Conjugate Gradient MDF)

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

### Strategy 3: Conjugate Gradient (PFDKF-CG)

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

## 7. The 2M-Point FFT

### The Problem: M-point FFT Can't Model Sub-Partition Delays

```
With M-point FFT (M=256), each partition models exactly p×M samples delay:

  Partition 0 → delay 0
  Partition 1 → delay 256
  Partition 2 → delay 512
  Partition 3 → delay 768
  ...

  TRUE DELAY = 720 samples (45ms at 16kHz)

  ┌──────────────────────────────────────────────┐
  │  P2 (delay 512) ────────────┐                │
  │                              │ 720 ← TRUE    │
  │  P3 (delay 768) ────────────┘   GAP = 208!   │
  │                                              │
  │  The delay 720 is NOT representable.         │
  │  Energy splits across P2 and P3,             │
  │  and the CG solver assigns it to the         │
  │  wrong partition.                            │
  └──────────────────────────────────────────────┘
```

### The Solution: 2M-point FFT with Overlap-Save

```
With 2M-point FFT (512 for M=256):

  FFT size = 2M = 512 → frequency resolution = 2M bins
  Each partition still covers M samples, but the
  2M-point FFT provides phase information WITHIN each partition.
```

---

## 8. Echo Path Generation and Verification

### Generation Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  test_subband_echo_cancellation.py → create_echo() / LibriSpeech sample   │
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

  ref[n] ──→ ┌──────────────────────────────────────────────────────┐
             │  BlockRLSBishengMDF (test_subband_echo_cancellation) │
             │                                                      │
             │  ┌─────────────┐    ┌─────────────────┐             │
  x_block ──→│  │  rfft(x)    │───→│                 │             │
  (fft_size) │  │  → X[k]     │    │  RLSBishengMDF  │             │
             │  └─────────────┘    │  .apply(D, X)   │             │
  d_block ──→│  ┌─────────────┐    │                 │             │
  (fft_size) │  │  rfft(d)    │───→│  Returns E[k]   │             │
             │  │  → D[k]     │    │                 │             │
             │  └─────────────┘    └────────┬────────┘             │
             │                              │                       │
             │                     ┌────────▼────────┐             │
             │                     │  irfft(E)        │             │
             │                     │  → e_time        │             │
             │                     │  take last       │             │
             │                     │  step_size       │             │
             │                     │  samples         │             │
             │                     └────────┬────────┘             │
             │                              │                       │
             └──────────────────────────────┼───────────────────────┘
                                            │
  e_block ──────────────────────────────────┘
  (step_size)

  FRAME ADVANCE:
  ──────────────
  step_size = fft_size × (1 - overlap)
            = 256 × (1 - 0.75) = 64 samples

  Each frame:
    Input:  fft_size = 256 samples (with 75% overlap from previous)
    Output: step_size = 64 valid samples

  Total frames for 10s at 16kHz:
    n_frames = (160000 - 256) / 64 ≈ 2496 frames
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
  │  │  │  │  ROOT CAUSE:                 │        │      │       │
  │  │  │  │  M-point FFT can't model     │        │      │       │
  │  │  │  │  sub-partition delays        │        │      │       │
  │  │  │  │                              │        │      │       │
  │  │  │  │  FIX: Use 2M-point FFT       │        │      │       │
  │  │  │  └──────────────────────────────┘        │      │       │
  │  │  └──────────────────────────────────────────┘      │       │
  │  └─────────────────────────────────────────────────────┘       │
  └─────────────────────────────────────────────────────────────────┘

  ALWAYS simplify first:
    Complex path → single tap → partition-aligned → check internals
```

---

## Quick Reference: File → Algorithm → Data Flow Section

| File | Algorithm | See Section |
|------|-----------|-------------|
| `pfadf_mdf_cg.py` | PFADF MDF CG | [§4](#4-pfdaf-data-flow-lms-baseline) |
| `conjugate_mdf.py` | Conjugate Gradient MDF | [§5](#5-conjugate-gradient-mdf-data-flow) |
| `pfdaf_cg.py` | PFDAF-CG | [§6 Strategy 3](#6-weight-update-strategies) |
| `test_subband_echo_cancellation.py` | Block wrapper | [§9](#9-test-harness-flow) |

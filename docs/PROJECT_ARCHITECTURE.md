# Project Architecture — pdfaf_mdf Frequency Domain Adaptive Filters

## 1. Project Overview

This project implements and compares frequency-domain adaptive filtering algorithms for **Acoustic Echo Cancellation (AEC)**. The core problem: a loudspeaker plays a reference signal, a microphone picks up the echo through room acoustics, and an adaptive filter estimates and removes the echo.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AEC SYSTEM                                   │
│                                                                     │
│  Reference (x) ──→ [Loudspeaker] ──→ [Room Acoustics] ──→ [Mic]    │
│         │                                          │        │       │
│         │                    h (echo path)          │        │       │
│         │                         │                 │        │       │
│         └──→ [Adaptive Filter H] ──→ echo_est (y)   │        │       │
│                                    │                │        │       │
│                                    └──→ (-) ←───────┘        │       │
│                                          │                    │       │
│                                      error (e) ──→ output    │       │
│                                          │                    │       │
│                                          └──→ [Weight Update] │       │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. Algorithm Family Tree

```
                    Frequency Domain Adaptive Filters
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  Overlap-Save family        canonical MDF          PBFDAF-CG
  (FDAF baselines)          ([0;e] + beta avg)     (AES 2006)
        │                     │                     │
  conjugate_mdf.py       conjugate_mdf.py       pfdaf_cg.py
  (instantaneous grad)   (beta option = the former
                          CONJUGATE_MDF, merged 2026-10-06)
        │
        │  + WOLA filterbank family (second machine, 2026-10-06)
        │
  conjugate_fb_toeplitz.py
  (per-bin RLS/CG, key fbtoe — champion)

### Algorithm Descriptions

| Algorithm | File | Adaptation | Partitions | Use Case |
|---|---|---|---|---|
| **FD_NLMS** | `conjugate_mdf.py` | Normalized [0;e] gradient | N blocks | Classic FDAF baseline |
| **FD_NLMS (β option)** | `conjugate_mdf.py` | Instantaneous gradient; β>0 = gradient averaging (the former CONJUGATE_MDF, merged bit-exact 2026-10-06) + gate | N blocks | Reference implementation, cliff-free |
| **PBFDAF-CG** | `pfdaf_cg.py` | CG direction on averaged gradient + Gram line search | N blocks | Fast on longer signals |
| **FB-Toeplitz (RLS)** | `conjugate_fb_toeplitz.py` | Per-bin exponentially-windowed RLS in a WOLA filterbank | n_g subband taps/bin | **Champion — 44.68 dB** |
| **FB-Toeplitz (CG)** | `conjugate_fb_toeplitz.py` | Per-bin CG, error-gradient direction + Toeplitz curvature | n_g subband taps/bin | Runner-up (29.78), teaching A/B |
| **WOLA-NLMS** | ~~`conjugate_full.py`~~ | removed 2026-10-06 (10.66 dB; NLMS baseline covered by FD_NLMS) | — | — |

Note: the legacy full-frame `conjugate_toeplitz.py` (criterion-cliff
counterexample, 5.94 dB) was removed 2026-10-06 — code lives at git `2f23d9d`.
Theory: `docs/wola_vs_overlapsave.md`, `docs/autocorr_matrix_methods.md`.

## 3. Data Flow

### 3.1 Signal Processing Pipeline

```
Input Signals                    Algorithm Internals                     Output
─────────────                    ─────────────────                     ──────

                    ┌──────────────────────────────────┐
ref.wav ───────────→│  Overlap-Save Block Processing    │
(16kHz speech)      │                                   │
                    │  x_block = ref[i*M : i*M + 2M]   │──→ [2M-point FFT]
                    │  d_block = mic[i*M : i*M + 2M]   │──→ [2M-point FFT]
                    │                                   │
mic.wav ───────────→│  ┌─────────────────────────────┐ │
(convolved with     │  │  Partitioned Buffer (X_buf)  │ │
 echo path h)       │  │  [N, N_freq] complex         │ │
                    │  │  X_buf[0] = newest frame     │ │
                    │  │  X_buf[N-1] = oldest frame   │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────▼──────────────────┐ │
                    │  │  Echo Estimate               │ │
                    │  │  Y = Σ_p H[p] * X_buf[p]    │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────▼──────────────────┐ │
                    │  │  Overlap-Save Extraction     │ │
                    │  │  y = irfft(Y)[M:]  (last M)  │ │
                    │  │  e = d - y                   │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────▼──────────────────┐ │
                    │  │  Weight Update               │ │
                    │  │  (LMS / RLS / CG / Kalman)   │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    └─────────────┼─────────────────────┘
                                  │
                    output ───────┘
                    (echo-cancelled signal)
```

### 3.2 Weight Update Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WEIGHT UPDATE COMPARISON                          │
│                                                                     │
│  LMS/NLMS:     H += μ * conj(X) * E / (|X|² + ε)                  │
│                Simple, stable, slow convergence                    │
│                                                                     │
│  RLS:          H += step * T⁻¹ * (rcross - T * H)                 │
│                T = Toeplitz(autoR) — inter-partition correlation   │
│                Fast convergence, needs careful regularization       │
│                                                                     │
│  CG:           H += α * p;  p = g + β * p_prev                   │
│                Conjugate direction — avoids zigzag of gradient      │
│                                                                     │
│  Kalman:       H += K * E;  K = P * conj(X) / (X^H * P * X + σ²) │
│                Optimal for noisy environments                       │
│                                                                     │
│  PFADF MDF CG: H += μ * conj(X) * E / (X2 + ε)                   │
│                Normalized gradient, total power normalization      │
│                Stable, sub-partition delay resolution               │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Echo Path Generation and Verification

Echo paths can be generated using the `create_test_files.py` script, which creates reference and microphone WAV files from LibriSpeech audio samples. The echo path generation creates:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ECHO PATH PARAMETERS                                │
│                                                                     │
│  sampling_rate = 16000 Hz                                           │
│  echo_delay = 50 ms (800 samples)                                   │
│  echo_decay = 0.4 (-8 dB attenuation)                               │
│                                                                     │
│  1. Direct path: h[delay] = 1.0                                     │
│  2. Create microphone signal:                                       │
│     mic = delayed + attenuated copy of reference                    │
│                                                                     │
│  Metrics:                                                           │
│    ERLE          = 10*log10(mic_power / output_power)               │
│    NMSE          = 10*log10(Σ(est-true)² / Σ(true²))               │
│    Correlation   = corrcoef(est, true)                              │
│    Coherence     = |Σ(H_est * conj(H_true))| / (|H_est|*|H_true|) │
│    Delay error   = |argmax(|est|) - argmax(|true|)|                │
│    Amplitude err = 20*log10(|est_peak| / |true_peak|)              │
└─────────────────────────────────────────────────────────────────────┘
```

## 5. Key Files Quick Reference

### Core Algorithms
| File | Class/Function | Description |
|---|---|---|
| `FD_NLMS.py` | `FD_NLMS` | Canonical [0;e] MDF with the β gradient-averaging option (absorbed CONJUGATE_MDF 2026-10-06) |
| `pfadf_mdf_cg.py` | `PFADFMDFCG` | Inline CG-MDF with filt()/update() interface |
| `pfdaf_cg.py` | `PFDAF_CG`, `PFDAFCG` | PBFDAF-CG (AES 2006) — CG on the memory-averaged gradient; official key pfcg |
| `conjugate_fb_toeplitz.py` | `CONJUGATE_FB_TOEPLITZ` | WOLA filterbank + per-bin RLS (solver='rls', champion 44.68) or CG (error/correlation gradient); official key fbtoe |

### Test Scripts
| File | What it tests |
|---|---|
| `test_subband_echo_cancellation.py` | Official suite — keys: nlms / fdnlms / cgmdf / pfcg / fbtoe / wola |
| `test_echo_path_comparison.py` | Echo path estimation verification |
| `visualize_weight_convergence.py` | Weight convergence visualization |
| `create_test_files.py` | Generate test audio from LibriSpeech samples |

## 6. Critical Design Decisions

### Decision 1: 2M-point FFT vs M-point FFT

| Aspect | M-point FFT | 2M-point FFT |
|---|---|---|
| Frequency bins | M/2 + 1 | M + 1 |
| Delay resolution | Multiples of M only | Any position within partition |
| Overlap-save | Not possible | Standard technique |
| Used by | Original Conjugate Gradient MDF | PFADFMDFCG |

**Verdict**: 2M-point FFT is required for real-world echo paths with arbitrary delays.

### Decision 2: Weight Update Strategy

| Strategy | Stability | Convergence Speed | Complexity |
|---|---|---|---|
| Normalized gradient (PFDAF) | High | Medium | Low |
| Toeplitz RLS (Levinson) | Low (ill-conditioned) | Fast | High |
| CG with Toeplitz | Medium | Fast | High |
| Diagonal RLS | High | Medium | Low |

**Verdict**: Normalized gradient with total power normalization is the most robust. The Toeplitz-based CG is theoretically superior but practically unstable for large N_G.

### Decision 3: Regularization Factor ε

| Value | Use Case | Reference |
|---|---|---|
| delta (relative loading, eq. 22) | 0.5 | pfdaf_cg.py (AES 2006 stability constant) |
| mean(e²)/mean(x²) | Signal-proportional (optimal) | Haykin, REGULARIZATION_RESEARCH.md |

**Verdict**: Signal-proportional ε = mean(e²)/mean(x²) gives best results (ERLE: 23→70 dB on identity test).

## 7. Common Pitfalls

1. **FFT size mismatch**: CG-MDF filter expects M-point FFT bins but receives 2M-point FFT bins → wrong delay mapping
2. **CG step overshoot**: Single CG step on Toeplitz matrix overshoots for non-diagonal systems
3. **Rtoa initialization**: Must initialize to expected FFT power (2*M) not near-zero
4. **Accumulated vs instantaneous power**: Accumulated correlations grow unbounded, causing gradient explosion

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
              ┌───────────────┼───────────────────┐
              │               │                   │
         Single-Block    Partitioned-Block    RLS Bisheng
         (short echo)    (long echo)          MDF (advanced)
              │               │                   │
         ┌────┤          ┌────┼────┐         rls_bisheng_mdf.py
         │    │          │    │    │         (Toeplitz RLS)
        FDAF  FDKF     PFDAF │  PFDKF
        (LMS) (Kalman)  │  PFDAF-CG
                      PFADF-NLMS
```

### Algorithm Descriptions

| Algorithm | File | Adaptation | Partitions | Use Case |
|---|---|---|---|---|
| **FDAF** | `fdaf.py` | LMS (NLMS) | 1 (single block) | Short echo paths |
| **FDKF** | `fdkf.py` | Kalman | 1 (single block) | Short echo, noisy env |
| **PFDAF** | `pfdaf.py` | LMS (NLMS) | N blocks | Long echo paths |
| **PFDAF-CG** | `pfdaf_cg.py` | Conjugate Gradient | N blocks | Fast convergence |
| **PFDKF** | `pfdkf.py` | Kalman | N blocks | Long echo, noisy env |
| **PFADF-NLMS** | `pfadf_nlms.py` | NLMS | N blocks | Stable baseline |
| **RLS Bisheng MDF** | `rls_bisheng_mdf.py` | RLS (Toeplitz) | N blocks | Fastest convergence |
| **PFADF MDF CG** | `pfadf_mdf_cg.py` | Normalized gradient | N blocks | Stable + sub-partition delay |

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
                    │  │      * modulation(-1)^kp     │ │
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
│                T = Toeplitz(Rtoe) — inter-partition correlation    │
│                Fast convergence, needs careful regularization       │
│                                                                     │
│  CG:           H += α * p;  p = g + β * p_prev                   │
│                Conjugate direction — avoids zigzag of gradient      │
│                                                                     │
│  Kalman:       H += K * E;  K = P * conj(X) / (X^H * P * X + σ²) │
│                Optimal for noisy environments                       │
│                                                                     │
│  PFADF MDF CG: H += μ * conj(X) * E * modulation / (X2 + ε)      │
│                Normalized gradient with (-1)^(kp) modulation       │
│                Stable, sub-partition delay resolution               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 The 2M-point FFT and Modulation

With 2M-point FFTs and partition size M, partition p has delay p×M samples:

```
Delay of p*M in frequency domain:
  exp(-j * 2π * k * p*M / (2M)) = exp(-j * π * k * p) = (-1)^(k*p)

  k\p    0     1     2     3
  ─────────────────────────────
   0   [+1]  [+1]  [+1]  [+1]
   1   [+1]  [-1]  [+1]  [-1]
   2   [+1]  [+1]  [+1]  [+1]
   3   [+1]  [-1]  [+1]  [-1]
   ...
```

This modulation **must** be applied in both filtering and weight update.

## 4. Echo Path Generation and Verification

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ECHO PATH GENERATION FLOW                           │
│                                                                     │
│  echo_path_generator.py                                            │
│  ┌──────────────────────────────────────────────┐                   │
│  │ RoomParameters:                               │                   │
│  │   sampling_rate = 16000 Hz                   │                   │
│  │   filter_length = 500 ms (8000 samples)      │                   │
│  │   initial_delay = 45 ms (720 samples)        │                   │
│  │   RT60 = 300 ms                              │                   │
│  └──────────────┬───────────────────────────────┘                   │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────────────────────┐                   │
│  │ generate_time_domain_echo_path(seed=42)       │                   │
│  │                                               │                   │
│  │  1. Direct path: h[720] = 0.6                │                   │
│  │  2. Early reflections: 8 taps in [721, 1520] │                   │
│  │  3. Late reverberation: filtered noise decay  │                   │
│  │  4. Secondary reflections: 15 taps            │                   │
│  │  5. Fade-out envelope                         │                   │
│  └──────────────┬───────────────────────────────┘                   │
│                 │                                                    │
│                 ▼                                                    │
│  true_path (8000 samples)                                          │
│                                                                     │
│  ┌──────────────────────────────────────────────┐                   │
│  │ Create microphone signal:                    │                   │
│  │   mic = conv(ref, true_path)[:len(ref)]      │                   │
│  └──────────────┬───────────────────────────────┘                   │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────────────────────┐                   │
│  │ Run adaptive filter on (ref, mic)            │                   │
│  │   → Extract estimated echo path from weights │                   │
│  │   → Compare with true_path                   │                   │
│  └──────────────┬───────────────────────────────┘                   │
│                 │                                                    │
│                 ▼                                                    │
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

### Core Algorithms (implement these)
| File | Class/Function | Description |
|---|---|---|
| `rls_bisheng_mdf.py` | `RLSBishengMDF` | Primary RLS MDF with 2M-point FFT, modulation, normalized gradient |
| `pfadf_mdf_cg.py` | `PFADFMDFCG` | Inline RLS MDF with filt()/update() interface |
| `pfdaf.py` | `PFDAF`, `pfdaf()` | Partitioned-block LMS baseline |
| `pfdaf_cg.py` | `PFDAFCG` | Conjugate gradient variant |

### Echo Path Tools (generate and verify)
| File | Function | Description |
|---|---|---|
| `echo_path_generator.py` | `generate_time_domain_echo_path()` | Creates realistic room impulse responses |
| `echo_path_data.py` | `compare_echo_paths()` | Computes NMSE, correlation, coherence, delay error |

### Test and Debug (verify algorithms)
| File | What it tests |
|---|---|
| `test_compare_rls.py` | Fair RLS vs PFADF comparison (both 2M-point FFT) |
| `test_subband_echo_cancellation.py` | End-to-end NLMS vs RLS echo cancellation |
| `test_rls_bisheng_mdf.py` | Unit tests for RLS MDF |
| `debug_rls_*.py` | Various debug scripts for RLS convergence issues |

### Data Files
| File | Content |
|---|---|
| `original_speech.wav` | Clean reference speech (16kHz, 10s) |
| `ground_truth_echo_path.json` | Pre-generated echo path (8000 samples) |
| `ground_truth_echo.wav` | Pre-generated microphone signal |
| `ground_truth_reference.wav` | Pre-generated reference signal |

## 6. Critical Design Decisions

### Decision 1: 2M-point FFT vs M-point FFT

| Aspect | M-point FFT | 2M-point FFT |
|---|---|---|
| Frequency bins | M/2 + 1 | M + 1 |
| Delay resolution | Multiples of M only | Any position within partition |
| Modulation needed | No | Yes: (-1)^(kp) |
| Overlap-save | Not possible | Standard technique |
| Used by | Original RLS Bisheng MDF | PFADFMDFCG |

**Verdict**: 2M-point FFT is required for real-world echo paths with arbitrary delays.

### Decision 2: Weight Update Strategy

| Strategy | Stability | Convergence Speed | Complexity |
|---|---|---|---|
| Normalized gradient (PFDAF) | High | Medium | Low |
| Toeplitz RLS (Levinson) | Low (ill-conditioned) | Fast | High |
| CG with Toeplitz | Medium | Fast | High |
| Diagonal RLS | High | Medium | Low |

**Verdict**: Normalized gradient with total power normalization is the most robust. The Toeplitz-based RLS is theoretically superior but practically unstable for large N_G.

### Decision 3: Regularization Factor ε

| Value | Use Case | Reference |
|---|---|---|
| 1e-10 | Fixed, tiny (under-regularized) | pfdaf.py, pfadf_nlms.py |
| 1e-3 | Fixed, moderate | fdaf.py |
| 0.5 | Fixed, large (over-regularized) | pfdaf_cg.py |
| mean(e²)/mean(x²) | Signal-proportional (optimal) | Haykin, REGULARIZATION_RESEARCH.md |

**Verdict**: Signal-proportional ε = mean(e²)/mean(x²) gives best results (ERLE: 23→70 dB on identity test).

## 7. Common Pitfalls

1. **FFT size mismatch**: RLS filter expects M-point FFT bins but receives 2M-point FFT bins → wrong delay mapping
2. **Missing modulation**: 2M-point FFT requires (-1)^(kp) per partition → wrong echo estimate
3. **CG step overshoot**: Single CG step on Toeplitz matrix overshoots for non-diagonal systems
4. **Rtoa initialization**: Must initialize to expected FFT power (2*M) not near-zero
5. **Accumulated vs instantaneous power**: Accumulated correlations grow unbounded, causing gradient explosion

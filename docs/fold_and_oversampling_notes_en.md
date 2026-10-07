# The rig's fold trick · three kinds of "folding" disambiguated · oversampling conventions

2026-10-06. English version of
[fold_and_oversampling_notes.md](fold_and_oversampling_notes.md). Compiled
from the discussion of `polyphase_dft_fb_analysis.m` (principle figure:
[osfb_analysis_buffer.png](osfb_analysis_buffer.png)) and this project's WOLA
engine (`conjugate_fb_toeplitz.py`). Sister docs:
[wola_vs_overlapsave_en.md](wola_vs_overlapsave_en.md) (the full OLS-vs-WOLA
theory, Chinese original [wola_vs_overlapsave.md](wola_vs_overlapsave.md)).

---

## 1. What the rig computes, and where fold enters

Rig parameters: prototype length Lp = OS·nfft = 1024 taps, nfft = 512,
BLOCKSIZE = hop = 256, OS = 2. Each tick (every 256 samples) must produce all
257 bin outputs:

```
y_b[t] = Σ_{d=0}^{1023} p[d]·x[t−d]·e^{−j2πk d/512}
```

Naive direct computation: 257 channels × 1024 taps of complex MACs ≈
**2.1 Mflops/tick**.

![the fold trick](fold_trick.png)

The fold pipeline (row ② of the rig principle figure): **window → flip →
split mod nfft → pointwise add (fold) → ONE 512-point FFT**:

1. **After flip, index = delay d**: the newest sample sits at index 0, so
   "delays differing by 512" land exactly 512 positions apart;
2. **Why fold is legal**: the demodulation waveform e^{−j2πkd/512} has period
   nfft = 512 in d, so the terms at delays d and d+512 carry **identical
   modulation phase** — same-phase terms can be added BEFORE the transform
   (a distributive re-association), then one FFT finishes the job;
3. The identity (numerically verified to 1e-14; rig caption,
   scratch_osfb_fold_proof.py):

```
Σ_d p[d]x[t−d]·W^{kd}  =  Σ_{j=0}^{511} W^{kj} · Σ_m p[j+512m]x[t−j−512m]
                           └ 512-point DFT ┘   └─ fold: same-phase segment add ─┘
```

One sentence: **alias the DELAY axis mod nfft before the FFT** — the classic
**polyphase decomposition** of an oversampled DFT filterbank.

## 2. The trick's advantages and costs

| Advantage | Why |
|---|---|
| **① Filter length decouples from FFT cost** (the structural one) | FFT size locked at 512; the prototype can be as long as you like (steeper stopband, better neighbor suppression) at zero extra FFT cost — only the window multiply grows linearly. At OS=4 still one 512-FFT. **Filter quality becomes nearly free** |
| **② Everything downstream halves** | 257 bin streams instead of 513 — storage, per-bin adaptive filters, synthesis all halve; in AEC the O(nbin) part dominates |
| **③ Zero wasted spectral grid** | A direct 1024-FFT would deliver 513 bins on a fs/1024 grid, but the channels are spaced fs/512 — the 256 interleaved bins are wasted work; fold pre-merges same-phase terms and computes only the real channels |
| **④ Hardware/fixed-point friendly** | smaller FFT = fewer butterflies, less memory (the standard implementation argument of the Crochiere & Rabiner era) |

| Cost | Why |
|---|---|
| Loses the fine grid | only fs/512-spaced channels exist; designs that need fine bins (fbtoe leans on bin redundancy for the off-grid floor) lose something |
| Bookkeeping burden | flip/fold alignment, the ×OS replicate on synthesis, the nfft·blocksize "magic gain" (a MATLAB-ifft accounting constant) — all plumbing this trick requires |

Cost magnitudes per tick (order of, log scale): direct 2.1 Mflops → no-fold
1024-FFT route ~51 kflops → **fold + 512-FFT ~23 kflops** (2× cheaper than
the long-FFT route, ~90× vs direct).

## 3. Our engine: the OS=1 special case, fold is the identity

In `conjugate_fb_toeplitz.py` the prototype window q has length **exactly
nfft** (a 1024 window with a 1024-point rfft), so "split mod nfft" yields one
segment and fold degenerates to the identity:

```python
X = np.fft.rfft(self.q * x_frame)   # conjugate_fb_toeplitz.py:220 — window = FFT length
```

| | rig (polyphase_dft_fb_analysis) | ours (conjugate_fb_toeplitz) |
|---|---|---|
| Prototype length Lp | 1024 = OS·nfft | 1024 = nfft |
| FFT size | 512 | 1024 |
| fold | needed (OS=2 segment add) | **identity (one segment)** |
| Product | 257 coarse bin streams | 513 fine bin streams |

Two routes of the same coin: **the fold trick lets you tune "how long is the
prototype" (filter quality) independently of "how dense are the bins"
(channel count)**. We spent the budget on fine bins (15.6 Hz spacing —
neighbor redundancy and the off-grid floor benefit); the rig spent it on a
long prototype with coarse bins. Should a "steeper prototype + half the
channels" variant ever be wanted, the trick transfers directly.

## 4. Three kinds of "folding/overlap" disambiguated (the most confusable section)

| | OLS signal wrap | **the rig's analysis fold** | WOLA synthesis OLA |
|---|---|---|---|
| Object | the **signal** convolution values | the **filter/delay axis** | the **frame output stream** |
| Mechanism | modulo arithmetic wraps the frame tail onto the head | same-phase segments added pointwise | overlapping frame outputs added samplewise |
| Creates wrong values? | **yes** (values absent from the linear convolution) | no (exact to 1e-14) | no (this IS the perfect-reconstruction mechanism) |
| Consequence | head must be discarded / [0;e] | harmless, pure compute speedup | Σq² = k/2 constant, normalized by `_syn = 2/k` |

Litmus line: **wrap requires "two same-frame sequences in a circular
convolution"; fold merely "moves the addition before the FFT" (only
same-phase terms may be added); OLA is "place offset and add" (the COLA
condition guarantees flatness)**. The three are mutually distinct.
Visualization: [wola_folding.png](wola_folding.png) (top left: the OLS wrap;
top right: WOLA's per-bin FIR has no frame edge to wrap; middle: WOLA
synthesis 8-window additive coverage, Σq² = 4 ± 4e-16; bottom numeric proof:
zero-weight passthrough reconstruction 1.3e-15).

## 5. The two oversampling conventions (user's correction, 2026-10-06)

| Convention | Definition | the 1024/128 geometry |
|---|---|---|
| **Total redundancy (the filterbank standard)** | oversampling ratio = independent subbands / decimation ≈ **nbin/hop**; oversampled ⇔ nbin > hop | 513/128 ≈ **4×** |
| **Per-channel (STFT/WOLA engineering; the rig's k)** | how far each complex bin is sampled above its own Nyquist = **nfft/hop** | 8× |

For real signals the two differ by exactly the conjugate factor 2
(independent bins ≈ nfft/2). **Conversion: standard oversampling = nbin/hop
= k/2 = overlap factor / 2**; overlap percentage = (nfft−hop)/nfft = 1 − 1/k.

Re-labeled geometries (physical numbers unchanged, names only):

| geometry | nbin/hop (standard) | nfft/hop (= overlap factor) | verdict |
|---|---|---|---|
| rig 512/256 | ≈1 | 2 | **critically sampled** (in the standard convention; the old "oversampled" filename spoke the per-channel dialect) |
| harness official 512/128 | ≈2 | 4 | 2× oversampled |
| fbtoe wrapper 1024/256 | ≈2 | 4 | 2× oversampled |
| sweet spot 1024/128 | ≈4 | 8 | 4× oversampled |
| 1024/64 | ≈8 | 16 | 8× (still collapses — lag collinearity grows with nfft/hop; oversampling cannot save it) |

## 6. What is the decimation factor

**The decimation factor = hop**: the decimation stride of every subband
output sequence — the bandpass filters run continuously at fs, and 1 sample
per hop = 128 input samples is kept, so the subband rate = fs/hop = 125 Hz.
In code it is simply the stride `process()` advances each frame
(`conjugate_fb_toeplitz.py:211`; the block length the driver loop passes in).
Critical-sampling contrast: M = R is critical (MDCT audio coding, to save
bits); we deliberately run M/R ≈ 4× redundant to buy three things AEC wants:
repeated coverage of neighbor-band leakage (lower off-grid floor), a finer
lag grid (d mod hop representable), and a higher adaptation rate (fs/hop
updates per second).

## 7. Cheat sheet

- fold = polyphase decomposition: alias the delay axis mod nfft + one
  nfft-FFT, exact to 1e-14
- fold exists only when Lp > nfft (prototype longer than the FFT); ours has
  Lp = nfft — no fold
- three foldings: OLS wrap (harmful, the only one) ≠ rig fold (harmless) ≠
  WOLA OLA (PR itself)
- standard oversampling = nbin/hop (>1 to be called oversampled);
  k = nfft/hop is the per-channel dialect = 2× the standard
- hop = decimation factor = lag grid = the inverse-rate of adaptation = the
  unit of coverage (n_g·hop)

**References**: Crochiere & Rabiner, *Multirate Digital Signal Processing*,
1983 (WOLA / polyphase decomposition); Vaidyanathan, *Multirate Systems and
Filter Banks*, 1993; this repo:
[osfb_analysis_buffer.png](osfb_analysis_buffer.png) (rig principle),
[fold_trick.png](fold_trick.png) (this note's figure),
[wola_folding.png](wola_folding.png),
[wola_vs_overlapsave_en.md](wola_vs_overlapsave_en.md).

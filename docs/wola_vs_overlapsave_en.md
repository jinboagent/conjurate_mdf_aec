# WOLA (Filterbank) vs Overlap-Save: Two Different Machines

2026-10-06. English version of [wola_vs_overlapsave.md](wola_vs_overlapsave.md) (same content, same day, same verified numbers). It answers three questions in increasing depth:

1. Are WOLA and Overlap-Save two different methods? (Yes — and sharper than "two methods": they are two different *machines*.)
2. Is WOLA the same thing as Overlap-Add? (**No — this is the biggest naming trap**, see §6.1.)
3. Is "frequency-domain filtering" the essential difference between Overlap-Add and Overlap-Save? (**No.** Between OLA and OLS themselves there is no essential difference at all — they are two bookkeeping schemes of the same machine. "Frequency-domain filtering" actually separates the **fast-convolution family (OLA/OLS)** from the **filterbank family (WOLA)**, see §4.)

Code references: OLS machine = `FD_NLMS.py` (formerly conjugate_mdf.py; CONJUGATE_MDF was merged into FD_NLMS as its β option on 2026-10-06) and `pfdaf_cg.py`; WOLA machine = `conjugate_fb_toeplitz.py` (the NLMS sibling `conjugate_full.py` was removed 2026-10-06; its measurement is archived). All numbers were re-run on the official benchmark pair `test_subband_echo_cancellation.py` (canonical: 800-sample delay + 0.40 gain) on the day of writing.

---

## 0. One-page summary

| | Fast-convolution family | Filterbank family |
|---|---|---|
| Representatives | Overlap-Save, Overlap-Add | WOLA (weighted overlap-add) |
| Role of the FFT | **Computational tool**: speeds up a time-domain convolution | **The filterbank itself**: bin b IS the output of the b-th bandpass filter |
| Where the filter lives | Time domain (one full FIR; the spectrum is only its representation) | Subband domain (one complex FIR per bin, along the subband time axis) |
| Meaning of the spectral product X·W | One circular convolution (= linear convolution minus head / with zero-padding) | Never happens (there is no "same-frame spectrum × same-frame spectrum" product) |
| Circular-aliasing head | Exists, must be handled | **Structurally absent** |
| [0;e] criterion | **Required** for OLS adaptation (discard head / stale region) | **Nowhere to insert, and not needed** |
| Weight array | `W [nbin, N_G]` — each column is a cross-bin spectrum (G projection couples bins) | `w [nbin, n_g]` — each row is bin b's own taps (zero cross-bin coupling) |
| Where the criterion is scored | Time-domain frame: irfft → e frame → [0;e] → rfft | Subband-domain scalars: `E_b[m] = D_b[m] − Σⱼ w_b[j]X_b[m−j]` |
| Analysis/synthesis windows | None (rectangular frames + save/discard bookkeeping) | sqrt-Hann on both sides + PR reconstruction |
| Causal latency | nfft − hop | nfft − hop (same formula) |
| Official ERLE | NLMS 22.27 / FD_NLMS 23.77 / CGMDF 23.77 / PFCG 26.11 | WOLA-NLMS 10.66 / **FB-Toeplitz 44.68 (champion, RLS solver)** |

One sentence: **the OLA/OLS difference is "how block edges are accounted for"; the fast-convolution-vs-WOLA difference is "what the FFT actually is".** "Frequency-domain filtering" (per-band subband filtering) belongs to the latter alone.

---

## 1. The fast-convolution family: OLA and OLS are the same thing

### 1.1 Both accelerate the same time-domain FIR

Let the filter be a time-domain FIR `h[0..M−1]` (in this repo's context, the echo-path estimate), and we want `y = x * h` (linear convolution). Directly, each output sample costs M multiply-accumulates; the FFT turns it into a pointwise spectral product at O(nfft·log nfft) per block.

Key insight: **in both OLA and OLS the frequency domain is only an accelerator; the filtering semantics live in the time domain from start to finish.** The spectrum `W = FFT(h)` is not "the filter" — it is just h written another way. The two methods compute **the same** linear convolution and are provably identical (Appendix A, 1.8e-15 numerically). Their entire difference is **how block edges are stitched together**.

### 1.2 Overlap-Add: zero-pad + add the tails

```
input:        |—L—|—L—|—L—|            (non-overlapping blocks)
              ↓ zero-pad each block to N ≥ L+M−1
block conv:   y_i = IFFT( FFT(x_i padded) · FFT(h padded) )   ← length N
output:       |—L—|
                 ＼＼＼
                  |—L—|               ← block i's tail (M−1 samples)
                   ＼＼＼                 lands in block i+1's output region
                    |—L—|                 → ADD
y = y_0 + y_1 + y_2 + ...   (placed at offsets i·L and accumulated)
```

Because of the zero-padding to N ≥ L+M−1, each block's circular convolution **is already** the linear convolution — no aliasing, no head to discard. Block i's output tail extends past its own L samples into the next block's territory, and adding reconstructs the continuous convolution exactly. OLA is aliasing-free from start to finish; it never even needs to discard a head.

### 1.3 Overlap-Save: sliding frames + discard the head

```
sliding frames (advance L per step, frame length N, containing M−1 history samples):
   frame:  [old x … old x | new x new x … new x]
            ←– M−1 –→   ←––– L new samples –––→

circular convolution:  y_circ = IFFT( FFT(frame) · FFT(h padded) )
           y_circ[0 … M−2]   ← head: wraps around (garbage, discard)
           y_circ[M−1 … N−1] ← tail: = linear convolution (keep, length N−M+1 = L)
```

The frame is **not** zero-padded (history samples occupy the front), so `FFT(frame)·FFT(h)` followed by IFFT is a **circular** convolution: the frame's tail wraps around into its head, contaminating the first M−1 samples; the rest equals the linear convolution. That is where "Save" comes from: save (keep) the history inside the frame, **discard the aliased head**, keep only the last L clean samples.

(Textbook single-filter OLS slides by L = N−M+1; this repo's FDAF is its heavily-overlapped variant: stride hop = N/4, see §3.)

### 1.4 Where the head comes from (one paragraph)

For two sequences of lengths A and B, the circular convolution (length max(A,B)) differs from the linear convolution (length A+B−1) in that the linear result's overflowing B−1 tail samples are **wrapped back** to the beginning by the modulo arithmetic. OLS does not zero-pad (B fills the whole frame), so the head is unavoidable; OLA zero-pads (N ≥ L+M−1), so the overflow region does not exist. **The head is not "an error" — it is a mathematical artifact of the bookkeeping choice** — but once an adaptation criterion *scores* it, it becomes a real problem (§5).

### 1.5 OLA vs OLS: only the bookkeeping differs

| | Overlap-Add | Overlap-Save |
|---|---|---|
| Input blocks | Non-overlapping, zero-padded | Sliding and overlapping, not zero-padded |
| Aliasing of the circular convolution | None (removed by zero-padding) | Yes (head of M−1 samples) |
| Edge handling | **Add** the tails | **Discard** the head |
| Output | Each block contributes L samples (plus an M−1 tail) | Each block contributes its last L samples |
| Result | Linear convolution | Linear convolution (the same one) |

The two are alternative block-partition schemes of the same linear convolution, with no performance or semantic difference — so "which is essentially better, OLA or OLS" is an empty question. **The real question is one level deeper: when this machine is used for adaptation, the head stops being a mere bookkeeping artifact** (§5).

---

## 2. The filterbank family: WOLA

### 2.1 Here the FFT is not an accelerator — it IS the filterbank

The WOLA machine has no "time-domain FIR, then transform" step. Bin b of the FFT **is** the output of the b-th bandpass filter:

- Bandpass filter b's impulse response = the prototype window q modulated to center frequency `b·sr/nfft`;
- `X_b[m] = rfft(q · frame_m)[b]` = that filter's output at time m, decimated by hop (subband sample rate sr/hop);
- The filterbank's passband shape = the frequency response of q → adjacent bands overlap → this overlap is absorbed by the **oversampling factor k = nfft/hop** and the **perfect-reconstruction design** (§2.4).

This is an identity, not an analogy: expanding "window → FFT → advance by hop" bin-by-bin into the time domain yields exactly 513 bandpass filters, each convolving and then decimating (repo verification: docs/osfb_bandpass_view.png — a single tone lands in its bin; docs/osfb_analysis_buffer.png — the analysis-side flip+fold+FFT matches per-filter time-domain convolution sample-by-sample to 1e-14).

### 2.2 The three indices: b, m, j

The adaptive weights `w [nbin, n_g]` = `[513, 8]` mean (`conjugate_fb_toeplitz.py:195`):

```
        frequency axis b = 0…512 (513 mutually independent complex FIRs)
                ↓
bin b's subband sequence:  … X_b[m−3]  X_b[m−2]  X_b[m−1]  X_b[m]
                     ↕          ↕         ↕        ↕
                   w_b[3]     w_b[2]    w_b[1]   w_b[0]     ← lag axis j = 0…7
Ŷ_b[m] = Σⱼ w_b[j]·X_b[m−j]
```

- **b (bin)**: which FIR. Each one only sees the narrow band ~sr/nfft around its center frequency.
- **m (subband tick / frame number)**: this FIR's clock. m advances by 1 = full-rate time advances by hop.
- **j (subband lag / tap index)**: tap j handles the path segment from j ticks ago (≈ j·hop full-rate samples).

`X_b[m−j]` comes from the shift register `buf_X` (`conjugate_fb_toeplitz.py:226-228`, rolled once per frame) — **genuinely different subband instants** of real history, not any wrap-around within a frame.

### 2.3 The per-bin FIR is true linear convolution — the root of "no head"

`Ŷ_b[m] = Σⱼ w_b[j]·X_b[m−j]` differs from an ordinary FIR only in that it acts on decimated subband sequences with complex coefficients. It is an ordinary convolution — **circularity has nowhere to occur**:

- No two same-frame spectra are ever multiplied (one side is a scalar weight, the other a stream of history samples);
- No frame-level IFFT-then-keep-the-tail action (the error is formed directly in the subband domain, §5.2);
- Bins are independent → 513 small 8×8 systems `T_b·w_b = rcross_b`, solved per bin, zero coupling between bins.

### 2.4 How the path is represented: hop grid + complex taps + PR

The full-rate path h[n] (e.g. 2048 real taps) projects into the subband domain as ≈ one n_g-tap complex FIR per bin:

- **Coverage** = n_g·hop = 8×256 = 2048 samples, matching the full-rate path length;
- **Delay grid** = hop: the true delay d is first absorbed "in bulk" (a bulk delay shifts every subband identically), and the remainder `d mod hop` must be expressed by subband lags — so `d mod hop ≠ 0` falls between grid points and must be interpolated by adjacent complex taps (the off-grid floor: a soft −2.5 dB slope at 4× oversampling, versus a −13 dB cliff under the legacy OLS full-frame criterion);
- **Phase**: bin b's subband signal carries an `e^{j2πb·hop·m/nfft}`-type rotation, which complex taps absorb naturally;
- **Inter-band leakage**: the analysis window is non-ideal, so bin b contains neighboring-band components. Strictly independent per-bin weights cannot handle cross-band coupling — that bill is paid by three layers: oversampling redundancy (at k=4 every frequency is covered by 4 bins), the matched synthesis window (the PR design cancels the aliasing), and the complex taps' degrees of freedom. The rigorous fix is a cross-band weight matrix (multichannel Gram — the PFCG direction), not needed in this project so far.
- **Perfect reconstruction**: the analysis window q and synthesis window q together give q² = Hann, which satisfies COLA (the overlap sum Σₘ q²[n−m·hop] = k/2, a constant) → with no filtering, the output equals the input exactly. That is where `_syn = 2/(nfft//hop)` comes from (`conjugate_fb_toeplitz.py:185`; this constant was once wrong (1 instead of k/2), doubling the output stream at 4× — the signature was a PR probe reading exactly 0.0 dB).

### 2.5 Where the name WOLA comes from

**W**eighted **O**verlap-**A**dd: the synthesis side places each frame's `q·irfft(E)` into an accumulator, sliding by hop and **adding the overlaps** (`conjugate_fb_toeplitz.py:389-391`); "weighted" refers to the windows on both sides. It comes from the multirate signal processing classics (Crochiere & Rabiner, 1983) — and it is the **same name as, but a different thing from**, the fast-convolution OLA of §1.2; see §6.1.

---

## 3. The two machines are astonishingly isomorphic — the difference collapses to three axes

Looking at the OLS family in its actual form here (partitioned FDAF, N_G=8), its multiplication structure is **isomorphic** to WOLA's:

```
FD_NLMS (conjugate_mdf.py:369-378):          WOLA (conjugate_fb_toeplitz.py:226-232):
buf_Y_rx ← roll(past frame spectra), push Y_rx    buf_X ← roll(past frame spectra), push X
est_b = Σ_p W[b,p]·Y_rx[b, m−p]              Ŷ_b  = Σ_j w[b,j]·X_b[m−j]
       └── per-bin, cross-frame-lag convolution ──┘   └── per-bin, cross-frame-lag convolution ──┘
```

Both sides are "one cross-frame-lag FIR per bin", with lag stride hop on both sides. **The real difference reduces to exactly three axes**, each independently switchable:

| # | Axis | OLS family | WOLA family |
|---|---|---|---|
| 1 | **Where the error is formed and scored** | Dragged back to a **time-domain frame**: `irfft(R)` → e frame → [0;e] → rfft (`conjugate_mdf.py:379-384`) | Left in the **subband domain**: `E_b[m] = D_b[m] − Ŷ_b[m]`, a scalar scored directly (`:232`) |
| 2 | **Windows** | None (rectangular frames; zero windowing) | q on both sides + PR reconstruction |
| 3 | **Geometry of the weights** | Full spectrum columns `W[:,p]`: irfft back to time = a hop-length filter; G projection (`:419-423`) couples bins | Per-bin scalar rows `w[b,:]`: no cross-bin operation of any kind |

(In conversation I once shorthand-ed OLS as "same-frame spectral multiply" — that is the textbook single-filter form; the partitioned form is as above, and the two families are isomorphic. This refinement makes the distinction sharper, not blurrier: **the difference is not in the multiplication structure, but in the error-formation domain, the windows, and the weight geometry.**)

---

## 4. The core question: is "frequency-domain filtering" the essential OLA/OLS difference?

**No.** In two layers:

**Layer 1: between OLA and OLS there is no essential difference.** They are two bookkeeping schemes of the same fast-convolution machine (§1.5), numerically identical (Appendix A). In both, "the frequency domain" is only a way to compute faster — the filter is a time-domain FIR, the spectral product equals the convolution, and that is all.

**Layer 2: "frequency-domain filtering" is the boundary between the fast-convolution family and the filterbank family.** If "frequency-domain filtering" means "the filter itself lives in the frequency/subband domain and each band is filtered independently" (WOLA's way), then it is precisely what OLS/OLA **do not have**:

- OLS/OLA: **time-domain filter, frequency-domain acceleration**. `W = FFT(h)` irfft'd back must reproduce the same h — the spectrum is a representation, not a home.
- WOLA: **the filter lives in the subband domain**. `w_b[j]` has no per-bin meaning as "that time-domain FIR" (transforming a row `w[b,:]` back to the full rate does not yield the path; transforming a column `w[:,j]` yields a cross-bin time-domain shape — the object G projection governs, not the filtering action itself). The signal's home is also the subband domain: the error, the gradient, and the Toeplitz systems are all formed there; the time domain appears only at the entrance (analysis) and the exit (synthesis).

**Litmus test (works in one move)**: check whether the weight array survives reinterpretation.

- OLS `W [257, 8]`: `irfft` of one column → a 512-point time-domain filter (support ≤ hop under the G constraint). "Each column is a time-domain filter" ✓.
- WOLA `w [513, 8]`: one row, `w[b,:]`, is bin b's subband FIR — it acts on the **decimated subband sequence**, not the full-rate signal. No row or column irffts back to "that" time-domain echo path. ✗

So the accurate one-liner: **"frequency-domain filtering" (per-band subband filtering) is the essential difference between WOLA and the entire fast-convolution family (both OLA and OLS); between OLA and OLS themselves, there is not even a difference — only bookkeeping.**

---

## 5. Why [0;e] exists on only one of the two machines

### 5.1 The head's precondition: same-frame product + time-domain scoring

The **necessary and sufficient scenario** for a circular-aliasing head: multiply two spectra of the same frame, then IFFT to time. The OLS family does exactly this every frame:

```
R = Y − Σ_p W_p ⊙ Y_rx(p frames ago)     ← pointwise spectral product (same-frame spectrum × same-frame spectrum)
e_frame = irfft(R)                        ← time-domain frame
```

The G constraint compresses each partition filter's time-domain support to ≤ hop → each frame's circular aliasing contaminates only the first hop−1 samples; the middle of the frame is clean but belongs to **old data already emitted**; only the last hop is "clean and fresh". Hence:

- **Output side**: emit the last hop (`conjugate_mdf.py:379`) — discard the head, universal;
- **Adaptation side**: [0;e] zeroes the rest of the e frame before FFT-ing it as the gradient (`:381-384`) — the aliased head is garbage and must not be scored, and the old region would double-count.

**What happens without [0;e] (scoring the whole frame)**: garbage from the head leaks into the criterion, varying with delay, and the adaptation fits it. Measured (the former `conjugate_toeplitz.py` engine, deleted 2026-10-06; code recoverable from git 2f23d9d): at delay mod hop ≠ 0, ERLE collapses to **5.94 dB** (the cliff); at perfectly aligned delays it is actually better (113 dB — no head to get wrong). That is the criterion cliff — full geometry in docs/criterion_cliff.png and docs/cliff_geometry.png, frame geometry in docs/overlap_save_head.png and docs/window_placement.png.

### 5.2 WOLA: the head has nowhere to exist

WOLA's error **never becomes a time-domain frame**: `E_b[m] = D_b[m] − Σⱼ w_b[j]X_b[m−j]` is a subband-domain scalar, scored, differentiated, and solved directly. The time domain appears only at the exit — the synthesis side windows and overlap-adds the error spectrum `irfft(E)` into the output stream (`conjugate_fb_toeplitz.py:387-392`), which is **reconstruction**, not scoring.

No "same-frame spectral product followed by IFFT" action → no circular convolution → no head → **[0;e] has no place to be inserted in this machine**. This is not "we choose not to score the head" — the head, as an object, does not exist (structural, not a rule).

### 5.3 The three defense layers

| Layer | Governs | OLS family's approach | WOLA family's approach |
|---|---|---|---|
| **Criterion layer** (what is scored) | Head/stale region must not enter the gradient | [0;e] (explicit rule) | Per-bin scalar error (structurally headless) |
| **Path layer** (what the weights can represent) | Circular weights must not pollute the path | G projection / constraint (cross-bin) | PR window design + complex taps (no cross-bin operation) |
| **Solver layer** (how it is solved) | Step size / direction | NLMS, CG (FD_NLMS β option, PFCG) | Per-bin NLMS, subband Toeplitz-CG (FB-Toeplitz) |

The three layers are **mutually independent**, with experimental evidence: unconstrained + [0;e] = 26.16 dB > constrained 23.77 (turning the path layer OFF helps — wrap taps give sub-hop alignment freedom); WOLA needs neither rule. The counterexample (the former `conjugate_toeplitz.py`, deleted) turns the criterion layer off (full-frame scoring) → 5.94 dB, proving the criterion layer is where the cliff lives.

---

## 6. FAQ (the four knots people get tangled in)

### 6.1 Is WOLA's "OLA" the fast-convolution Overlap-Add? — No!

| | Fast-convolution OLA (§1.2) | WOLA's synthesis OLA (§2.5) |
|---|---|---|
| Purpose | Speed up a **time-domain convolution** | **Reconstruct** the subband error stream to time |
| What is multiplied | FFT(x block)·FFT(h) | Nothing (E is already the result spectrum) |
| Window | None (rectangular, relies on zero-padding) | Synthesis window q (relies on weighting) |
| What it suppresses | No aliasing to suppress | Inter-band aliasing, by the PR design |

The two OLAs share only the action "add overlapping output blocks". WOLA ≠ OLA + a window; **WOLA is not in the fast-convolution family at all**. Likewise, asking "does WOLA need [0;e]?" is a category error — [0;e] is a patch for the fast-convolution family's adaptation side; WOLA doesn't have the disease.

### 6.2 Per-bin independent filtering loses inter-band information — is that a problem?

The analysis window is non-ideal → each bin contains neighboring-band components → strictly, independent per-bin weights are an approximation of the true path. The approximation error is absorbed by oversampling redundancy (larger k = more bins covering each frequency) + PR synthesis + complex taps. At k=4 the official pair gives FB-Toeplitz 29.78 dB (correlation 0.9995), showing the approximation is good enough at this geometry. The rigorous fix is a cross-band weight matrix (bin×bin coupling) at the cost of a multichannel Gram — PFCG's direction, not the current bottleneck.

### 6.3 Why does the intra-bin conditioning worsen with oversampling?

hop is the lag grid: the larger k = nfft/hop, the more collinear the adjacent-lag subband regressors (at 4×, adjacent-lag correlation ρ≈0.76; white-noise true condition ~4, but the lag-direction condition number ~173). This is the **price of expressiveness** (a finer grid is more accurate), not a disease: the solver's δ·P curvature floor compresses the effective condition number to ~1+1/δ (δ=1 degrades to the NLMS step at worst), and FB-Toeplitz is nevertheless the champion at 4×.

### 6.4 Delay, grid, coverage — how do they reconcile?

- The bulk delay ⌊d/hop⌋ is absorbed by a shift common to every subband (external alignment in the harness, or subband lags);
- The remainder d mod hop falls between lag-grid points → the off-grid floor (a soft −2.5 dB slope at 4×);
- The path must satisfy length ≤ n_g·hop to be representable (canonical 50 ms = 800 samples << 8×256 = 2048 ✓);
- Both machines share the causal latency formula: nfft − hop (OLS 512/128 → 384 samples = 24 ms; WOLA 1024/256 → 768 = 48 ms).

---

### 6.5 The two oversampling conventions (user correction, 2026-10-06)

| Convention | Definition | This doc's geometry (1024/128) |
|---|---|---|
| **Total redundancy (the filterbank standard)** | oversampling ratio = independent subbands / decimation ≈ **nbin/hop**; oversampled ⇔ nbin > hop | 513/128 ≈ **4×** |
| **Per-channel (STFT/WOLA engineering; this doc's k, following the rig)** | how far each complex bin is sampled above its own Nyquist = **nfft/hop** | 8× |

For real signals the two differ by exactly the conjugate factor 2 (independent bins ≈ nfft/2). Conversion: **standard oversampling = nbin/hop = k/2 = overlap factor / 2**. All "k = nfft/hop" statements in this doc convert by that rule; the rig 512/256 is **critically sampled** (≈1×) in the standard convention — its former filename oversampled_filterbank_*.m spoke the per-channel dialect (renamed to polyphase_dft_fb_*.m). All measured physics (cond, collinearity, rates, coverage) are functions of the geometry itself and are unaffected by naming.

---

## 7. Repo engine landscape (re-run on 2026-10-06) (re-run on 2026-10-06)

| Engine | Machine | Criterion layer | Solver layer | canonical ERLE | Verdict |
|---|---|---|---|---|---|
| NLMS (time domain) | No FFT (gold standard) | Per-sample | NLMS | 22.27 | PASS |
| FD_NLMS | OLS | [0;e] + G projection | NLMS | 23.77 | PASS |
| CGMDF (FD_NLMS's β option; the former CONJUGATE_MDF merged into FD_NLMS 2026-10-06) | OLS | [0;e] + G projection | β=0 ≡ FD_NLMS (suite); run_fdnlms β=0.3 → 24.47 | 23.77 | PASS |
| PFCG (PFDAF_CG) | OLS | [0;e] (γ-averaged error gradient) | Per-bin Gram + CG | 26.11 | PASS |
| CONJUGATE_TOEPLITZ | OLS | **Full-frame (no [0;e])** | Long-window Toeplitz-CG | 5.94 | FAIL (cliff counterexample; **deleted 2026-10-06**) |
| WOLA-NLMS (CONJUGATE_FULL) | WOLA | Per-bin, headless | Per-bin NLMS | 10.66 | FAIL (NLMS step starvation at 4×; **deleted 2026-10-06** — the NLMS baseline role is covered by FD_NLMS) |
| **FB-Toeplitz (CONJUGATE_FB_TOEPLITZ)** | WOLA | Per-bin, headless | Per-bin RLS (solver='rls', λ=0.999, δ=0.1; runner-up: CG error gradient 29.78) | **44.68** | **PASS (champion)** |

How to read it: the machine (OLS/WOLA) does not decide the outcome — the **criterion × solver** combination does. On the same WOLA machine: NLMS solver 10.66, RLS solver 44.68. Within the same OLS family: [0;e] 23.77, full-frame 5.94. CONJUGATE_TOEPLITZ existed precisely to prove that removing [0;e] alone (not anything else) causes the cliff (the engine was deleted on 2026-10-06; the measurement is archived in results/2026-10-06_conjugate-toeplitz/FINDINGS.md).

Code map:

- OLS machine: `conjugate_mdf.py` (FD_NLMS: error formation :379-384, [0;e], G projection :419-423), `pfdaf_cg.py`
- Full-frame counterexample: `conjugate_toeplitz.py` (**deleted 2026-10-06**; was a legacy restoration bit-identical to 2f23d9d, recoverable from that commit; `_herm_toeplitz_matvec` now vendored into conjugate_fb_toeplitz.py)
- WOLA machine: `conjugate_fb_toeplitz.py` (analysis :218-223, shift register :226-228, subband error :232, solve :266-383, synthesis :387-392, _syn :185); `conjugate_full.py` (**deleted 2026-10-06**, `_sqrt_hann` vendored into fb_toeplitz)
- Figures: docs/overlap_save_head*.png (OLS head / discard), docs/window_placement.png (frame geometry), docs/criterion_cliff.png + docs/cliff_geometry.png (the cliff), docs/osfb_analysis_buffer.png + docs/osfb_bandpass_view.png (the filterbank view)

---

## Appendix A: OLA ≡ OLS ≡ linear convolution (verified)

`scratch_ols_ola_demo.py`, random x(64), h(9), N=16, L=8:

```
max |OLS - linear| = 1.78e-15
max |OLA - linear| = 1.78e-15
```

Core code (excerpt):

```python
y_lin = np.convolve(x, h)[:len(x)]            # gold standard

# OLA: non-overlapping blocks + zero-pad + add tails
for i in range(0, len(x), L):
    blk = np.zeros(N); blk[:L] = x[i:i+L]
    y_ola[i:i+N] += np.fft.irfft(np.fft.rfft(blk) * np.fft.rfft(h, n=N))[:N]

# OLS: sliding frames (with M−1 history) + full-frame product + discard head M−1
for i in range(0, len(x), L):
    frame = np.zeros(N)
    tail = x[max(0, i-(M-1)):i+L]; frame[N-len(tail):] = tail
    yf = np.fft.irfft(np.fft.rfft(frame) * np.fft.rfft(h, n=N))
    y_ols[i:i+L] = yf[-L:]
```

Two "different machines" produce the same result as `np.convolve` — the numerical proof of §1's "OLA and OLS are the same thing". The WOLA family has no such identity to write down: its filter does not live in the time domain, so there is no "that h" to compare against (§4's litmus test).

## Appendix B: References

- Allen & Rabiner, *A Unified Approach to Short-Time Fourier Analysis and Synthesis*, Proc. IEEE 1977 (the unified STFT/WOLA view)
- Crochiere & Rabiner, *Multirate Digital Signal Processing*, 1983 (multirate WOLA filterbanks)
- Shynk, *Frequency-Domain and Multichannel Adaptive Filtering*, IEEE SP Magazine 1992 (FDAF / circular convolution and constraints survey)
- Chang & Willson, *Analysis of Conjugate Gradient Algorithms for Adaptive Filtering*, IEEE Trans. SP 2000 (FB-Toeplitz solver layer)
- PAES/Eneman et al. 2006 (the γ-averaged error gradient φ = γφ + (1−γ)·conj(X)E)
- This repo: docs/DATAFLOW.md §7 (criterion-cliff theory), results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md (all WOLA-family experiments)

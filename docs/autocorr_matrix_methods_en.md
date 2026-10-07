# Four uses of the autocorrelation matrix: correlation / error / PFCG / RLS

2026-10-06. English version of [autocorr_matrix_methods.md](autocorr_matrix_methods.md).
It answers two questions: (1) is the error gradient already inside PFCG
(`pfdaf_cg.py`)? — **yes**; (2) among the methods that use the autocorrelation
(Toeplitz) matrix, what actually differs? — **the matrix's JOB, not its form**.
Companion figure: [autocorr_matrix_roles_en.png](autocorr_matrix_roles_en.png)
(Chinese-labeled original: [autocorr_matrix_roles.png](autocorr_matrix_roles.png)).
Sister docs: [wola_vs_overlapsave_en.md](wola_vs_overlapsave_en.md),
[fold_and_oversampling_notes_en.md](fold_and_oversampling_notes_en.md).

---

## 1. Direct answer: the error gradient was already inside PFCG

`PFDAF_CG.apply()`, step 4 (`pfdaf_cg.py:178-194`; García Morales 2006 eq. 18):

```python
gacc[iref] = γ·gacc[iref] + (1−γ)·conj(rx)·E_zh     # Φ: γ-averaged instantaneous [0;e] error gradient
R[iref]    = γ·R[iref]    + (1−γ)·conj(rx)·rxᵀ      # Gram: averaged with the SAME γ
```

The direction Φ **is** the γ-averaged instantaneous error gradient — the same
unbiased family as fb-toeplitz's error mode (`φ = γφ + (1−γ)·conj(rx)·E`), the
same origin (PAES/AES 2006), and neither is in Chang & Willson (TSP 2000)
(the paper's method = the correlation residual, see §5).

Six fine differences between PFCG and fb-toeplitz error mode:

| Axis | PFCG | fb-toeplitz error |
|---|---|---|
| Machine / criterion | OLS + **[0;e]** (time-frame error) | WOLA + **per-bin scalar** (structurally headless) |
| Matrix form | **FULL Gram** R (Q×Q, no Toeplitz projection) | **Toeplitz projection** T = toeplitz(autoR) |
| Time constants | gradient and R share **the same γ** (the paper's N-block average, aging together) | direction γ=0.1, matrix β=0.999 (**decoupled** long window) |
| Curvature floor | P_b = trace(R)/N_G (averaged scalar) + alpha_max clip | current-frame per-bin power X2 (vector; δ=1 → NLMS worst-case step) |
| Inner iterations | k_max model iterations g ← g − α·R·v (streaming still prefers 1) | k_max>1 diverges (frozen-T mismatch) |
| Path layer | G projection [I_hop,0] (every frame) | none needed (per-bin scalar weights) |

## 2. One-page summary (panel ① of the figure)

| Method | Direction from | Matrix's job | Form / upkeep | canonical ERLE |
|---|---|---|---|---|
| correlation (fb-toeplitz) | windowed system residual g = rcross − T·w | **defines the TARGET + stride** (target T⁻¹rcross wanders every frame) | Toeplitz projection, β long window | 9.35 (best β=1, δ=8) |
| error hybrid (fb-toeplitz) | instantaneous error gradient φ (zero mean at the true path: unbiased) | **stride only** ⟨v,Tv⟩ + δ·P_inst floor | Toeplitz projection, β window, γ independent | 29.78 |
| PFCG (pfdaf_cg) | γ-avg instantaneous [0;e] gradient Φ (same unbiased family) | **stride only** + k_max model iterations | FULL Gram, no projection, same γ | 26.11* |
| RLS (fb-toeplitz, solver='rls') | exact Newton K·ξ | **everything**: P = R⁻¹ exact inverse, direction+stride in one shot | FULL Gram, Sherman-Morrison rank-1 | 44.68 |

*PFCG at OLS 512/128 geometry, the rest at WOLA 1024/256 — not a strict
same-machine A/B.

## 3. The four update rules side by side

**① correlation** (class default; `gradient='correlation'`)

```
T  = toeplitz(autoR)                     autoR ← β·autoR + α_acc·conj(rx)Xᵀ
g  = rcross − T·w                        ← T appearance #1: defines the target (it wanders)
v  = g (+ β_cg·v_prev)
α  = 0.999·⟨g,v⟩ / (⟨v,T·v⟩ + δ·P̄·‖v‖²)   ← T appearance #2; P̄ = window-average power (scalar)
w += α·v
```

**② error hybrid** (`gradient='error'`)

```
T  as above (β=0.999 long window)
φ  ← γ·φ + (1−γ)·conj(rx)·E              ← T absent; E = D − Σ wⱼrxⱼ (unbiased direction)
v  = g (+ β_cg·v_prev)
α  = 0.999·⟨g,v⟩ / (⟨v,T·v⟩ + δ·P_inst·‖v‖²)  ← T's ONLY appearance; P_inst = current-frame power (vector)
w += α·v
```

**③ PFCG** (`pfdaf_cg.py`; OLS + [0;e] + G projection)

```
R  ← γ·R + (1−γ)·conj(rx)rxᵀ             ← FULL Gram, same γ as the gradient
Φ  ← γ·Φ + (1−γ)·conj(rx)·E_[0;e]        ← direction; E_[0;e] = rfft([0; irfft tail])
g  = Φ (restart each frame; k_max inner iterations g ← g − α·R·v)
α  = −⟨g,v⟩ / (⟨v,R·v⟩ + δ·P_b·‖v‖²), clipped to alpha_max·sd   ← R here (+ model iterations) only
w += α·v → G projection [I_hop,0]
```

**④ RLS** (`solver='rls'`; Haykin ch. 9)

```
K  = P·conj(u) / (λ + uᴴPu)              ← P = R⁻¹ maintained by Sherman-Morrison rank-1
P  ← (P − K·(uᴴP)) / λ
w += K·ξ                                  ← direction + stride exact in one shot (Newton); ξ = a-priori error E
```

## 4. Core thesis: the matrix's JOB decides, its form is innocent

The dissection experiments (FINDINGS Addendum 5) proved the Toeplitz
projection itself is nearly exact: on real data, bin 100, final window,
**‖R−T‖/‖R‖ = 0.37%**, cond(T) ≈ cond(R) ≈ 220 (the two heatmaps in the
figure are visually indistinguishable). The frozen exact solve of that same
T system = **45.4 dB** — even correlation mode's "noisy system" has an
excellent fixed point. So the entire 35 dB spread across the four methods
(9.35 → 44.68) comes from **what job the matrix is given**:

- **Define the target** (correlation): the target = T⁻¹·rcross; rcross is
  rewritten by b-noise every frame, and cond≈220 amplifies the displacement
  220× → the direction points at a wandering spot. Damping cannot save it
  (δ 1→8 buys only +0.4 dB, Addendum 8) — the direction itself is wrong.
- **Stride only** (error / PFCG): the direction comes from the unbiased
  instantaneous error gradient (zero mean at the true path); matrix errors
  change only the stride length (speed), never the destination.
- **Exact inverse** (RLS): P = R⁻¹ maintained recursively + fading loading
  δ·λⁿ + gate as anti-wind-up → direction and stride both exact → one step
  is worth hundreds of CG frames.

## 5. Numbers and history

Numbers (figure ③): correlation 9.35 (best β=1, δ=8; speech 20 s asymptote
14.46; stationary on-grid asymptote 38.08@64s — the one bright spot) <
PFCG 26.11 ≈ error 29.78 < RLS 44.68; ceiling 45.4 (frozen exact solve).
Full official suite (2026-10-06): nlms 22.27 / fdnlms 23.77 / cgmdf 23.77 /
pfcg 26.11 / **fbtoe 44.68 champion**.

History: Chang & Willson (2000) correlation-CG is the starting point (Table I
implemented equation by equation; its "eigenvalue-spread-independent
convergence" verified on the identity probe) → its direction exposed on
streaming speech → PAES 2006's instantaneous error gradient (PFCG's Φ,
fb-toeplitz's φ) takes over the direction while the matrix is demoted to
stride → the dissection proves the matrix form is innocent and the
trajectory is everything → RLS (Sherman-Morrison + fading loading +
anti-wind-up gate) ends the contest. The full-frame correlation
counterexample (cliff, 5.94 dB) was deleted on 2026-10-06 (code at git
2f23d9d).

**References**: Chang & Willson, IEEE TSP 2000 (CG1/CG2/Table I); García
Morales et al., AES 2006 (PBFDAF-CG, eqs. 8/10/18/22); Haykin, *Adaptive
Filter Theory* ch. 9 (RLS); this repo:
results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md (Addenda 4–8, all
experiments).

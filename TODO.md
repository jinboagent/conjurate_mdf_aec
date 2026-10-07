# TODO — Linear AEC Project

> Updated 2026-10-06 after the WOLA/RLS cycle (docs/OVERVIEW.md is the
> front-door technical document).
> Status quo: official suite 5 keys, all engines healthy —
> **fbtoe (per-bin RLS) 44.68 dB / corr 1.0000 = champion**;
> nlms 22.27 / fdnlms 23.77 / cgmdf 23.77 (β=0 wrapper; β=0.3 → 24.47) /
> pfcg 26.11. Criterion cliff solved ([0;e] on OLS; structural on WOLA).
> Engine set reduced to 5 keys — the two failed/redundant engines were
> removed with their measurements archived.

## Completed this cycle (2026-10-06)

- [x] **WOLA filterbank family built** (`conjugate_fb_toeplitz.py`,
      nfft 1024 / hop 256, per-bin scalar FIRs across subband ticks,
      structurally headless — no [0;e], no G projection needed).
      Theory: docs/wola_vs_overlapsave.md (CN/EN).
- [x] **Per-bin RLS solver — new champion.** solver='rls'
      (Sherman-Morrison P = R⁻¹, P(0) = I/δ fading loading, λ = 0.999,
      gate = anti-wind-up): **44.68 dB / corr 1.0000**; +11–14 dB over the
      CG hybrid on every speech pair; d512@64 s 74.6 dB (87.9 ungated).
      Without the gate it wind-ups to 26.5 — the gate is mandatory.
- [x] **Chang & Willson (TSP 2000) trial** — Table I implemented
      equation-by-equation (CG1/CG2, 4 β methods, periodic reset, k_max);
      limits mapped; the paper's correlation direction is the family
      member that fails on streaming speech.
- [x] **Multi-K / paper-reset trial** — k_max > 1 overfits nonstationary
      speech (β/γ/δ all swept); reset flat; internal='true' is 4× faster
      on stationary probes but loses on pairs. k_max = 1 stays.
- [x] **Correlation-mode autopsy** — best possible tuning (β=1, δ=8):
      9.35 dB; damping buys only +0.4 (the target wanders, not the steps);
      speech asymptote 14.5 @ 20 s; stationary on-grid 38.1 @ 64 s
      (beats the CG champion there — the one honest regime).
- [x] **Dissection: the Toeplitz projection is innocent** (|R−T|/|R| =
      0.37%, frozen T-solve 45.4 dB ≈ RLS) — the performance spread is
      TRAJECTORY: constant-ridge exact re-solve explodes in low
      excitation; RLS's fading loading + recursive smoothing survives.
      FINDINGS Addendum 5.
- [x] **Geometry sweep** — hop 128 (8× overlap) = sweet spot: RLS 55.30 /
      CG 36.92 (scratch-verified); hop 64 collapses (lag collinearity);
      latency 56 ms, compute ×2. Not wired into the wrapper yet (item
      below).
- [x] **Engine cleanup** — CONJUGATE_MDF merged into FD_NLMS as the
      `beta` option (bit-exact at β = 0.3 / 0.05 / preemph; class
      deleted; module renamed conjugate_mdf.py → FD_NLMS.py);
      conjugate_toeplitz.py (cliff counterexample, 5.94 dB) and
      conjugate_full.py (WOLA-NLMS, 10.66 dB) removed — measurements
      archived in results/2026-10-06_conjugate-fb-toeplitz/FINDINGS.md
      and results/2026-10-06_conjugate-toeplitz/.
- [x] **Docs overhaul** — docs/OVERVIEW.md (front door: theory, code,
      config, the tangled questions, results);
      wola_vs_overlapsave CN+EN; autocorr_matrix_methods CN+EN;
      fold_and_oversampling_notes CN+EN; DATAFLOW §11 (WOLA family);
      README suite table; oversampling conventions corrected
      (standard ratio = nbin/hop).

## Bottlenecks (evidence-based)

1. **Dense-reverb per-bin convergence starvation** — N_G=64 reaches only
   7.61 dB in 4 s on the conference pair; step dilution (normalizer sums
   all partitions) + tap collinearity (decaying-noise tail) + few
   independent samples. Full-tap time-domain NLMS still wins there
   (16.03 dB). Untested candidate: RLS + n_g=32 (reverb30 scratch:
   RLS 26.31 vs CG 23.85 — coverage first, then the solver).
2. **Colored-speech conditioning** — same engine: 48 dB on flat noise vs
   21–31 dB on speech; voiced seconds reach 47–57 dB, pauses drag the
   average. Partial fix: pre-emphasis (+10.3 dB speechlp).
3. **FD_NLMS anomaly on repeated content** — ERLE *drops* 23.77 → 17.69
   on 20 s tiled canonical while PBFDAF-CG climbs to 27.63. ROOT CAUSED:
   exactly-periodic excitation makes the mu=1 constrained FDAF
   weight-error accumulate coherently; fresh 20 s noise = 51.8 dB. Not a
   bug. Mitigations: mu ≤ 0.5, β > 0, CG/RLS family.

## Quick wins (hours)

- [x] **Pre-emphasis whitening** — DONE: --preemph in run_fdnlms;
      raw-domain (alpha=0.95): speechlp FD_NLMS +10.3 dB (21.7 → 32.0),
      PFCG +2.6; canonical +1.0/+2.3. Whitened-domain metric misleads —
      always de-emphasize for reporting. results/2026-10-06_preemph/.
- [x] **Long reverb soak** — DONE (30 s, seed-123 conference RIR): data
      quantity was the under-weighted variable — N_G=64 goes 7.6 dB (5 s)
      → 18.5–23.3 (30 s noise), parity with full-tap NLMS (23.4). NEW
      PFCG rule: delta 4–8 on colored dense reverb.
      results/2026-10-06_reverb-soak/.
- [x] **Debug the FD_NLMS 20 s degradation** — ROOT CAUSED (tiling
      artifact, see Bottleneck 3). results/2026-10-06_anomaly-20s/.
- [x] **Proportionate step** — DONE: FD_NLMS(rho) + --proportionate.
      Sparse/slack-N_G wins +4.3 dB (28.03, beats uniform PFCG); fully
      used filter loses −3.9 at N_G=8. rho ≤ 0.5.
      results/2026-10-06_proportionate/.

## Medium effort (days)

- [ ] **Re-run the official suite at hop 128** — RLS 55.30 / CG 36.92 at
      1024/128 are scratch-verified; wire the geometry into the FBTOE
      wrapper once the benchmark-freeze policy is decided (latency
      768 → 896 samples = 56 ms; compute ×2).
- [ ] **reverb30 pairs with n_g=32 + RLS** — the only scenario family
      never run through the official harness (800 ms RIR ≫ n_g·hop =
      2048 at n_g=8; coverage is the binding constraint, then the solver).
- [ ] **Port pre-emphasis + proportionate ρ to the WOLA family** — proven
      OS-family wins (speechlp +10.3; sparse +4.3) never reached fbtoe.
- [ ] **Double-talk benchmark + gate ROC** — near-end + echo mixed pairs;
      quantify gate false-open / false-closed rates. If unstable: two-H
      architecture (background filter for control, G.168-style).
- [ ] **Path-change tracking benchmark** — delay/path switch mid-signal,
      measure re-convergence time; VFF-RLS (variable forgetting factor,
      Paleologu/Benesty lineage) is the literature candidate; consider
      dual-rate (fast/slow bank, min-power select).
- [ ] **Constraint cost study** — systematically compare
      constrain='full' vs 'partial' vs 'none' at N_G ∈ {8, 64} on
      canonical + reverb. (Unconstrained already wins +2.4 dB on
      canonical; 'partial' saves N_G−1× projection FFTs.)

## Bigger projects (weeks, paper-grade)

- [ ] **True Table-2 constrained CG** (AES 2006 full machinery:
      truncated-circulant D_T, G-wrapped products, kmax > 1) — the
      principled fix for inner-iteration divergence; the theory-grade
      answer to the ill-conditioned reverb Hessian.
- [ ] **True multichannel (P > 1)** — coupled (P·N_G)×(P·N_G) block Gram
      per bin (the paper's actual use case); current Nrxref keeps
      per-reference independent Grams.
- [ ] **Cross-band weights** — per-bin independence is the WOLA family's
      remaining approximation (neighbor-band leakage absorbed by
      oversampling + PR); a bin×bin weight matrix (multichannel Gram) is
      the rigorous fix if the off-grid floor ever matters.

## Hygiene / follow-ups

- [x] Stale references to the deleted `pfadf_mdf_cg.py` — harness/
      adapters cleaned (dead adapter + factory removed, verified);
      DATAFLOW §4 + PROJECT_ARCHITECTURE rows left as historical record
      (noted in place).
- [x] `conjugate_mdfbackup` stray backup — verified absent from disk.
- [x] Module rename conjugate_mdf.py → FD_NLMS.py — all imports and docs
      re-pointed; scratch archival scripts verified import-clean.
- [ ] `visualize_weight_convergence.py` imports a nonexistent module
      (pre-existing); fix or retire.
- [ ] Commit the working tree — the whole 2026-10-06 cycle (FD_NLMS.py
      rename, fbtoe RLS solver, two engine deletions, CONJUGATE_MDF
      merge, docs + figures) is uncommitted.
- [ ] Re-verify metrics targets after any algorithm change:
      ERLE >= 15 dB, corr >= 0.9, learned path exact on canonical.
- [ ] Optional: EN-labeled variant of docs/osfb_analysis_buffer.png
      (its pixels still carry the old rig filename).

## Out of scope (by project decision)

- Nonlinear echo / NLP / residual suppression (linear AEC only).
- Analysis-filterbank (OSFB) designs — parked 2026-10-05; files kept.

"""
Optimized Toeplitz-Matrix Conjugate Gradient Echo Canceller

Vectorized implementation of the Conjugate Gradient MDF (Multi-Delay Filter) echo canceller.
Each instance handles a single echo canceller configuration.
For multiple configurations, create multiple instances and select minimum power output.

Two criteria (selected by the `hop` constructor parameter):

  hop = M (RECOMMENDED) — the classic [0;e] / canonical MDF criterion:
      INPUT/OUTPUT FRAME GEOMETRY IS IDENTICAL TO FD_NLMS: full L-sample
      frame spectra in (Y, Y_rx), zero-headed a priori error spectrum
      rfft([0_{L-M}; e]) out — the two classes differ ONLY in the weight
      update. The adaptation accumulates the EXACT gradient of the
      valid-region-only objective, stepped with FD-NLMS normalization
      (mu, beta = gradient averaging, 0 = exactly FD_NLMS's
      instantaneous update), the G = [I_M, 0] weight constraint (Lee &
      Chang eq. 10: W = FFT[H; 0...]) is applied automatically, and a
      reference-excitation gate freezes the adaptation when the
      reference is not identifiable (reverb ring-down / digital silence
      / double talk). This matches the published MDF recipe
      equation-for-equation (Lee & Chang, "On the Convergence Properties
      of MDF", eqs. 10/12/14/16-19) and is FREE OF THE CRITERION CLIFF:
      the legacy full-frame criterion saturates at ~10 dB for echo
      delays that are not multiples of the hop (docs/DATAFLOW.md §7);
      the [0;e] criterion does not. Recommended: mu = 1.0, beta in
      [0, 0.5].
      KNOWN LIMIT (measured): per-bin normalized updates converge
      ~n_g-fold slower than a time-domain NLMS on dense reverberant
      speech in short files — for realistic reverb the full-tap engine
      remains the workhorse (results/2026-09-24_noise-oracle-debug/
      FINDINGS.md §7).

  hop = None (legacy) — the original full-frame Toeplitz-CG criterion
      (the original MATLAB reference conjurate_mdf.m lineage): per-bin Rayleigh line-search step on
      toeplitz(autoR) with the positive-curvature guard. Retained
      bit-identical for reproducibility. WARNING: this criterion has the
      delay-mod-hop cliff — do not use it for non-hop-aligned echo paths.

References:
    - "Analysis of Conjugate Gradient Algorithms for Adaptive Filtering"
      (see aec/AnalysisofConjugateGradientAlgorithmsforAdaptiveFiltering.pdf)
    - Lee & Chang, "On the Convergence Properties of Multidelay Frequency
      Domain Adaptive Filter" (aec/ON THE CONVERGENCE PROPERTIES ...pdf)
    - Lalos & Berberidis, "Frequency Domain... Conjugate Gradient"
      (aec/freqgradientconjugate.pdf)
    - Valin, J.-M. (2007). "On Adjusting the Learning Rate in Frequency Domain 
      Echo Cancellation With Double-Talk." ICASSP 2007.
    - Sayed, A.H. (2003). "Fundamentals of Adaptive Filtering." Wiley.

Also provides FD_NLMS: a partitioned-block frequency-domain NLMS with the
classic overlap-save conventions (error on the new block only, [0;e]) — the
baseline the CG-MDF family is measured against (see docs/DATAFLOW.md §7).
"""

import numpy as np
from scipy.linalg import toeplitz


# Recommended parameter values based on literature and practical AEC implementations
CONJUGATE_MDF_PARAMS_CONSERVATIVE = {
    'alpha': 0.02,   # Learning rate - stable, slower convergence
    'beta': 0.99,    # Forgetting factor - high stability
}

CONJUGATE_MDF_PARAMS_BALANCED = {
    'alpha': 0.05,   # Learning rate - recommended starting point
    'beta': 0.97,    # Forgetting factor - balance speed/stability
}

CONJUGATE_MDF_PARAMS_AGGRESSIVE = {
    'alpha': 0.1,    # Learning rate - fast convergence
    'beta': 0.95,    # Forgetting factor - fast tracking
}

# Recommended hop-mode ([0;e], cliff-free) parameters: gradient averaging
# 0.3 trades a little speed for smoothing (beta=0 = exactly FD_NLMS).
CONJUGATE_MDF_PARAMS_HOP = {
    'alpha': 0.03,   # gradient-accumulation scale (cancels; any > 0)
    'beta': 0.3,     # gradient averaging factor (0..0.5; NOT 0.99)
    'mu': 1.0,       # NLMS-normalized step
}


class CONJUGATE_MDF:
    """
    Optimized Conjugate Gradient MDF Echo Canceller for a single configuration.
    
    For multiple filter length configurations, create multiple instances
    and select the output with minimum power.
    """
    
    __slots__ = ['nchan', 'nbin', 'N_G',
                 'alpha', 'beta', 'bin_lim', 'Nrxref', 'bin_skip',
                 'tap_constraint', 'hop', 'mu', 'Pn',
                 'gate_rel', 'gate_hold', 'gate_decay', 'gate_floor', '_Prun', '_hold',
                 'buf_Y_rx', 'buf_Y', 'autoR', 'rcross', 'gacc', 'w', 'w_last',
                 'cntTrig', 'P_X_rx', 'output', 'e', 'max_N_G']

    def __init__(self, NCHAN, NBIN, N_G,
                 alpha, beta, bin_lim, Nrxref=1, bin_skip=0,
                 tap_constraint=None, hop=None, mu=1.0,
                 gate_rel=0.3, gate_hold=2, gate_decay=1.5, gate_floor=1e-6):
        """
        Initialize the Conjugate Gradient MDF echo canceller.

        Parameters
        ----------
        NCHAN : int
            Number of input microphones
        NBIN : int
            Number of frequency bins
        N_G : int
            Filter length for this EC configuration
        alpha : float
            Learning rate parameter
        beta : float
            Forgetting factor
        bin_lim : int
            Bin processing limit
        Nrxref : int, optional
            Number of reference channels (default 1)
        hop : int or None, optional
            Block size M (valid output samples per frame). If set, the
            classic [0;e] convention is used: the adaptation accumulates
            the EXACT gradient of the valid-region-only criterion
            (error taken on the new block, rfft([0_{L-M}; e]) — the same
            criterion as FD_NLMS and the classic FDAF/MDF literature),
            which removes the criterion cliff for non-hop-aligned delays
            (docs/DATAFLOW.md §7). None (default) keeps the legacy
            full-frame criterion bit-identical.
        """
        self.nchan = NCHAN
        self.nbin = NBIN
        self.N_G = int(N_G)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.bin_lim = int(bin_lim)
        self.Nrxref = Nrxref
        # Lowest bin_skip bins are frozen at zero weight (no echo estimate,
        # content passes through): DC / ultra-low bins are normally removed
        # by a downstream high-pass filter, and their STFT-frame correlation
        # matrices are the most ill-conditioned. Default 0 = adapt all bins.
        self.bin_skip = int(bin_skip)
        # Paper Table 2's G = [I_M, 0] constraint (PAES 2006 / Lee & Chang
        # MDF eq. 10): keep only the first tap_constraint time-domain taps
        # of each partition weight and zero the rest (project w after every
        # update). This confines the per-bin weights to the causal
        # partition family so the frequency products cannot fit
        # circular-wrap garbage. In the CLASSIC MDF the constraint is part
        # of the weight DEFINITION (W = FFT[H; 0...]) — therefore in hop
        # ([0;e]) mode it is applied AUTOMATICALLY as tap_constraint=hop
        # unless explicitly overridden (every measurement favoured it:
        # speech pair +2.6 dB at beta=0; neutral-to-better on white noise;
        # the stability-critical config on reverb). None = off in legacy
        # mode (bit-identical historical behaviour); in hop mode None
        # resolves to hop itself.
        self.tap_constraint = None if tap_constraint is None else int(tap_constraint)
        # Classic [0;e] mode (see __init__ docstring): None = legacy.
        self.hop = None if hop is None else int(hop)
        if self.hop is not None and not 0 < self.hop <= 2 * (NBIN - 1) // 2:
            raise ValueError("hop must be in (0, nfft/2]")
        if self.hop is not None and self.tap_constraint is None:
            self.tap_constraint = self.hop        # canonical MDF, eq. 10
        # [0;e] mode: reference-excitation gate (hop mode only; legacy is
        # untouched so it stays bit-identical). Open when the reference is
        # identifiable: P_ref > gate_rel*P_mic AND P_ref > gate_floor*P_run
        # (running peak). gate_hold frames of hysteresis, gate_decay s
        # peak time constant. gate_rel = None disables. See the gate block
        # in apply() for the measured rationale.
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))  # hold>=1: 0 would latch the gate shut
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        # [0;e] mode step size for the normalized averaged-gradient update
        # (0 < mu <= 2; alpha/beta still control gradient/power averaging).
        # NOTE (2026-09-24, reverb conference pair): this per-bin normalized
        # update converges ~n_g-fold slower than time-domain NLMS on dense
        # reverberant speech (step dilution mu/n_g) and can degrade during
        # speech pauses (reverb rings on while per-bin |X| -> 0). It is
        # unbiased (climbs past the legacy plateau given data) but not the
        # workhorse for realistic reverb in short files. Tried and failed
        # here: Tikhonov/gated/smoothed/per-partition-normalized and
        # Rayleigh-on-[0;e]-gradient variants — see
        # results/2026-09-24_noise-oracle-debug/FINDINGS.md.
        self.mu = float(mu)
        self.max_N_G = self.N_G
        
        self.reset()
    
    def reset(self):
        """Reset all internal state buffers."""
        # Buffers: [nbin, N_G, Nrxref/nchan]
        self.buf_Y_rx = np.zeros((self.nbin, self.N_G, self.Nrxref), dtype=complex)
        self.buf_Y = np.zeros((self.nbin, 1, self.nchan), dtype=complex)
        
        # State arrays
        # autoR: [nbin, N_G] per reference channel
        # rcross, w: [nbin, N_G, nchan] per reference channel
        # Zero init for BOTH correlations: they must satisfy the same
        # relation as the data (for mic=ref, rcross == autoR elementwise,
        # so w=e1 solves toeplitz(autoR) w = rcross exactly). The old 1e-4
        # autoR init broke that relation with a phantom bias -1e-4*ones,
        # whose chase explodes when T goes near-singular during speech
        # pauses. With zero init, T -> 0 in silence and the |den| guard
        # self-suspends updates (the MATLAB class needs explicit
        # thr_loud/cnt_loud scheduling for the same effect).
        self.autoR = [np.zeros((self.nbin, self.N_G), dtype=complex)
                     for _ in range(self.Nrxref)]
        self.rcross = [np.zeros((self.nbin, self.N_G, self.nchan), dtype=complex)
                       for _ in range(self.Nrxref)]
        # [0;e] mode: accumulated EXACT gradient of the valid-region-only
        # criterion (zero-headed error). Replaces rcross in the CG step.
        self.gacc = [np.zeros((self.nbin, self.N_G, self.nchan), dtype=complex)
                     for _ in range(self.Nrxref)]
        # [0;e] mode: smoothed full-frame reference power per bin,
        # Pn = beta*Pn + (1-beta)*sum_j sum_iref |X(t-j)|^2 — the FD-NLMS
        # normalizer scale (sum over partitions and ref channels).
        self.Pn = np.zeros(self.nbin)
        # [0;e] mode: reference-excitation gate (user-requested 2026-09-26;
        # the reference MATLAB's thr_loud/cnt_loud scheduling). A running-peak tracker on the
        # total reference frame power opens the adaptation when
        # P > gate_rel * P_run; a hold counter bridges micro-gaps. During
        # closed frames the gradient accumulation AND the weight update
        # are suspended (the pause-kick failure: reverb rings in the mic
        # while the reference is silent — |E|/|X| unbounded).
        self._Prun = 0.0
        self._hold = 0
        # Initialize weights with small random values for numerical stability
        self.w = [(np.random.randn(self.nbin, self.N_G, self.nchan) +
                   1j * np.random.randn(self.nbin, self.N_G, self.nchan)) * 1e-6
                  for _ in range(self.Nrxref)]
        self.w_last = [(np.random.randn(self.nbin, self.N_G, self.nchan) +
                        1j * np.random.randn(self.nbin, self.N_G, self.nchan)) * 1e-6
                       for _ in range(self.Nrxref)]
        for w_ in self.w + self.w_last:
            w_[:self.bin_skip] = 0.0      # frozen bins: no echo estimate
        
        # Outputs: [nbin, nchan]
        self.output = np.zeros((self.nbin, self.nchan), dtype=complex)
        self.e = np.zeros((self.nbin, self.nchan), dtype=complex)
        
        # Debug
        self.cntTrig = np.zeros(self.nbin, dtype=np.int32)
        self.P_X_rx = np.zeros((self.nbin, self.Nrxref))
    
    def apply(self, Y, Y_rx):
        """
        Apply echo cancellation to frequency domain input.
        
        Parameters
        ----------
        Y : ndarray
            Microphone input [NBIN, NCHAN]
        Y_rx : ndarray
            Reference signal [NBIN, Nrxref]
        
        Returns
        -------
        ndarray
            Echo cancelled output [NBIN, NCHAN]
        """
        nbin, nchan = self.nbin, self.nchan
        N_G, alpha, beta = self.N_G, self.alpha, self.beta
        
        # =========================================================================
        # 1. Update buffers (vectorized)
        # =========================================================================
        # Example: if buf_Y_rx is [[a, b, c], [d, e, f]] along axis=1,
        # np.roll(..., -1, axis=1) shifts to [[b, c, a], [e, f, d]]
        # so the oldest frame is removed and space is made for the new one.
        self.buf_Y_rx = np.roll(self.buf_Y_rx, -1, axis=1)
        self.buf_Y_rx[:, -1, :] = Y_rx
        
        self.buf_Y = np.roll(self.buf_Y, -1, axis=1)
        self.buf_Y[:, -1, :] = Y
        
        # =========================================================================
        # 2. Apply filtering with w_last
        # =========================================================================
        rx_flipped = np.flip(self.buf_Y_rx, axis=1)  # [nbin, N_G, Nrxref]
        
        # Echo estimate: sum over reference channels and filter taps
        micest = np.zeros((nbin, nchan), dtype=complex)
        for iref in range(self.Nrxref):
            micest += np.sum(self.w_last[iref] * rx_flipped[:, :, iref:iref+1], axis=1)
        
        # Output and error
        self.output = Y - micest
        self.e = self.output.copy()
        
        # =========================================================================
        # 3. Update accumulated power (for diagnostics)
        # =========================================================================
        # Two criterion modes:
        # - legacy (hop=None): full-frame correlations. autoR/rcross encode
        #   the FULL-frame LS criterion; the CG step solves its normal
        #   equations. This criterion scores the overlap-save head and has
        #   the delay-mod-hop cliff (docs/DATAFLOW.md §7).
        # - classic [0;e] (hop set): autoR stays a full-frame Toeplitz
        #   CURVATURE estimate (step size only), but the adaptation
        #   direction is the accumulated EXACT gradient of the
        #   valid-region-only criterion: E = rfft([0_{L-hop}; e_tail]),
        #   grad_j = conj(X(t-j)) * E. The time-gating operator is linear,
        #   so this gradient is exact (no bin-decoupling approximation);
        #   the fixed point is the [0;e] optimum, which for representable
        #   paths has ZERO residual - no cliff (verified: FD_NLMS reaches
        #   the 49 dB oracle ceiling on the same geometry).
        # =========================================================================
        rx_buf_flip = np.flip(self.buf_Y_rx, axis=1)  # [nbin, N_G, Nrxref]

        # Reference-excitation gate (hop mode): open only when the
        # reference carries identifiable excitation — BOTH
        #   P_ref > gate_rel * P_mic   (reference dominates the mic: the
        #                               echo is identifiable. Closed during
        #                               reverb ring-down (mic rings, ref
        #                               silent) AND during double talk
        #                               (near-end dominates the mic))
        #   P_ref > gate_floor * P_run (not digitally silent; P_run =
        #                               running peak of P_ref, slow decay)
        # plus a hold counter that bridges micro-gaps. This is the reference
        # MATLAB's thr_loud/cnt_loud scheduling made scale-free AND ratio-aware: a pure
        # LEVEL threshold cannot separate quiet speech (adapt!) from a
        # reverberant pause (freeze!) — measured 2026-09-26: level gate at
        # -30 dB gave reverb +4 dB but canonical speech -12 dB.
        gate_open = True
        if self.hop is not None and self.gate_rel is not None:
            P_ref = float(np.sum(np.abs(Y_rx) ** 2))
            P_mic = float(np.sum(np.abs(Y) ** 2))
            decay = np.exp(-self.hop / (16000.0 * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0

        # Zero-headed spectra for [0;e] mode: E_zh = zero-headed error —
        # the EXACT gradient side (the time-gating operator is linear, so
        # accumulating conj(X(t-j))*E_zh gives the exact valid-region
        # gradient; correlation-side decompositions are NOT exact — see
        # the adaptation-step notes). X_zh = zero-headed reference —
        # pairs with the regressor for the autoR curvature diagnostic.
        E_zh = None
        X_zh = None
        if self.hop is not None:
            nfft = 2 * (self.nbin - 1)
            tail_len = self.hop
            e_tail = np.fft.irfft(self.output, n=nfft, axis=0)[nfft - tail_len:]
            x_tail = np.fft.irfft(Y_rx, n=nfft, axis=0)[nfft - tail_len:]
            head = np.zeros((nfft, self.nchan + self.Nrxref))
            head[nfft - tail_len:, :self.nchan] = e_tail
            head[nfft - tail_len:, self.nchan:] = x_tail
            spec = np.fft.rfft(head, axis=0)
            E_zh = spec[:, :self.nchan]                      # [nbin, nchan]
            X_zh = spec[:, self.nchan:]                      # [nbin, Nrxref]

        for iref in range(self.Nrxref):
            # autoR: autocorrelation of reference — conjugate on the regressor:
            # autoR[k] = E[conj(X(t-k)) * X(t)], same side as rcross.
            # The class filters y = sum_j w_j X(t-j) (no conj on w), whose normal
            # equations are toeplitz(autoR) w = rcross with BOTH correlations
            # conjugated on X. Conjugating only one of them mixes the two
            # Hermitian conventions and yields w = T^-1 conj(T) w_true
            # (identity probe 2026-09-23: |w - e1| = 79.5 for the mixed pairing,
            # ~1e-12 when both match; conjurate_mdf.m inherited the same mix).
            # In freq domain: conj(X(t-k)) * X(t)
            # [0;e] mode pairs the regressor with the ZERO-HEADED reference
            # spectrum (curvature of the valid-region criterion).
            # Y_rx shape: [nbin, Nrxref], extract column for this ref
            Y_rx_col = (X_zh if self.hop is not None else Y_rx)[:, iref]

            new_R1 = np.conj(rx_buf_flip[:, :, iref]) * Y_rx_col[:, np.newaxis]  # [nbin, N_G]
            self.autoR[iref] += alpha * new_R1

            # rcross / gacc accumulation per criterion mode
            for i_m in range(nchan):
                Y_col = (E_zh if self.hop is not None else Y)[:, i_m]  # [nbin]
                # conj on the reference (paper Table 2: V^H D; conjurate_mdf.m:210,216)
                # so the accumulation is b_m = sum_t conj(X(t-m)) * target_m(t)
                # ([0;e] mode: target = zero-headed error -> exact gradient)
                new_acc = np.conj(rx_buf_flip[:, :, iref]) * Y_col[:, np.newaxis]  # [nbin, N_G]
                if self.hop is not None:
                    if gate_open:
                        # decay-first so beta=0 => pure instantaneous gradient
                        self.gacc[iref][:, :, i_m] = beta * self.gacc[iref][:, :, i_m] \
                            + alpha * new_acc
                    else:
                        # gate closed (pause): decay only — let the stale
                        # gradient fade instead of accumulating ring-down
                        self.gacc[iref][:, :, i_m] *= beta
                else:
                    self.rcross[iref][:, :, i_m] += alpha * new_acc

            # Apply forgetting factor
            self.autoR[iref] *= beta
            if self.hop is not None:
                self.Pn = beta * self.Pn + (1 - beta) * np.sum(
                    np.abs(rx_buf_flip[:, :, iref]) ** 2, axis=1)
            else:
                self.rcross[iref] *= beta
        
        # =========================================================================
        # 4. Conjugate gradient adaptation (per frequency bin)
        # =========================================================================
        # IMPORTANT: For stability with large N_G, we add regularization and clamping
        # =========================================================================
        for iref in range(self.Nrxref):
            autoR = self.autoR[iref]
            rcross = self.rcross[iref]
            gacc = self.gacc[iref]
            w_last = self.w_last[iref]

            if self.hop is not None and not gate_open:
                # Reference gate closed (pause): freeze the weights this
                # frame — the pause-kick failure mode (reverb rings in the
                # mic while the reference is silent; |E|/|X| unbounded)
                # must not touch the filter.
                self.w[iref] = w_last.copy()
                continue
            if self.hop is not None:
                # ------------------------------------------------------------
                # Classic [0;e] adaptation: averaged exact gradient (PAES
                # eq.-18-style) with FD-NLMS power normalization.
                # The valid-region Hessian is NOT per-bin decoupled NOR
                # Toeplitz in the lag index (the time-domain zero-head
                # gate mixes bins; E[conj(X(t-j)) zhat X(t-k)] is
                # causal-banded: nonzero only for k >= j — which is why
                # paper Table 2 needs its circulant + G-tilde machinery).
                # Every correlation-based second-order step was tried and
                # FAILED on 2026-09-24 (full-frame Toeplitz Rayleigh and
                # zero-headed Toeplitz: divergence; Hermitian-symmetrized
                # banded solve: the mirror system; exact upper-banded
                # solve H w = b AND damped Newton mu*H^-1*g: land on the
                # SAME ~10 dB biased fixed point as the full-frame
                # criterion, because b = E[conj(X) zhat D] itself embeds
                # the per-bin decoupling approximation Z(w*X) ~ w*Z(X).
                # Only the ERROR-based gradient Z(D - sum w X) is exact.
                # Hence: g_avg = gacc*(1-beta)/alpha (the exponential
                # average of the exact [0;e] gradient) stepped with
                # mu/(Pn+eps), where Pn is the smoothed
                # sum-over-partitions full-frame power — the SAME
                # per-frame update scale as FD_NLMS (verified stable and,
                # at beta=0, weight-trajectory-identical to FD_NLMS), but
                # with the class's beta-averaged gradient.
                # ------------------------------------------------------------
                g_scale = (1.0 - beta) / alpha if alpha > 0 else 1.0
                for ibin in range(self.bin_skip, nbin):
                    den = self.Pn[ibin] + 1e-10
                    if self.Pn[ibin] <= 1e-12:     # silence: suspend
                        self.w[iref][ibin] = w_last[ibin]
                        continue
                    self.w[iref][ibin] = w_last[ibin] + \
                        (self.mu * g_scale / den) * gacc[ibin]
            else:
                for ibin in range(self.bin_skip, nbin):
                    # Build Toeplitz matrix from autoR
                    # NOTE: the 1e-30 term below is nominal only (diagonal values
                    # are orders of magnitude larger) — it does NOT regularize.
                    # Measured (2026-09-23): meaningful relative loading trades
                    # identity-echo ERLE for slightly less negative delayed-echo
                    # ERLE; neither setting reaches useful delayed cancellation.
                    r_vec = autoR[ibin, :]

                    r_vec_reg = r_vec.copy()
                    r_vec_reg += 1e-30  # nominal diagonal term

                    T = toeplitz(r_vec_reg)

                    for i_m in range(nchan):
                        # Gradient of the full-frame criterion at w_last:
                        # normal-equations residual (exact for the
                        # accumulated full-frame LS)
                        g0 = rcross[ibin, :, i_m] - T @ w_last[ibin, :, i_m]

                        # Search direction
                        p = g0.copy()

                        # T @ p
                        rp = T @ p

                        # Step size — conjurate_mdf.m line 235:
                        #   alf = 0.999*(conj(p)*(gd_last.')) / ((conj(p)*(rp.')) + 1.0000e-30)
                        # MATLAB conj(p_row)*(x_row).' == np.vdot(p, x): scalar complex inner
                        # product. gd_last == g0 here (single step); alf stays complex — no
                        # real(), no clipping, exactly as in the MATLAB reference.
                        #alf = 0.999 * np.vdot(p, g0) / (np.vdot(p, rp))
                        num = 0.999 * np.vdot(p, g0)
                        den = np.vdot(p, rp)

                        # Positive-curvature + nonzero-denominator guard:
                        # - den = p^H T p is real for Hermitian T; a den <= 0
                        #   (T is a nonstationary windowed estimate and can be
                        #   slightly indefinite) would step uphill and explode.
                        # - den ~ 0 happens in silent/near-silent bins where
                        #   T -> 0 (zero init); skipping there self-suspends the
                        #   update — the equivalent of conjurate_mdf.m's
                        #   thr_loud/cnt_loud scheduling, which this port does
                        #   not carry. The threshold must stay TINY (1e-30, the
                        #   MATLAB constant): a larger absolute threshold (e.g.
                        #   1e-10) also skips legitimate low-excitation updates
                        #   and leaves those bins unconverged (identity probe
                        #   2026-09-23: den straddles 1e-12..1e-8 in quiet bins;
                        #   1e-10 guard -> ERLE -14 dB, 1e-30 guard -> 87 dB).
                        if den.real > 0 and abs(den) > 1e-30:
                            alf = num / den
                        else:
                            alf = 0.0

                        # Update weight
                        self.w[iref][ibin, :, i_m] = w_last[ibin, :, i_m] + alf * p

                        # WEIGHT CLAMPING: prevent individual weights from
                        #@ai_instruction we dont need to do weight clamping.
                        # The only way the weights can explode is cause error explode.
                        # So we don't do to do the weight clamping here.

            # G = [I_M, 0] projection (Table 2 / PAES): after the per-bin CG
            # sweep, confine every partition weight to the first
            # tap_constraint time taps. Without it the unconstrained per-bin
            # weights acquire circular-wrap components that fit frequency
            # regions overlap-save discards (noise-pair oracle 2026-09-24:
            # identical data, tap-constrained partition geometry reaches the
            # time-domain Wiener ceiling while the unconstrained fit stalls
            # under 8 dB). Projection is self-adjoint; the next frame's
            # gradient is taken at the projected point (projected steepest
            # descent -> constrained optimum for psd T).
            if self.tap_constraint is not None:
                nfft = 2 * (self.nbin - 1)
                Wt = np.fft.irfft(self.w[iref], n=nfft, axis=0)
                Wt[self.tap_constraint:, :, :] = 0.0
                self.w[iref] = np.fft.rfft(Wt, n=nfft, axis=0)
        # =========================================================================
        # 5. Apply filtering with updated weights
        # =========================================================================
        # Legacy mode re-filters (Rayleigh steps are small, a priori ~ a
        # posteriori). Hop mode keeps the A PRIORI output from step 2 — the
        # standard convention: at mu ~ 1 the a posteriori residual is
        # degenerate (the update projects the current frame's error — noise
        # included — onto the weights, driving the reported residual to ~0
        # while the filter chases noise; measured 2026-09-24: bogus 245 dB
        # 'ERLE' on a 20 dB-SNR mic).
        if self.hop is None:
            rx_flipped = np.flip(self.buf_Y_rx, axis=1)

            micest = np.zeros((nbin, nchan), dtype=complex)
            for iref in range(self.Nrxref):
                micest += np.sum(self.w[iref] * rx_flipped[:, :, iref:iref+1], axis=1)

            self.output = Y - micest
            self.e = self.output.copy()

        # =========================================================================
        # 5b. Update w_last for next frame
        # =========================================================================
        for iref in range(self.Nrxref):
            self.w_last[iref] = self.w[iref].copy()

        # =========================================================================
        # 6. Scheduling (countdown control)
        # =========================================================================
        # rx power per bin
        P_X_rx = np.sum(np.abs(Y_rx) ** 2, axis=1, keepdims=True)
        self.P_X_rx = P_X_rx

        # =========================================================================
        # 7. Output frame geometry (hop mode): match FD_NLMS exactly —
        #    return the ZERO-HEADED a priori error spectrum rfft([0; e]),
        #    not the full-frame residual. The tail samples are the same
        #    either way (the wrapper keeps irfft(output)[-hop:]), but the
        #    spectrum object is now interchangeable with FD_NLMS's: a
        #    downstream consumer (chain EC / synthesis filterbank) sees
        #    identical head content from both classes. E_zh was computed
        #    from this frame's a priori residual in section 3. Legacy mode
        #    keeps returning the full-frame spectrum (bit-identical).
        # =========================================================================
        if self.hop is not None:
            self.output = E_zh.copy()
            self.e = np.fft.irfft(E_zh, n=2 * (self.nbin - 1),
                                  axis=0)[2 * (self.nbin - 1) - self.hop:]

        return self.output


class FD_NLMS:
    """
    Partitioned-block frequency-domain NLMS (classic overlap-save FDAF).

    Same frequency-frame interface as CONJUGATE_MDF — apply(Y, Y_rx) with
    FULL L-sample frame spectra and N_G lagged partitions — but the classic
    FDAF weight update (Ferrara 1980; Soo & Pang 1990 MDF; same update rule
    as the project's pfadf_mdf_cg.py / DATAFLOW.md §4):

        e   = last `hop` samples of irfft(Y - est)          (new block only)
        E   = rfft([0_{L-hop}, e])                          ([0; e] convention)
        X2  = sum_j |X(t-j)|^2                              (per bin, all partitions)
        w_j += mu * conj(X(t-j)) * E / (X2 + eps)

    The [0; e] convention is the load-bearing difference vs CONJUGATE_MDF's
    full-frame correlations: the adaptation direction is the gradient of the
    VALID-REGION-ONLY criterion, so the discarded head never contaminates
    the update (docs/DATAFLOW.md §7 — the criterion cliff). Setting
    full_frame_error=True reproduces CONJUGATE_MDF's convention inside the
    same NLMS machinery (controlled ablation of the cliff).

    Parameters
    ----------
    NCHAN, NBIN, N_G : as in CONJUGATE_MDF (NBIN = L/2 + 1 bins)
    mu : NLMS step size (0 < mu <= 2, typical 0.5-1.0)
    hop : block size M — valid output samples per frame (lag spacing).
        Must divide the frame: the wrapper advances by `hop` and keeps the
        LAST hop samples of irfft(output).
    constraint : apply the classic G = [I_hop, 0] weight projection after
        every update (each partition confined to its first hop time taps —
        exact linear-convolution partitions; default True).
    full_frame_error : ablation — update from the full-frame residual
        spectrum (head included, CONJUGATE_MDF's convention) instead of the
        zero-headed [0; e] error. Default False (classic).
    gate_rel / gate_hold / gate_decay / gate_floor : reference-excitation
        gate, the same one CONJUGATE_MDF uses in hop mode: the weight
        update is suspended when the reference is not identifiable —
        P_ref <= gate_rel*P_mic (mic dominated by reverb ring-down or
        near-end/double talk) or P_ref <= gate_floor*P_run (digital
        silence). Defaults 0.3 / 2 / 1.5 s / 1e-6; gate_rel = None
        disables. Kept identical to CONJUGATE_MDF so the beta=0
        structural equivalence between the two classes is preserved.
    """

    __slots__ = ['nchan', 'nbin', 'N_G', 'Nrxref', 'mu', 'hop', 'nfft',
                 'constraint', 'full_frame_error', 'eps',
                 'gate_rel', 'gate_hold', 'gate_decay', 'gate_floor', '_Prun', '_hold',
                 'buf_Y_rx', 'w', 'P_X', 'output', 'e', 'X2']

    def __init__(self, NCHAN, NBIN, N_G, mu, hop,
                 Nrxref=1, constraint=True, full_frame_error=False,
                 eps=1e-10, gate_rel=0.3, gate_hold=2, gate_decay=1.5, gate_floor=1e-6):
        self.nchan = NCHAN
        self.nbin = NBIN
        self.N_G = int(N_G)
        self.Nrxref = Nrxref
        self.mu = float(mu)
        self.nfft = 2 * (NBIN - 1)
        self.hop = int(hop)
        if not 0 < self.hop <= self.nfft // 2:
            raise ValueError("hop must be in (0, nfft/2] "
                             f"(got hop={hop}, nfft={self.nfft})")
        self.constraint = bool(constraint)
        self.full_frame_error = bool(full_frame_error)
        self.eps = float(eps)
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))  # hold>=1: 0 would latch the gate shut
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        self.reset()

    def reset(self):
        self.buf_Y_rx = np.zeros((self.nbin, self.N_G, self.Nrxref),
                                 dtype=complex)
        # NLMS starts from zero weights (no correlation-consistency
        # requirement like the Toeplitz CG class — the update is local).
        self.w = [np.zeros((self.nbin, self.N_G, self.nchan), dtype=complex)
                  for _ in range(self.Nrxref)]
        self.output = np.zeros((self.nbin, self.nchan), dtype=complex)
        self.e = np.zeros(self.hop, dtype=float)
        self.X2 = np.zeros(self.nbin)
        self._Prun = 0.0
        self._hold = 0

    def apply(self, Y, Y_rx):
        """
        Process one frame. Y [NBIN, NCHAN] mic spectrum, Y_rx [NBIN, Nrxref]
        reference spectrum (both FULL L-sample frame FFTs, L = 2*(NBIN-1)).
        Returns the output spectrum E = rfft([0; e]) — the wrapper keeps the
        last `hop` samples of irfft(E).
        """
        nbin, N_G = self.nbin, self.N_G

        # 1. buffer update (same layout as CONJUGATE_MDF)
        self.buf_Y_rx = np.roll(self.buf_Y_rx, -1, axis=1)
        self.buf_Y_rx[:, -1, :] = Y_rx
        rx_flipped = np.flip(self.buf_Y_rx, axis=1)   # rx_flipped[:, j] = X(t-j)

        # 2. a priori estimate and residual
        est = np.zeros((nbin, self.nchan), dtype=complex)
        for iref in range(self.Nrxref):
            est += np.sum(self.w[iref] * rx_flipped[:, :, iref:iref+1], axis=1)
        R = Y - est

        # 3. error for the update: classic [0; e] or full-frame ablation
        e_time = np.fft.irfft(R, n=self.nfft, axis=0)[self.nfft - self.hop:]
        if self.full_frame_error:
            E_upd = R
        else:
            E_head = np.zeros((self.nfft, self.nchan))
            E_head[self.nfft - self.hop:] = e_time
            E_upd = np.fft.rfft(E_head, axis=0)

        # 4. per-bin power across partitions, normalized update —
        #    suspended when the reference-excitation gate is closed (pause)
        X2 = np.sum(np.abs(rx_flipped) ** 2, axis=(1, 2))       # [nbin]
        self.X2 = X2
        gate_open = True
        if self.gate_rel is not None:
            P_ref = float(np.sum(np.abs(Y_rx) ** 2))
            P_mic = float(np.sum(np.abs(Y) ** 2))
            decay = np.exp(-self.hop / (16000.0 * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0
        if gate_open:
            gain = self.mu * E_upd / (X2[:, np.newaxis] + self.eps)  # [nbin, nchan]
            for iref in range(self.Nrxref):
                self.w[iref] += np.conj(rx_flipped[:, :, iref:iref+1]) * gain[:, np.newaxis, :]

            # 5. classic G = [I_hop, 0] weight constraint (per partition)
            if self.constraint:
                for iref in range(self.Nrxref):
                    Wt = np.fft.irfft(self.w[iref], n=self.nfft, axis=0)
                    Wt[self.hop:] = 0.0
                    self.w[iref] = np.fft.rfft(Wt, n=self.nfft, axis=0)

        # 6. output: zero-headed error spectrum (wrapper takes last hop)
        self.e = e_time
        E_out = np.zeros((self.nfft, self.nchan))
        E_out[self.nfft - self.hop:] = e_time
        self.output = np.fft.rfft(E_out, axis=0)
        return self.output


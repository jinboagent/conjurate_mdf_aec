"""
Optimized Toeplitz-Matrix Conjugate Gradient Echo Canceller

Vectorized implementation of the Conjugate Gradient MDF (Multi-Delay Filter) echo canceller.
Each instance handles a single echo canceller configuration.
For multiple configurations, create multiple instances and select minimum power output.

References:
    - "Analysis of Conjugate Gradient Algorithms for Adaptive Filtering"
      (see aec/AnalysisofConjugateGradientAlgorithmsforAdaptiveFiltering.pdf)
    - Valin, J.-M. (2007). "On Adjusting the Learning Rate in Frequency Domain 
      Echo Cancellation With Double-Talk." ICASSP 2007.
    - Sayed, A.H. (2003). "Fundamentals of Adaptive Filtering." Wiley.
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


class CONJUGATE_MDF:
    """
    Optimized Conjugate Gradient MDF Echo Canceller for a single configuration.
    
    For multiple filter length configurations, create multiple instances
    and select the output with minimum power.
    """
    
    __slots__ = ['nchan', 'nbin', 'N_G',
                 'alpha', 'beta', 'bin_lim', 'Nrxref', 'bin_skip',
                 'buf_Y_rx', 'buf_Y', 'autoR', 'rcross', 'w', 'w_last',
                 'cntTrig', 'P_X_rx', 'output', 'e', 'max_N_G']

    def __init__(self, NCHAN, NBIN, N_G,
                 alpha, beta, bin_lim, Nrxref=1, bin_skip=0):
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
            Number of reference channels (default: 1)
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
        # Note: For proper complex correlation:
        # - autoR (autocorrelation): E[X * conj(X)] - should be real for same signal
        # - rcross (cross-correlation): E[X * conj(Y)] where Y is microphone
        # 
        # The flip accounts for time-reversal in correlation computation
        # =========================================================================
        rx_buf_flip = np.flip(self.buf_Y_rx, axis=1)  # [nbin, N_G, Nrxref]

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
            # Y_rx shape: [nbin, Nrxref], extract column for this ref
            Y_rx_col = Y_rx[:, iref]  # [nbin]

            #@ai-instruction  instru there should be a complex value mulitplication

            new_R1 = np.conj(rx_buf_flip[:, :, iref]) * Y_rx_col[:, np.newaxis]  # [nbin, N_G]
            self.autoR[iref] += alpha * new_R1

            # rcross: cross-correlation between reference and microphone
            # rcross[k] = sum_n(rx[n] * conj(Y[n-k]))
            # In freq domain: X * conj(Y)
            # Y shape: [nbin, nchan], result should be [nbin, N_G, nchan]
            for i_m in range(nchan):
                Y_col = Y[:, i_m]  # [nbin]
                # conj on the reference (paper Table 2: V^H D; conjurate_mdf.m:210,216)
                # so rcross accumulates b_m = sum_t conj(X(t-m)) * Y(t)
                new_rcross = np.conj(rx_buf_flip[:, :, iref]) * Y_col[:, np.newaxis]  # [nbin, N_G]
                self.rcross[iref][:, :, i_m] += alpha * new_rcross

            # Apply forgetting factor
            self.autoR[iref] *= beta
            self.rcross[iref] *= beta
        
        # =========================================================================
        # 4. Conjugate gradient adaptation (per frequency bin)
        # =========================================================================
        # IMPORTANT: For stability with large N_G, we add regularization and clamping
        # =========================================================================
        for iref in range(self.Nrxref):
            autoR = self.autoR[iref]
            rcross = self.rcross[iref]
            w_last = self.w_last[iref]

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
                    # Gradient: g0 = rcross - T @ w_last
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
        # =========================================================================
        # 5. Apply filtering with updated weights
        # =========================================================================
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
        

        
        return self.output


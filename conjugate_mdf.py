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
                 'alpha', 'beta', 'bin_lim', 'Nrxref',
                 'buf_Y_rx', 'buf_Y', 'autoR', 'rcross', 'w', 'w_last',
                 'cntTrig', 'P_X_rx', 'output', 'e', 'max_N_G']
    
    def __init__(self, NCHAN, NBIN, N_G,
                 alpha, beta, bin_lim, Nrxref=1):
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
        self.autoR = [np.full((self.nbin, self.N_G), 1e-4, dtype=complex)
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
            # autoR: autocorrelation of reference
            # autoR[k] = sum_n(rx[n] * conj(rx[n-k]))
            # In freq domain: X * conj(X) = |X|^2 (real)
            # Y_rx shape: [nbin, Nrxref], extract column for this ref
            Y_rx_col = Y_rx[:, iref]  # [nbin]
            new_R1 = rx_buf_flip[:, :, iref] * np.conj(Y_rx_col[:, np.newaxis])  # [nbin, N_G]
            self.autoR[iref] += alpha * new_R1

            # rcross: cross-correlation between reference and microphone
            # rcross[k] = sum_n(rx[n] * conj(Y[n-k]))
            # In freq domain: X * conj(Y)
            # Y shape: [nbin, nchan], result should be [nbin, N_G, nchan]
            for i_m in range(nchan):
                Y_col = Y[:, i_m]  # [nbin]
                new_rcross = rx_buf_flip[:, :, iref] * np.conj(Y_col[:, np.newaxis])  # [nbin, N_G]
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

            for ibin in range(nbin):
                # Build Toeplitz matrix from autoR with regularization
                r_vec = autoR[ibin, :]
                
                # Add diagonal loading for numerical stability
                # This prevents ill-conditioning when autoR values are small
                r_vec_reg = r_vec.copy()
                r_vec_reg[0] += 1e-30  # Diagonal loading
                
                T = toeplitz(r_vec_reg)

                for i_m in range(nchan):
                    # Gradient: g0 = rcross - T @ w_last
                    g0 = rcross[ibin, :, i_m] - T @ w_last[ibin, :, i_m]

                    # Search direction
                    p = g0.copy()

                    # T @ p
                    rp = T @ p

                    # Step size: alf = 0.999 * (p^H @ g0) / (p^H @ rp + eps)
                    # IMPORTANT: Take REAL part - step size must be real for stability
                    # p^H @ g0 and p^H @ rp should be real for Hermitian T
                    num = 0.999 * np.real(np.vdot(p, g0))
                    den = np.real(np.vdot(p, rp)) + 1e-30  # Increased regularization for large N_G

                    if np.abs(den) > 1e-30:
                        alf = num / den
                    else:
                        alf = 0.0
                    
                    # CLAMP step size for stability with large N_G
                    # Prevents weight explosion when denominator is small
                    alf = np.clip(alf, -1.0, 1.0)

                    # Update weight
                    self.w[iref][ibin, :, i_m] = w_last[ibin, :, i_m] + alf * p
                    
                    # WEIGHT CLAMPING: Prevent individual weights from exploding
                    # This is a safety mechanism for numerical stability
                    w_current = self.w[iref][ibin, :, i_m]
                    max_weight = 10.0  # Reasonable bound for frequency domain weights
                    if np.any(np.abs(w_current) > max_weight):
                        w_current = np.clip(w_current, -max_weight, max_weight)
                        self.w[iref][ibin, :, i_m] = w_current

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



"""
Partitioned Block Frequency-Domain Adaptive Filter with Conjugate Gradient (PBFDAF-CG)

This module implements the PBFDAF-CG algorithm as described in:
"The Conjugate Gradient Partitioned Block Frequency-Domain Adaptive Filter
for Multichannel Acoustic Echo Cancellation"
Lino García et al., EUSIPCO 2006

The algorithm combines frequency-domain filtering with conjugate gradient
optimization to achieve faster convergence than standard PBFDAF (LMS-based).
"""

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


class PFDAFCG:
    """
    Partitioned Block Frequency-Domain Adaptive Filter with Conjugate Gradient
    
    Parameters
    ----------
    N : int
        Number of partitions/blocks
    winlen : int
        Window length (block size K)
    mu : float
        Base step size
    nchan : int
        Number of channels (default: 1)
    k_max : int
        Maximum number of CG iterations per block (default: 1)
    beta_method : str
        Beta calculation method: 'hestenes-stiefel', 'fletcher-reeves', 
        'polak-ribiere', or 'dai-yuan' (default: 'hestenes-stiefel')
    constrain : str
        Constraint type: 'full', 'partial', or 'none' (default: 'partial')
    """
    
    def __init__(self, N, winlen, mu, nchan=1, k_max=1, 
                 beta_method='hestenes-stiefel', constrain='partial'):
        self.N = N                    # Number of partitions
        self.M = winlen               # Block size K
        self.N_freq = 1 + winlen      # Number of frequency bins
        self.N_fft = 2 * winlen       # FFT size
        self.mu = mu                  # Step size
        self.nchan = nchan            # Number of channels
        self.k_max = k_max            # Max CG iterations
        self.beta_method = beta_method
        self.constrain = constrain
        self.partition_index = 0      # For partial constrain
        
        # Input buffer
        self.x_old = np.zeros(self.M, dtype=np.float64)
        
        # Frequency domain buffers
        self.X = np.zeros((self.N, self.N_freq), dtype=np.complex128)
        self.H = np.zeros((self.N, self.N_freq), dtype=np.complex128)
        
        # CG state variables
        self.g = np.zeros((self.N, self.N_freq), dtype=np.complex128)  # Gradient
        self.v = np.zeros((self.N, self.N_freq), dtype=np.complex128)  # Conjugate direction
        self.g_prev = np.zeros((self.N, self.N_freq), dtype=np.complex128)  # Previous gradient
        
        # Correlation estimates (for gradient computation)
        self.R = np.zeros((self.N, self.N_freq), dtype=np.float64)  # Autocorrelation
        self.C = np.zeros((self.N, self.N_freq), dtype=np.complex128)  # Cross-correlation
        
        # Averaging factors (from paper eq. 22-23)
        self.U = np.ones((self.N, self.N_freq), dtype=np.float64)  # Power normalization
        self.beta_avg = 0.25  # Averaging factor
        self.delta = 0.5      # Stability constant
        
        # Window
        self.window = np.hanning(self.M)
        
        # Statistics
        self.iteration_count = 0
        self.cg_iterations = []
    
    def filt(self, x, d):
        """
        Filter input block and compute error
        
        Parameters
        ----------
        x : ndarray
            Input signal block (length M)
        d : ndarray
            Desired signal block (length M)
            
        Returns
        -------
        e : ndarray
            Error signal (length M)
        """
        assert len(x) == self.M
        
        # Concatenate with old buffer and compute FFT
        x_now = np.concatenate((self.x_old, x))
        X = fft(x_now)
        
        # Update frequency domain buffer (shift and insert)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        
        # Update time domain buffer
        self.x_old = x.copy()
        
        # Compute filter output in frequency domain
        Y = np.sum(self.H * self.X, axis=0)
        
        # Convert to time domain and extract relevant part
        y_time = ifft(Y)[self.M:]
        
        # Compute error
        e = d - y_time
        
        return e
    
    def _compute_gradient(self):
        """
        Compute gradient using averaged correlation estimates
        
        Returns
        -------
        G : ndarray
            Gradient matrix (N x N_freq)
        """
        # Gradient: G = C - R * H (frequency domain)
        G = self.C - self.R * self.H
        return G
    
    def _compute_beta(self, g_current, g_prev, v_prev):
        """
        Compute beta parameter for conjugate gradient
        
        Parameters
        ----------
        g_current : ndarray
            Current gradient
        g_prev : ndarray
            Previous gradient
        v_prev : ndarray
            Previous conjugate direction
            
        Returns
        -------
        beta : ndarray
            Beta parameter
        """
        if self.beta_method == 'fletcher-reeves':
            # Eq. (19): β_FR = (g[k+1]^H * g[k+1]) / (g[k]^H * g[k])
            num = np.sum(np.conj(g_current) * g_current, axis=0, keepdims=True)
            den = np.sum(np.conj(g_prev) * g_prev, axis=0, keepdims=True) + 1e-10
            beta = num / den
            
        elif self.beta_method == 'polak-ribiere':
            # Eq. (20): β_PR = ((g[k+1] - g[k])^H * g[k+1]) / (g[k]^H * g[k])
            g_diff = g_current - g_prev
            num = np.sum(np.conj(g_diff) * g_current, axis=0, keepdims=True)
            den = np.sum(np.conj(g_prev) * g_prev, axis=0, keepdims=True) + 1e-10
            beta = num / den
            
        elif self.beta_method == 'dai-yuan':
            # Eq. (21): β_DY = (g[k+1]^H * g[k+1]) / (v[k]^H * (g[k+1] - g[k]))
            g_diff = g_current - g_prev
            num = np.sum(np.conj(g_current) * g_current, axis=0, keepdims=True)
            den = np.sum(np.conj(v_prev) * g_diff, axis=0, keepdims=True) + 1e-10
            beta = num / den
            
        else:  # hestenes-stiefel (default)
            # Eq. (17): β_HS = ((g[k+1] - g[k])^H * g[k+1]) / (v[k]^H * (g[k+1] - g[k]))
            g_diff = g_current - g_prev
            num = np.sum(np.conj(g_diff) * g_current, axis=0, keepdims=True)
            den = np.sum(np.conj(v_prev) * g_diff, axis=0, keepdims=True) + 1e-10
            beta = num / den
        
        # Stability: reset beta if > 1 (from paper)
        beta = np.where(np.abs(beta) > 1, 1.0, beta)
        
        return beta
    
    def conjugate_gradient_update(self, e):
        """cc
        Perform conjugate gradient weight update
        
        Parameters
        ----------
        e : ndarray
            Error signal (length M)
        """
        # Transform error to frequency domain with windowing
        e_fft = np.zeros(self.N_fft, dtype=np.float64)
        e_fft[self.M:] = e * self.window
        E = fft(e_fft)
        
        # Store previous gradient
        self.g_prev = self.g.copy()
        
        # Compute current gradient
        self.g = self._compute_gradient()
        
        # Initialize conjugate direction on first iteration
        if self.iteration_count == 0:
            self.v = -self.g
        else:
            # Compute beta and update conjugate direction
            beta = self._compute_beta(self.g, self.g_prev, self.v)
            self.v = -self.g + beta * self.v
        
        # Compute step size alpha
        # α = -(g^H * v) / (v^H * R * v)
        num = -np.sum(np.conj(self.g) * self.v, axis=0, keepdims=True)
        den = np.sum(np.conj(self.v) * self.R * self.v, axis=0, keepdims=True) + 1e-10
        alpha = self.mu * num / den
        
        # Update weights: H = H + alpha * v
        self.H = self.H + alpha * self.v
        
        # Update correlation estimates (exponential averaging)
        # Power normalization from paper eq. (22-23)
        self.U = (1 - self.beta_avg) * self.U + self.beta_avg * np.abs(self.X)**2
        X2 = self.U + self.delta
        
        # Gradient-based update for correlation
        G_norm = self.mu * E / X2
        self.H = self.H + np.conj(self.X) * G_norm
        
        self.iteration_count += 1
        self.cg_iterations.append(self.k_max)
    
    def update_correlation(self):
        """
        Update autocorrelation and cross-correlation estimates
        """
        # Update autocorrelation: R = (R * (N-1) + X^H * X) / N
        R_new = np.sum(np.abs(self.X[1:])**2, axis=0)
        self.R = (self.R * (self.N - 1) + R_new) / self.N
        
        # Update cross-correlation: C = (C * (N-1) + X^H * D) / N
        # Note: D would come from desired signal in adaptive mode
        C_new = np.sum(np.conj(self.X[1:]) * self.D[1:], axis=0)
        self.C = (self.C * (self.N - 1) + C_new) / self.N
    
    def apply_constraint(self):
        """
        Apply time-domain constraint to filter coefficients
        """
        if self.constrain == 'none':
            return
        elif self.constrain == 'full':
            # Constrain all partitions
            for p in range(self.N):
                h = ifft(self.H[p])
                h[self.M:] = 0
                self.H[p] = fft(h)
        elif self.constrain == 'partial':
            # Constrain only current partition (cycling)
            h = ifft(self.H[self.partition_index])
            h[self.M:] = 0
            self.H[self.partition_index] = fft(h)
            self.partition_index = (self.partition_index + 1) % self.N
    
    def update(self, e, d=None):
        """
        Update filter using conjugate gradient
        
        Parameters
        ----------
        e : ndarray
            Error signal (length M)
        d : ndarray, optional
            Desired signal (for correlation update)
        """
        if d is not None:
            # Update desired signal buffer
            d_now = np.concatenate((self.d_old, d))
            D = fft(d_now)
            self.D[1:] = self.D[:-1]
            self.D[0] = D
            self.d_old = d.copy()
            
            # Update correlation estimates
            self.update_correlation()
        
        # Perform CG iterations
        for k in range(self.k_max):
            self.conjugate_gradient_update(e)
        
        # Apply constraint
        self.apply_constraint()


def pfdaf_cg(x, d, N=4, M=64, mu=0.2, nchan=1, k_max=1, 
             beta_method='hestenes-stiefel', constrain='partial'):
    """
    Partitioned Block Frequency-Domain Adaptive Filter with Conjugate Gradient
    
    Parameters
    ----------
    x : ndarray
        Input/reference signal (far-end)
    d : ndarray
        Desired signal (microphone with echo)
    N : int
        Number of partitions (default: 4)
    M : int
        Block size (default: 64)
    mu : float
        Step size (default: 0.2)
    nchan : int
        Number of channels (default: 1)
    k_max : int
        Maximum CG iterations per block (default: 1)
    beta_method : str
        Beta calculation method (default: 'hestenes-stiefel')
    constrain : str
        Constraint type: 'full', 'partial', or 'none' (default: 'partial')
        
    Returns
    -------
    e : ndarray
        Echo-cancelled output signal
    """
    # Initialize filter
    ft = PFDAFCG(N, M, mu, nchan, k_max, beta_method, constrain)
    
    # Calculate number of blocks
    num_block = min(len(x), len(d)) // M
    
    # Initialize output
    e = np.zeros(num_block * M)
    
    # Process each block
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        
        # Filter and get error
        e_n = ft.filt(x_n, d_n)
        
        # Update filter
        ft.update(e_n, d_n)
        
        # Store output
        e[n*M:(n+1)*M] = e_n
    
    return e


def pfdaf_cg_simple(x, d, N=4, M=64, mu=0.05, k_max=1, constrain=True):
    """
    Simplified PBFDAF-CG for single-channel AEC
    
    This implementation uses a hybrid approach:
    - Frequency domain filtering for efficiency
    - Time-domain CG-inspired weight update for stability
    
    The key insight from the paper is using conjugate directions
    instead of pure steepest descent. Here we approximate this
    by maintaining momentum in the weight updates.
    
    Parameters
    ----------
    x : ndarray
        Input/reference signal (far-end)
    d : ndarray
        Desired signal (microphone with echo)
    N : int
        Number of partitions (default: 4)
    M : int
        Block size (default: 64)
    mu : float
        Step size (default: 0.05)
    k_max : int
        Not used in this simplified version
    constrain : bool
        Apply time-domain constraint (default: True)
        
    Returns
    -------
    e : ndarray
        Echo-cancelled output signal
    """
    N_freq = M + 1
    N_fft = 2 * M
    
    # Initialize frequency domain filter
    H = np.zeros((N, N_freq), dtype=np.complex128)
    X = np.zeros((N, N_freq), dtype=np.complex128)
    
    # Momentum term (CG-inspired)
    momentum = np.zeros((N, N_freq), dtype=np.complex128)
    
    # Power normalization (exponential smoothing)
    U = np.ones((N, N_freq), dtype=np.float64)
    beta_avg = 0.1
    
    # Momentum decay factor (like beta in CG)
    momentum_decay = 0.3
    
    # Buffers
    x_old = np.zeros(M, dtype=np.float64)
    window = np.hanning(M)
    
    # Output
    num_block = min(len(x), len(d)) // M
    e = np.zeros(num_block * M)
    
    for n in range(num_block):
        x_n = x[n*M:(n+1)*M]
        d_n = d[n*M:(n+1)*M]
        
        # Input FFT
        x_now = np.concatenate((x_old, x_n))
        X_n = fft(x_now)
        
        # Update buffer
        X[1:] = X[:-1]
        X[0] = X_n
        x_old = x_n.copy()
        
        # Filter output
        Y = np.sum(H * X, axis=0)
        y_n = ifft(Y)[M:]
        
        # Error
        e_n = d_n - y_n
        e[n*M:(n+1)*M] = e_n
        
        # Error FFT (with windowing)
        e_fft = np.zeros(N_fft, dtype=np.float64)
        e_fft[M:] = e_n * window
        E = fft(e_fft)
        
        # Update power normalization
        X2 = np.sum(np.abs(X)**2, axis=0)
        U = (1 - beta_avg) * U + beta_avg * X2
        
        # Compute gradient direction (steepest descent)
        # G = X^H * E / (|X|^2 + eps)
        G = np.conj(X) * E / (U + 1e-6)
        
        # CG-inspired momentum update
        # p = G + beta * p_prev (momentum in gradient direction)
        momentum = G + momentum_decay * momentum
        
        # Normalize momentum to prevent explosion
        momentum_mag = np.abs(momentum)
        max_mag = 10.0
        momentum = np.where(momentum_mag > max_mag, 
                           momentum * (max_mag / (momentum_mag + 1e-10)),
                           momentum)
        
        # Compute step size (normalized)
        alpha = mu / (X2 + 1e-6)
        
        # Update filter along momentum direction
        H = H + alpha * momentum
        
        # Apply constraint
        if constrain:
            p_idx = n % N
            h = ifft(H[p_idx])
            h[M:] = 0
            H[p_idx] = fft(h)
    
    return e

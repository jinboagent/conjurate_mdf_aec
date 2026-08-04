
"""
Adapters for connecting existing algorithms to the harness interface.

These adapters wrap algorithm implementations to implement the
AdaptiveFilter interface required by the test harness.
"""

import numpy as np
from ..core.interfaces import AdaptiveFilter


class RLSBishengMDFAdapter(AdaptiveFilter):
    """
    Adapter for RLSBishengMDF to implement AdaptiveFilter interface.
    
    This adapter wraps the RLS Bisheng MDF algorithm to work with
    the test harness. It handles the conversion between time-domain
    blocks (harness standard) and frequency-domain (RLS internal).
    
    Parameters
    ----------
    n_g : int
        Number of delay blocks (default: 64)
    fft_size : int
        FFT size (default: 256)
    mu : float
        Step size / learning rate (default: 0.03)
    beta : float
        Forgetting factor (default: 0.97)
    nchan : int
        Number of microphone channels (default: 1)
        
    Example
    -------
    >>> from harness.adapters import RLSBishengMDFAdapter
    >>> adapter = RLSBishengMDFAdapter(n_g=64, fft_size=256, mu=0.03)
    >>> e = adapter.filt(x_block, d_block)
    >>> adapter.update(e)
    """
    
    def __init__(
        self,
        n_g: int = 64,
        fft_size: int = 256,
        mu: float = 0.03,
        beta: float = 0.97,
        nchan: int = 1
    ):
        from ..rls_bisheng_mdf import RLSBishengMDF
        
        self.n_g = n_g
        self.fft_size = fft_size
        self.mu = mu
        self.beta = beta
        self.nchan = nchan
        self.nbin = fft_size // 2 + 1
        
        # Initialize RLS filter
        self.rls_filter = RLSBishengMDF(
            NCHAN=nchan,
            NBIN=self.nbin,
            N_G=n_g,
            alpha=mu,
            beta=beta,
            bin_lim=self.nbin,
            Nrxref=1
        )
        
        # Time-domain buffers for overlap-save
        self.x_old = np.zeros(fft_size, dtype=np.float64)
        self.d_old = np.zeros(fft_size, dtype=np.float64)
        
    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Filter input block using RLS Bisheng MDF.
        
        Parameters
        ----------
        x : ndarray
            Reference signal block
        d : ndarray
            Desired signal block
            
        Returns
        -------
        e : ndarray
            Error signal (echo-cancelled output)
        """
        # Concatenate with old buffer
        x_now = np.concatenate((self.x_old, x))
        d_now = np.concatenate((self.d_old, d))
        
        # FFT to frequency domain
        X = np.fft.rfft(x_now)
        D = np.fft.rfft(d_now)
        
        # Reshape for RLS: [nbin, 1]
        X = X.reshape(-1, 1)
        D = D.reshape(-1, 1)
        
        # Apply RLS echo cancellation
        E = self.rls_filter.apply(D, X)
        
        # Convert to time domain
        e_time = np.fft.irfft(E[:, 0], n=len(x_now))
        
        # Update buffers
        self.x_old = x.copy()
        self.d_old = d.copy()
        
        # Return last portion (overlap-save)
        # Valid output is at the end of the IFFT
        step_size = len(x)
        e = e_time[-step_size:]
        
        return e
    
    def update(self, e: np.ndarray) -> None:
        """
        Update filter coefficients (no-op for RLS).
        
        RLS Bisheng MDF updates weights internally in apply(),
        so this method is a no-op for API compatibility.
        """
        pass
    
    def reset(self) -> None:
        """Reset filter state."""
        self.rls_filter.reset()
        self.x_old = np.zeros(self.fft_size, dtype=np.float64)
        self.d_old = np.zeros(self.fft_size, dtype=np.float64)

    def get_echo_path(self) -> np.ndarray:
        """
        Extract estimated echo path from RLS filter weights.

        The RLS weights are [nbin, N_G, nchan] per reference channel.
        For each partition, IFFT across frequency bins gives the
        time-domain impulse response contribution.

        Returns
        -------
        ndarray
            Estimated echo path impulse response
        """
        w = self.rls_filter.w_last[0]  # [nbin, N_G, nchan]
        # Average across channels, take IFFT across bins
        w_avg = np.mean(w, axis=2)  # [nbin, N_G]
        h_partitions = []
        for p in range(self.rls_filter.N_G):
            h_p = np.fft.irfft(w_avg[:, p], n=self.fft_size)
            h_partitions.append(h_p[:self.fft_size // 2])
        return np.concatenate(h_partitions)


class PFADFMDFCGAdapter(AdaptiveFilter):
    """
    Adapter for PFADFMDFCG to implement AdaptiveFilter interface.
    
    This is a thin wrapper since PFADFMDFCG already implements
    the correct interface.
    
    Parameters
    ----------
    n : int
        Number of partitions (default: 64)
    winlen : int
        Block size (default: 256)
    mu : float
        Step size (default: 0.03)
    beta : float
        Forgetting factor (default: 0.97)
    nchan : int
        Number of channels (default: 1)
        
    Example
    -------
    >>> from harness.adapters import PFADFMDFCGAdapter
    >>> adapter = PFADFMDFCGAdapter(n=64, winlen=256, mu=0.03)
    >>> e = adapter.filt(x_block, d_block)
    >>> adapter.update(e)
    """
    
    def __init__(
        self,
        n: int = 64,
        winlen: int = 256,
        mu: float = 0.03,
        beta: float = 0.97,
        nchan: int = 1
    ):
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from pfadf_mdf_cg import PFADFMDFCG
        
        self.n = n
        self.winlen = winlen
        self.mu = mu
        self.beta = beta
        self.nchan = nchan
        
        # Initialize filter
        self.filter = PFADFMDFCG(
            N=n,
            winlen=winlen,
            mu=mu,
            beta=beta
        )
        
    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Filter input block.
        
        Parameters
        ----------
        x : ndarray
            Reference signal block
        d : ndarray
            Desired signal block
            
        Returns
        -------
        e : ndarray
            Error signal
        """
        return self.filter.filt(x, d)
    
    def update(self, e: np.ndarray) -> None:
        """
        Update filter coefficients.
        
        Parameters
        ----------
        e : ndarray
            Error signal
        """
        self.filter.update(e)
    
    def reset(self) -> None:
        """Reset filter state."""
        self.filter.reset()

    def get_echo_path(self) -> np.ndarray:
        """
        Extract estimated echo path from filter weights.

        Returns
        -------
        ndarray
            Estimated echo path impulse response
        """
        return self.filter.get_echo_path()


# Convenience function to create standard adapters
def create_standard_rlsmdf_adapter(
    n_g: int = 64,
    fft_size: int = 256,
    mu: float = 0.03
) -> RLSBishengMDFAdapter:
    """
    Create a standard RLS Bisheng MDF adapter with recommended parameters.
    
    Parameters
    ----------
    n_g : int
        Number of delay blocks
    fft_size : int
        FFT size
    mu : float
        Step size
        
    Returns
    -------
    RLSBishengMDFAdapter
        Configured adapter instance
    """
    return RLSBishengMDFAdapter(n_g=n_g, fft_size=fft_size, mu=mu)


def create_standard_pfadf_mdf_cg_adapter(
    n: int = 64,
    winlen: int = 256,
    mu: float = 0.03
) -> PFADFMDFCGAdapter:
    """
    Create a standard PFADF MDF CG adapter with recommended parameters.
    
    Parameters
    ----------
    n : int
        Number of partitions
    winlen : int
        Block size
    mu : float
        Step size
        
    Returns
    -------
    PFADFMDFCGAdapter
        Configured adapter instance
    """
    return PFADFMDFCGAdapter(n=n, winlen=winlen, mu=mu)

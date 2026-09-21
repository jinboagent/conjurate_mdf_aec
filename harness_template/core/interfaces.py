"""
Layer 1: Algorithm Interfaces

Standard interfaces that all algorithms must implement.
The harness uses these to test algorithms interchangeably.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class AdaptiveFilter(ABC):
    """
    Base class for time-domain block-based adaptive filters.

    Implement filt() and update() to create a new algorithm.
    Optionally implement get_echo_path() for echo path verification.
    """

    @abstractmethod
    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Filter one block of input.

        Parameters
        ----------
        x : ndarray, shape (M,)
            Reference signal block (far-end, loudspeaker)
        d : ndarray, shape (M,)
            Desired signal block (microphone with echo)

        Returns
        -------
        e : ndarray, shape (M,)
            Error signal (echo-cancelled output)
        """
        pass

    @abstractmethod
    def update(self, e: np.ndarray) -> None:
        """
        Update filter coefficients based on error signal.

        Parameters
        ----------
        e : ndarray, shape (M,)
            Error signal from filt()
        """
        pass

    def get_echo_path(self) -> Optional[np.ndarray]:
        """
        Return estimated echo path impulse response.

        Override this to enable echo path comparison metrics.

        Returns
        -------
        h_est : ndarray or None
            Estimated impulse response, or None if not available
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state to initial conditions."""
        pass


class FrequencyDomainFilter(ABC):
    """
    Base class for frequency-domain filters.

    Use this when your algorithm operates on pre-transformed
    frequency-domain signals (e.g., Conjugate Gradient MDF).
    """

    @abstractmethod
    def apply(self, Y: np.ndarray, Y_rx: np.ndarray) -> np.ndarray:
        """
        Apply filter to frequency-domain input.

        Parameters
        ----------
        Y : ndarray, shape (NBIN, NCHAN)
            Microphone input in frequency domain
        Y_rx : ndarray, shape (NBIN, Nrxref)
            Reference signal in frequency domain

        Returns
        -------
        E : ndarray, shape (NBIN, NCHAN)
            Echo cancelled output in frequency domain
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state."""
        pass


class BlockWrapper:
    """
    Wraps a FrequencyDomainFilter with time-domain overlap-save.

    Converts between time-domain blocks and frequency-domain frames,
    so FrequencyDomainFilters can be tested with the same harness.
    """

    def __init__(self, fd_filter: FrequencyDomainFilter, fft_size: int, overlap: float = 0.75):
        """
        Parameters
        ----------
        fd_filter : FrequencyDomainFilter
            The frequency-domain filter to wrap
        fft_size : int
            FFT size (e.g., 256 or 512)
        overlap : float
            Overlap ratio (0.75 = 75% overlap)
        """
        self.filter = fd_filter
        self.fft_size = fft_size
        self.step_size = int(fft_size * (1 - overlap))
        self.x_old = np.zeros(fft_size - self.step_size)

    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Process one step_size block via overlap-save FFT."""
        # Build overlap-save input
        x_now = np.concatenate([self.x_old, x[:self.step_size]])
        d_now = np.concatenate([self.x_old, d[:self.step_size]])

        # FFT
        X = np.fft.rfft(x_now).reshape(-1, 1)
        D = np.fft.rfft(d_now).reshape(-1, 1)

        # Apply frequency-domain filter
        E = self.filter.apply(D, X)

        # IFFT and extract valid output
        e_time = np.fft.irfft(E[:, 0], n=self.fft_size)
        self.x_old = x[:self.step_size]

        return e_time[-self.step_size:]

    def update(self, e: np.ndarray) -> None:
        """No-op — frequency-domain filters update internally in apply()."""
        pass

    def reset(self) -> None:
        """Reset wrapper and underlying filter."""
        self.x_old = np.zeros(self.fft_size - self.step_size)
        self.filter.reset()

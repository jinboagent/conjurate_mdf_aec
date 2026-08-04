"""
Abstract base classes for adaptive filters.

All adaptive filter implementations should inherit from AdaptiveFilter
to be compatible with the test harness.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class AdaptiveFilter(ABC):
    """
    Abstract base class for adaptive filters.
    
    This interface defines the standard methods that all adaptive filters
    must implement to work with the test harness.
    
    Example usage:
        class MyFilter(AdaptiveFilter):
            def __init__(self, **kwargs):
                # Initialize filter parameters
                pass
            
            def filt(self, x, d):
                # Process input block, return error
                pass
            
            def update(self, e):
                # Update filter coefficients
                pass
    """
    
    @abstractmethod
    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Filter input block and compute error signal.
        
        Parameters
        ----------
        x : ndarray
            Reference signal block (far-end input)
        d : ndarray
            Desired signal block (microphone with echo)
            
        Returns
        -------
        e : ndarray
            Error signal (echo-cancelled output)
        """
        pass
    
    @abstractmethod
    def update(self, e: np.ndarray) -> None:
        """
        Update filter coefficients based on error signal.
        
        Parameters
        ----------
        e : ndarray
            Error signal from filt() method
        """
        pass
    
    def reset(self) -> None:
        """
        Reset filter state to initial conditions.

        Optional method for filters that maintain internal state.
        """
        pass

    def get_echo_path(self) -> Optional[np.ndarray]:
        """
        Return estimated echo path impulse response.

        Optional method for filters that estimate the acoustic echo path.
        Returns None if the filter does not support echo path extraction.

        Returns
        -------
        ndarray or None
            Estimated echo path impulse response (time-domain), or None
        """
        return None


class BlockAdaptiveFilter(AdaptiveFilter):
    """
    Base class for block-based adaptive filters.
    
    Provides common block processing functionality including
    overlap-save/overlap-add handling.
    """
    
    def __init__(self, fft_size: int = 256, overlap: float = 0.75):
        """
        Initialize block adaptive filter.
        
        Parameters
        ----------
        fft_size : int
            FFT size (block size)
        overlap : float
            Overlap ratio (0.0 to 1.0)
        """
        self.fft_size = fft_size
        self.overlap = overlap
        self.step_size = int(fft_size * (1 - overlap))
        self.nbin = fft_size // 2 + 1
        
    def process_signal(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Process entire signal using block-based filtering.
        
        Parameters
        ----------
        x : ndarray
            Full reference signal
        d : ndarray
            Full desired signal
            
        Returns
        -------
        e : ndarray
            Full error signal (same length as input)
        """
        n_samples = len(x)
        e = np.zeros(n_samples)
        
        n_frames = (n_samples - self.fft_size) // self.step_size + 1
        
        for i in range(n_frames):
            start_idx = i * self.step_size
            end_idx = start_idx + self.fft_size
            
            x_block = x[start_idx:end_idx]
            d_block = d[start_idx:end_idx]
            
            e_block = self.filt(x_block, d_block)
            self.update(e_block)
            
            # Overlap-save: valid output is at the end of the block
            e[start_idx + self.fft_size - self.step_size:end_idx] = e_block[-self.step_size:]
        
        return e

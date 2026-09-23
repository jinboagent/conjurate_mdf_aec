
"""
Acoustic Echo Path Generator

Generates realistic acoustic echo impulse responses for typical office room environments.
Supports both time-domain and frequency-domain representations.

Parameters:
    - Sampling frequency: 16 kHz
    - Filter length: 500 ms (8000 samples)
    - Initial delay: 45 ms (720 samples) - direct path delay
    - Environment: Typical office room with moderate reverberation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, irfft, rfftfreq
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RoomParameters:
    """Parameters defining the acoustic room environment."""
    sampling_rate: int = 16000
    filter_length_ms: int = 500
    initial_delay_ms: float = 45.0
    reverberation_time_ms: float = 300.0  # RT60 equivalent
    num_reflections: int = 50
    decay_factor: float = 0.95  # Exponential decay rate
    
    @property
    def filter_length_samples(self) -> int:
        return int(self.sampling_rate * self.filter_length_ms / 1000)
    
    @property
    def initial_delay_samples(self) -> int:
        return int(self.sampling_rate * self.initial_delay_ms / 1000)
    
    @property
    def reverberation_time_samples(self) -> int:
        return int(self.sampling_rate * self.reverberation_time_ms / 1000)


def generate_time_domain_echo_path(
    params: Optional[RoomParameters] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a causal acoustic echo impulse response in time domain.
    
    The echo path models:
    1. Direct path delay (45ms - sound travel from loudspeaker to microphone)
    2. Early reflections (discrete echoes from walls, desk, etc.)
    3. Late reverberation (dense exponential decay tail)
    
    Parameters
    ----------
    params : RoomParameters, optional
        Room acoustic parameters. Uses defaults if not provided.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    h : ndarray
        Time-domain impulse response (filter_length_samples,)
        
    Example
    -------
    >>> h = generate_time_domain_echo_path()
    >>> print(f"Filter length: {len(h)} samples")
    >>> print(f"Initial delay: {params.initial_delay_samples} samples")
    """
    if params is None:
        params = RoomParameters()
    
    if seed is not None:
        np.random.seed(seed)
    
    filter_len = params.filter_length_samples
    delay = params.initial_delay_samples
    
    # Initialize impulse response
    h = np.zeros(filter_len)
    
    # =========================================================================
    # 1. Direct Path (attenuated delayed impulse)
    # =========================================================================
    # Direct path has the strongest amplitude
    direct_path_amplitude = 0.6
    h[delay] = direct_path_amplitude
    
    # =========================================================================
    # 2. Early Reflections (discrete echoes)
    # =========================================================================
    # Model first-order reflections from walls, floor, ceiling, desk
    # These arrive within first 50-100ms after direct path
    
    num_early = 8
    early_times = np.sort(np.random.uniform(
        delay + 1, 
        delay + int(0.05 * params.sampling_rate),  # Within 50ms
        num_early
    ))
    early_amplitudes = direct_path_amplitude * np.random.uniform(0.1, 0.4, num_early)
    early_phases = np.random.choice([-1, 1], num_early)  # Phase inversions
    
    for time, amp, phase in zip(early_times, early_amplitudes, early_phases):
        time = int(time)
        if time < filter_len:
            h[time] += phase * amp
    
    # =========================================================================
    # 3. Late Reverberation (exponential decay with noise)
    # =========================================================================
    # Dense reflections that decay exponentially
    # Modeled as filtered noise starting after early reflections
    
    reverb_start = int(delay + 0.05 * params.sampling_rate)  # After 50ms
    reverb_length = filter_len - reverb_start
    
    if reverb_length > 0:
        # Generate exponentially decaying noise
        t = np.arange(reverb_length) / params.sampling_rate
        decay = np.exp(-t * 6.9 / (params.reverberation_time_ms / 1000))  # RT60 decay
        
        # Shaped noise for late reverberation
        noise = np.random.randn(reverb_length)
        
        # Color the noise with a low-pass filter (rooms absorb high frequencies)
        b, a = signal.butter(2, 0.3, btype='low')
        noise = signal.lfilter(b, a, noise)
        
        # Apply exponential decay
        reverb = noise * decay * 0.15  # Scale to be weaker than early reflections
        
        # Smooth transition from early to late
        fade_in_len = int(0.01 * params.sampling_rate)  # 10ms fade-in
        fade_in = np.linspace(0, 1, fade_in_len)
        reverb[:fade_in_len] *= fade_in
        
        h[reverb_start:reverb_start + reverb_length] += reverb
    
    # =========================================================================
    # 4. Additional Discrete Reflections (secondary reflections)
    # =========================================================================
    # Add some stronger secondary reflections throughout the response
    
    num_secondary = 15
    secondary_times = np.sort(np.random.uniform(
        delay + int(0.05 * params.sampling_rate),
        filter_len - 1,
        num_secondary
    ))
    secondary_amplitudes = direct_path_amplitude * np.random.uniform(0.05, 0.2, num_secondary)
    secondary_phases = np.random.choice([-1, 1], num_secondary)
    
    for time, amp, phase in zip(secondary_times, secondary_amplitudes, secondary_phases):
        time = int(time)
        if time < filter_len:
            # Spread each reflection slightly (diffuse reflection)
            spread = 3
            for offset in range(-spread, spread + 1):
                idx = time + offset
                if 0 <= idx < filter_len:
                    window = np.hanning(2 * spread + 1)[offset + spread]
                    h[idx] += phase * amp * window * 0.5
    
    # =========================================================================
    # 5. Apply Overall Envelope (ensure smooth decay)
    # =========================================================================
    # Gentle fade-out at the end to avoid truncation artifacts
    fade_out_start = int(0.9 * filter_len)
    fade_out = np.linspace(1, 0, filter_len - fade_out_start)
    h[fade_out_start:] *= fade_out
    
    return h


def generate_frequency_domain_echo_path(
    h_time: Optional[np.ndarray] = None,
    params: Optional[RoomParameters] = None,
    nfft: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate frequency-domain representation of the echo path.
    
    Parameters
    ----------
    h_time : ndarray, optional
        Time-domain impulse response. Generated if not provided.
    params : RoomParameters, optional
        Room acoustic parameters.
    nfft : int, optional
        FFT size. Defaults to next power of 2 >= filter_length.
        
    Returns
    -------
    H : ndarray
        Frequency response (complex)
    freqs : ndarray
        Frequency bins (Hz)
        
    Example
    -------
    >>> H, freqs = generate_frequency_domain_echo_path()
    >>> magnitude = np.abs(H)
    >>> phase = np.angle(H)
    """
    if params is None:
        params = RoomParameters()
    
    if h_time is None:
        h_time = generate_time_domain_echo_path(params)
    
    # Determine FFT size
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(len(h_time))))
    
    # Compute frequency response
    H = rfft(h_time, n=nfft)
    freqs = rfftfreq(nfft, 1.0 / params.sampling_rate)
    
    return H, freqs


def plot_echo_path_time_domain(
    h: np.ndarray,
    params: Optional[RoomParameters] = None,
    title: str = "Acoustic Echo Path (Time Domain)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the time-domain echo path impulse response.
    
    Parameters
    ----------
    h : ndarray
        Impulse response
    params : RoomParameters, optional
        Room parameters for annotation
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if params is None:
        params = RoomParameters()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Time axis in samples and milliseconds
    samples = np.arange(len(h))
    time_ms = samples / params.sampling_rate * 1000
    
    # =========================================================================
    # Plot 1: Full impulse response
    # =========================================================================
    ax = axes[0]
    ax.plot(time_ms, h, linewidth=0.8, color='darkblue')
    ax.axvline(params.initial_delay_ms, color='red', linestyle='--', 
               linewidth=2, label=f'Direct path ({params.initial_delay_ms:.1f} ms)')
    ax.axvline(params.initial_delay_ms + 50, color='orange', linestyle=':', 
               linewidth=2, label='Early/Late boundary (50 ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{title}\nFull Impulse Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params.filter_length_ms)
    
    # =========================================================================
    # Plot 2: Zoomed view of early part (direct path + early reflections)
    # =========================================================================
    ax = axes[1]
    early_end = int(min(params.initial_delay_samples + 200, len(h)))
    ax.plot(time_ms[:early_end], h[:early_end], linewidth=1.2, color='darkgreen')
    ax.axvline(params.initial_delay_ms, color='red', linestyle='--', 
               linewidth=2, label=f'Direct path ({params.initial_delay_ms:.1f} ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Early Reflections (Direct Path + First 50ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(100, params.filter_length_ms))
    
    # =========================================================================
    # Plot 3: Energy decay curve (Schroeder integral)
    # =========================================================================
    ax = axes[2]
    energy = h ** 2
    # Reverse cumulative sum (Schroeder integration)
    energy_decay = np.cumsum(energy[::-1])[::-1]
    energy_decay_db = 10 * np.log10(energy_decay / (np.max(energy_decay) + 1e-10) + 1e-10)
    
    ax.plot(time_ms, energy_decay_db, linewidth=1.5, color='purple')
    ax.axvline(params.initial_delay_ms, color='red', linestyle='--', 
               linewidth=2, label=f'Direct path ({params.initial_delay_ms:.1f} ms)')
    
    # Mark -60 dB point (RT60 estimate)
    idx_60db = np.where(energy_decay_db <= -60)[0]
    if len(idx_60db) > 0:
        rt60_est = time_ms[idx_60db[0]]
        ax.axvline(rt60_est, color='green', linestyle=':', 
                   linewidth=2, label=f'RT60 est. ({rt60_est:.1f} ms)')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Energy (dB)')
    ax.set_title('Energy Decay Curve (Schroeder Integral)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params.filter_length_ms)
    ax.set_ylim(-80, 5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved time-domain plot to: {save_path}")
    
    return fig


def plot_echo_path_frequency_domain(
    H: np.ndarray,
    freqs: np.ndarray,
    params: Optional[RoomParameters] = None,
    title: str = "Acoustic Echo Path (Frequency Domain)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the frequency-domain echo path response.
    
    Parameters
    ----------
    H : ndarray
        Frequency response (complex)
    freqs : ndarray
        Frequency bins (Hz)
    params : RoomParameters, optional
        Room parameters
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if params is None:
        params = RoomParameters()
    
    # Compute magnitude and phase
    magnitude = np.abs(H)
    magnitude_db = 20 * np.log10(magnitude / (np.max(magnitude) + 1e-10) + 1e-10)
    phase = np.angle(H)
    group_delay = -np.diff(phase) / np.diff(freqs)  # Approximate group delay
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # =========================================================================
    # Plot 1: Magnitude response
    # =========================================================================
    ax = axes[0]
    ax.plot(freqs, magnitude_db, linewidth=1.2, color='darkblue')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'{title}\nMagnitude Response')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params.sampling_rate / 2)
    ax.set_ylim(-60, 5)
    
    # Mark key frequency regions
    ax.axvspan(0, 300, alpha=0.2, color='green', label='Low freq (<300 Hz)')
    ax.axvspan(300, 3000, alpha=0.2, color='yellow', label='Speech (300-3k Hz)')
    ax.axvspan(3000, 8000, alpha=0.2, color='red', label='High freq (>3k Hz)')
    ax.legend(loc='upper right')
    
    # =========================================================================
    # Plot 2: Phase response
    # =========================================================================
    ax = axes[1]
    ax.plot(freqs, phase, linewidth=0.8, color='darkgreen')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Phase Response')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params.sampling_rate / 2)
    
    # =========================================================================
    # Plot 3: Group delay
    # =========================================================================
    ax = axes[2]
    freqs_center = (freqs[:-1] + freqs[1:]) / 2
    ax.plot(freqs_center, group_delay, linewidth=0.8, color='purple')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_title('Group Delay')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, params.sampling_rate / 2)
    
    # Mark the direct path delay
    ax.axhline(params.initial_delay_samples, color='red', linestyle='--', 
               linewidth=2, label=f'Direct path delay ({params.initial_delay_samples} samples)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved frequency-domain plot to: {save_path}")
    
    return fig


def generate_and_plot_echo_path(
    seed: Optional[int] = None,
    save_plots: bool = True,
    output_dir: str = "."
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate echo path and create all visualization plots.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory to save plots
        
    Returns
    -------
    h : ndarray
        Time-domain impulse response
    H : ndarray
        Frequency response
    freqs : ndarray
        Frequency bins
    """
    import os
    
    params = RoomParameters()
    
    print("=" * 60)
    print("Acoustic Echo Path Generator")
    print("=" * 60)
    print(f"Sampling frequency: {params.sampling_rate} Hz")
    print(f"Filter length: {params.filter_length_ms} ms ({params.filter_length_samples} samples)")
    print(f"Initial delay: {params.initial_delay_ms} ms ({params.initial_delay_samples} samples)")
    print(f"Reverberation time: {params.reverberation_time_ms} ms")
    print(f"Causal: Yes (h[n] = 0 for n < {params.initial_delay_samples})")
    print("=" * 60)
    
    # Generate time-domain echo path
    print("\nGenerating time-domain echo path...")
    h = generate_time_domain_echo_path(params, seed=seed)
    
    # Generate frequency-domain representation
    print("Generating frequency-domain representation...")
    H, freqs = generate_frequency_domain_echo_path(h, params)
    
    # Print statistics
    print(f"\nEcho Path Statistics:")
    print(f"  Max amplitude: {np.max(np.abs(h)):.6f}")
    print(f"  Energy: {np.sum(h**2):.6f}")
    print(f"  Center of mass: {np.sum(np.arange(len(h)) * h**2) / np.sum(h**2):.1f} samples")
    
    # Plot time domain
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        time_plot_path = os.path.join(output_dir, "echo_path_time_domain.png")
        plot_echo_path_time_domain(h, params, save_path=time_plot_path)
        
        freq_plot_path = os.path.join(output_dir, "echo_path_frequency_domain.png")
        plot_echo_path_frequency_domain(H, freqs, params, save_path=freq_plot_path)
    else:
        plot_echo_path_time_domain(h, params)
        plot_echo_path_frequency_domain(H, freqs, params)
        plt.show()
    
    print(f"\n✓ Echo path generation complete!")
    
    return h, H, freqs


def demo():
    """Run a demonstration of the echo path generator."""
    # Generate with fixed seed for reproducibility
    h, H, freqs = generate_and_plot_echo_path(seed=42, save_plots=True)
    
    print("\n" + "=" * 60)
    print("Demo complete! Check the generated plots.")
    print("=" * 60)


if __name__ == "__main__":
    demo()

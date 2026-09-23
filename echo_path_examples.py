
"""
Examples of using the echo path generator for different scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from echo_path_generator import (
    RoomParameters,
    generate_time_domain_echo_path,
    generate_frequency_domain_echo_path,
    plot_echo_path_time_domain,
    plot_echo_path_frequency_domain
)


def example_office_room():
    """Example: Typical office room with 45ms delay."""
    print("\n" + "=" * 60)
    print("Example 1: Typical Office Room")
    print("=" * 60)
    
    params = RoomParameters(
        sampling_rate=16000,
        filter_length_ms=500,
        initial_delay_ms=45.0,
        reverberation_time_ms=300
    )
    
    h = generate_time_domain_echo_path(params, seed=42)
    
    print(f"Filter length: {len(h)} samples ({len(h)/16000*1000:.1f} ms)")
    print(f"Direct path delay: {params.initial_delay_samples} samples ({params.initial_delay_ms} ms)")
    print(f"Max amplitude: {np.max(np.abs(h)):.4f}")
    
    # Plot
    fig = plot_echo_path_time_domain(h, params, title="Office Room Echo Path")
    plt.show()
    
    return h, params


def example_large_conference_room():
    """Example: Large conference room with longer delay and reverb."""
    print("\n" + "=" * 60)
    print("Example 2: Large Conference Room")
    print("=" * 60)
    
    params = RoomParameters(
        sampling_rate=16000,
        filter_length_ms=800,  # Longer filter
        initial_delay_ms=60.0,  # Larger room = longer delay
        reverberation_time_ms=500  # More reverberation
    )
    
    h = generate_time_domain_echo_path(params, seed=123)
    
    print(f"Filter length: {len(h)} samples ({len(h)/16000*1000:.1f} ms)")
    print(f"Direct path delay: {params.initial_delay_samples} samples ({params.initial_delay_ms} ms)")
    print(f"Reverberation time: {params.reverberation_time_ms} ms")
    
    return h, params


def example_small_huddle_room():
    """Example: Small huddle room with short delay."""
    print("\n" + "=" * 60)
    print("Example 3: Small Huddle Room")
    print("=" * 60)
    
    params = RoomParameters(
        sampling_rate=16000,
        filter_length_ms=300,  # Shorter filter
        initial_delay_ms=20.0,  # Small room = shorter delay
        reverberation_time_ms=150  # Less reverberation
    )
    
    h = generate_time_domain_echo_path(params, seed=456)
    
    print(f"Filter length: {len(h)} samples ({len(h)/16000*1000:.1f} ms)")
    print(f"Direct path delay: {params.initial_delay_samples} samples ({params.initial_delay_ms} ms)")
    
    return h, params


def example_compare_scenarios():
    """Compare all three scenarios side by side."""
    print("\n" + "=" * 60)
    print("Example 4: Comparing Different Room Environments")
    print("=" * 60)
    
    # Generate all three scenarios
    h_office, params_office = example_office_room()
    h_conference, params_conference = example_large_conference_room()
    h_huddle, params_huddle = example_small_huddle_room()
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    max_len = max(len(h_office), len(h_conference), len(h_huddle))
    
    # Pad to same length for comparison
    h_office_padded = np.zeros(max_len)
    h_office_padded[:len(h_office)] = h_office
    
    h_conference_padded = np.zeros(max_len)
    h_conference_padded[:len(h_conference)] = h_conference
    
    h_huddle_padded = np.zeros(max_len)
    h_huddle_padded[:len(h_huddle)] = h_huddle
    
    time_ms = np.arange(max_len) / 16000 * 1000
    
    # Office
    axes[0].plot(time_ms, h_office_padded, linewidth=1.0, color='darkblue')
    axes[0].set_title('Office Room (45ms delay, 300ms reverb)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 500)
    
    # Conference
    axes[1].plot(time_ms, h_conference_padded, linewidth=1.0, color='darkgreen')
    axes[1].set_title('Large Conference Room (60ms delay, 500ms reverb)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 800)
    
    # Huddle
    axes[2].plot(time_ms, h_huddle_padded, linewidth=1.0, color='darkred')
    axes[2].set_title('Small Huddle Room (20ms delay, 150ms reverb)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (ms)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 300)
    
    plt.tight_layout()
    plt.savefig('echo_path_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to: echo_path_comparison.png")
    plt.show()


def example_frequency_analysis():
    """Example: Frequency domain analysis."""
    print("\n" + "=" * 60)
    print("Example 5: Frequency Domain Analysis")
    print("=" * 60)
    
    params = RoomParameters(
        sampling_rate=16000,
        filter_length_ms=500,
        initial_delay_ms=45.0
    )
    
    h = generate_time_domain_echo_path(params, seed=42)
    H, freqs = generate_frequency_domain_echo_path(h, params)
    
    # Analyze frequency characteristics
    magnitude = np.abs(H)
    magnitude_db = 20 * np.log10(magnitude / (np.max(magnitude) + 1e-10) + 1e-10)
    
    # Find frequency bands
    low_idx = (freqs >= 0) & (freqs < 300)
    mid_idx = (freqs >= 300) & (freqs < 3000)
    high_idx = (freqs >= 3000) & (freqs < 8000)
    
    print(f"\nFrequency Band Analysis:")
    print(f"  Low freq (0-300 Hz) avg magnitude:  {np.mean(magnitude_db[low_idx]):.2f} dB")
    print(f"  Mid freq (300-3k Hz) avg magnitude: {np.mean(magnitude_db[mid_idx]):.2f} dB")
    print(f"  High freq (3k-8k Hz) avg magnitude: {np.mean(magnitude_db[high_idx]):.2f} dB")
    
    # Plot
    plot_echo_path_frequency_domain(H, freqs, params)
    plt.show()


if __name__ == "__main__":
    # Run specific examples or all
    example_office_room()
    # example_compare_scenarios()
    # example_frequency_analysis()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

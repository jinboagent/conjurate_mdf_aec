"""
Visualization of RLS Bisheng MDF Weight Convergence Process

This script generates detailed visualizations showing how the filter weights
converge during the RLS adaptation process.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import soundfile as sf
import librosa
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rls_bisheng_mdf import RLSBishengMDF


def stft_analysis(sig, n_fft=256, hop_length=64):
    """STFT analysis"""
    D = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length, center=False)
    return D


def visualize_weight_convergence():
    """
    Visualize weight convergence during RLS adaptation
    """
    print("=" * 70)
    print("RLS Bisheng MDF - Weight Convergence Visualization")
    print("=" * 70)
    
    # Configuration
    sr = 16000
    duration = 3.0
    n_fft = 256
    hop_length = 64
    nbin = n_fft // 2 + 1
    N_G = 4
    
    # Create test signal - sine wave with amplitude envelope
    t = np.arange(int(sr * duration)) / sr
    envelope = np.exp(-t * 0.3)  # Decay
    ref_signal = 0.3 * np.sin(2 * np.pi * 440 * t) * envelope
    
    # Identical microphone signal (echo path = 1)
    mic_signal = ref_signal.copy()
    
    print(f"\nSignal: 440 Hz tone, {duration}s, decay envelope")
    print(f"Echo path: 1.0 (identical signals)")
    
    # STFT
    X_ref = stft_analysis(ref_signal, n_fft, hop_length)
    X_mic = stft_analysis(mic_signal, n_fft, hop_length)
    n_frames = X_ref.shape[1]
    
    print(f"STFT: {n_frames} frames, {nbin} bins, N_G={N_G}")
    
    # Create RLS filter
    rls = RLSBishengMDF(
        NCHAN=1, NBIN=nbin, N_G=N_G,
        N_cnt_loud=1, N_wait=10,
        thr_loud=1e-6, alpha=0.05, beta=0.97,
        bin_lim=nbin, Nrxref=1
    )
    
    # Track convergence
    weight_history = []      # Weight magnitude per frame
    ratio_history = []       # rcross/Rtoe ratio
    output_power_history = []  # Output power
    rtoe_history = []        # Rtoe magnitude
    rcross_history = []      # rcross magnitude
    tap_distribution = []    # Weight distribution across taps
    
    # Process and track
    print("\nProcessing and tracking convergence...")
    
    for frame_idx in range(n_frames):
        Y_rx = X_ref[:, frame_idx:frame_idx+1]
        Y = X_mic[:, frame_idx:frame_idx+1]
        E = rls.apply(Y, Y_rx)
        
        # Track weight magnitude (averaged over frequency bins)
        w = rls.w[0]  # [nbin, N_G, 1]
        w_mag_per_tap = np.mean(np.abs(w), axis=0)[:, 0]  # [N_G]
        weight_history.append(w_mag_per_tap.copy())
        
        # Track ratio
        rtoe_val = np.mean(np.abs(rls.Rtoe[0]))
        rcross_val = np.mean(np.abs(rls.rcross[0][:, :, 0]))
        ratio_history.append(rcross_val / rtoe_val if rtoe_val > 1e-10 else 0)
        rtoe_history.append(rtoe_val)
        rcross_history.append(rcross_val)
        
        # Track output power
        output_power_history.append(np.mean(np.abs(E)**2))
        
        # Track tap distribution
        tap_distribution.append(w_mag_per_tap / (np.sum(w_mag_per_tap) + 1e-10))
        
        if frame_idx % 500 == 0:
            print(f"  Frame {frame_idx:4d}: ratio={ratio_history[-1]:.4f}, "
                  f"output_power={output_power_history[-1]:.6f}")
    
    weight_history = np.array(weight_history)  # [n_frames, N_G]
    tap_distribution = np.array(tap_distribution)
    
    print(f"  Complete: {n_frames} frames tracked")
    
    # =========================================================================
    # Create comprehensive visualization
    # =========================================================================
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    frame_axis = np.arange(n_frames)
    tap_axis = np.arange(N_G)
    
    # -------------------------------------------------------------------------
    # Plot 1: Weight magnitude convergence (all taps)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for tap in range(N_G):
        ax1.plot(frame_axis, weight_history[:, tap], label=f'Tap {tap}', linewidth=1.5)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Weight Magnitude')
    ax1.set_title('Weight Magnitude Convergence (All Taps)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_frames)
    
    # Annotate convergence phases
    ax1.axvspan(0, 100, alpha=0.2, color='yellow', label='Fast convergence')
    ax1.axvspan(100, 500, alpha=0.2, color='green', label='Steady state')
    ax1.axvspan(500, n_frames, alpha=0.2, color='blue', label='Tracking')
    
    # -------------------------------------------------------------------------
    # Plot 2: Tap 0 dominance over time
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    tap0_dominance = weight_history[:, 0] / (np.sum(weight_history, axis=1) + 1e-10)
    ax2.plot(frame_axis, tap0_dominance, linewidth=2, color='darkblue')
    ax2.axhline(y=1/N_G, color='gray', linestyle='--', label=f'Uniform ({1/N_G:.2f})')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='50% dominance')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Tap 0 Fraction')
    ax2.set_title('Tap 0 Dominance\n(Should approach 1.0 for no-delay echo)')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_frames)
    ax2.set_ylim(0, 1.0)
    
    # -------------------------------------------------------------------------
    # Plot 3: Output power convergence
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(frame_axis, output_power_history, linewidth=1.5, color='red')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Output Power (log scale)')
    ax3.set_title('Echo Cancellation Convergence\n(Output Power → 0)')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(0, n_frames)
    
    # Mark convergence point
    conv_threshold = 0.001
    conv_frames = np.where(np.array(output_power_history) < conv_threshold)[0]
    if len(conv_frames) > 0:
        conv_frame = conv_frames[0]
        ax3.axvline(x=conv_frame, color='green', linestyle='--', 
                   label=f'Converged @ frame {conv_frame}')
        ax3.legend(loc='upper right', fontsize=8)
    
    # -------------------------------------------------------------------------
    # Plot 4: rcross/Rtoe ratio convergence
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(frame_axis, ratio_history, linewidth=2, color='purple')
    ax4.axhline(y=1.0, color='green', linestyle='--', label='Expected (1.0)')
    ax4.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=1.1, color='orange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('rcross/Rtoe Ratio')
    ax4.set_title('Echo Path Identification\n(rcross/Rtoe → 1.0 for identical signals)')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, n_frames)
    ax4.set_ylim(0, 1.5)
    
    # -------------------------------------------------------------------------
    # Plot 5: Rtoe and rcross magnitude
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(frame_axis, rtoe_history, label='|Rtoe|', linewidth=1.5, color='blue')
    ax5.plot(frame_axis, rcross_history, label='|rcross|', linewidth=1.5, color='red')
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Magnitude')
    ax5.set_title('Correlation Growth\n(Both grow with signal power)')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, n_frames)
    
    # -------------------------------------------------------------------------
    # Plot 6: Weight distribution across taps (heatmap)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(tap_distribution.T, aspect='auto', origin='lower', 
                    cmap='YlOrRd', extent=[0, n_frames, 0, N_G])
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('Tap Index')
    ax6.set_title('Weight Distribution Heatmap\n(Brighter = Higher Magnitude)')
    ax6.set_yticks([0, 1, 2, 3])
    plt.colorbar(im, ax=ax6, label='Normalized Magnitude')
    
    # -------------------------------------------------------------------------
    # Plot 7-9: Weight state at different time points
    # -------------------------------------------------------------------------
    time_points = [0, 50, 200, 1000, 3000]
    time_labels = ['Frame 0\n(Initial)', 'Frame 50\n(Start)', 'Frame 200\n(Converging)', 
                   'Frame 1000\n(Steady)', 'Frame 3000\n(Tracking)']
    
    ax7 = fig.add_subplot(gs[2, :])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
    
    for i, (frame, label) in enumerate(zip(time_points, time_labels)):
        if frame < n_frames:
            ax7.plot(tap_axis, weight_history[frame], marker='o', markersize=8,
                    linewidth=2, color=colors[i], label=label)
    
    ax7.set_xlabel('Tap Index')
    ax7.set_ylabel('Weight Magnitude')
    ax7.set_title('Weight Magnitude Distribution at Different Time Points')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xticks([0, 1, 2, 3])
    
    # -------------------------------------------------------------------------
    # Plot 10: Weight phase distribution (at steady state)
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[3, 0])
    steady_frame = min(2000, n_frames - 1)
    w_steady = rls.w[0][:, :, 0]  # [nbin, N_G]
    
    for tap in range(N_G):
        phase_mean = np.mean(np.angle(w_steady[:, tap]))
        phase_std = np.std(np.angle(w_steady[:, tap]))
        ax8.bar(tap, phase_mean, yerr=phase_std, capsize=5, 
               label=f'Tap {tap}', alpha=0.7)
    
    ax8.set_xlabel('Tap Index')
    ax8.set_ylabel('Phase (radians)')
    ax8.set_title(f'Weight Phase at Frame {steady_frame}\n(Mean ± Std Dev across frequency bins)')
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.set_xticks([0, 1, 2, 3])
    
    # -------------------------------------------------------------------------
    # Plot 11: Weight magnitude vs frequency (at steady state)
    # -------------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[3, 1])
    freq_axis = np.arange(nbin) * sr / n_fft
    
    for tap in range(N_G):
        ax9.plot(freq_axis, np.abs(w_steady[:, tap]), label=f'Tap {tap}', linewidth=1)
    
    ax9.set_xlabel('Frequency (Hz)')
    ax9.set_ylabel('Weight Magnitude')
    ax9.set_title(f'Weight Magnitude vs Frequency\n(Frame {steady_frame})')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, sr/2)
    
    # -------------------------------------------------------------------------
    # Plot 12: Convergence rate analysis
    # -------------------------------------------------------------------------
    ax10 = fig.add_subplot(gs[3, 2])
    
    # Compute convergence rate (derivative of output power)
    output_power_arr = np.array(output_power_history)
    convergence_rate = -np.diff(output_power_arr) / output_power_arr[:-1]
    
    ax10.plot(frame_axis[:-1], convergence_rate, linewidth=1, color='darkgreen')
    ax10.axhline(y=0.1, color='orange', linestyle='--', label='10% per frame')
    ax10.axhline(y=0.01, color='gray', linestyle='--', label='1% per frame')
    ax10.set_xlabel('Frame')
    ax10.set_ylabel('Relative Convergence Rate')
    ax10.set_title('Convergence Speed\n(Negative d(Power)/Power)')
    ax10.legend(loc='upper right', fontsize=8)
    ax10.grid(True, alpha=0.3)
    ax10.set_xlim(0, len(convergence_rate))
    ax10.set_ylim(0, 1)
    
    # Add text box with key metrics
    textstr = '\n'.join([
        f'Key Metrics:',
        f'  Initial power: {output_power_history[0]:.6f}',
        f'  Final power: {output_power_history[-1]:.6f}',
        f'  Reduction: {10*np.log10(output_power_history[0]/(output_power_history[-1]+1e-10)):.1f} dB',
        f'  Final ratio: {ratio_history[-1]:.4f}',
        f'  Tap 0 dominance: {tap0_dominance[-1]:.2%}'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
            bbox=props, family='monospace')
    
    # Save figure
    plt.savefig('rls_weight_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: rls_weight_convergence.png")
    plt.close()
    
    # =========================================================================
    # Create animation-style sequence (multiple subplots showing progression)
    # =========================================================================
    print("Generating convergence sequence...")
    
    fig2, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig2.suptitle('Weight Convergence Sequence', fontsize=14, fontweight='bold')
    
    sequence_frames = [0, 10, 50, 100, 300, 1000, 3000, 5000, min(n_frames-1, 7000)]
    
    for idx, (ax, frame) in enumerate(zip(axes.flat, sequence_frames)):
        if frame < n_frames:
            # Plot weight distribution at this frame
            ax.bar(tap_axis, weight_history[frame], color='steelblue', alpha=0.7)
            ax.set_xlabel('Tap')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'Frame {frame}')
            ax.set_ylim(0, max(np.max(weight_history), 0.1))
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add ratio annotation
            ratio_text = f'ratio={ratio_history[frame]:.3f}'
            ax.text(0.5, 0.95, ratio_text, transform=ax.transAxes, 
                   fontsize=8, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('rls_convergence_sequence.png', dpi=150, bbox_inches='tight')
    print("Saved: rls_convergence_sequence.png")
    plt.close()
    
    # =========================================================================
    # Create 3D surface plot of weight evolution
    # =========================================================================
    print("Generating 3D visualization...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig3 = plt.figure(figsize=(12, 8))
    ax3d = fig3.add_subplot(111, projection='3d')
    
    # Create meshgrid
    frame_mesh, tap_mesh = np.meshgrid(np.arange(0, n_frames, 50), tap_axis)
    weight_mesh = weight_history[::50, :].T
    
    # Plot surface
    surf = ax3d.plot_surface(frame_mesh, tap_mesh, weight_mesh, 
                            cmap='viridis', alpha=0.9, 
                            edgecolor='none', linewidth=0)
    
    ax3d.set_xlabel('Frame', labelpad=10)
    ax3d.set_ylabel('Tap Index', labelpad=10)
    ax3d.set_zlabel('Weight Magnitude', labelpad=10)
    ax3d.set_title('Weight Convergence Surface\n(3D View)', pad=20)
    
    # Add colorbar
    fig3.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label='Magnitude')
    
    plt.tight_layout()
    plt.savefig('rls_weight_surface_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: rls_weight_surface_3d.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. rls_weight_convergence.png - Comprehensive multi-panel view")
    print("  2. rls_convergence_sequence.png - Time progression sequence")
    print("  3. rls_weight_surface_3d.png - 3D surface visualization")
    print("=" * 70)


def visualize_known_delay_convergence():
    """
    Visualize weight convergence with a known delay in the echo path
    """
    print("\n" + "=" * 70)
    print("Known Delay Convergence Visualization")
    print("=" * 70)
    
    # Configuration
    sr = 16000
    duration = 3.0
    n_fft = 256
    hop_length = 64
    nbin = n_fft // 2 + 1
    N_G = 8  # More taps to capture delay
    
    # Create test signal with impulses
    t = np.arange(int(sr * duration)) / sr
    ref_signal = np.zeros_like(t)
    
    # Add impulses at known positions
    impulse_positions = [2000, 8000, 15000, 25000, 35000]
    for pos in impulse_positions:
        if pos < len(ref_signal):
            ref_signal[pos] = 0.5
    
    # Add noise
    ref_signal += 0.02 * np.random.randn(len(ref_signal))
    
    # Create echo with KNOWN delay
    known_delay = 256  # samples (16ms)
    known_decay = 0.7
    mic_signal = np.zeros_like(ref_signal)
    mic_signal[known_delay:] = known_decay * ref_signal[:-known_delay]
    
    print(f"\nEcho path: delay={known_delay} samples, decay={known_decay}")
    print(f"Expected tap position: {known_delay // hop_length}")
    
    # STFT
    X_ref = stft_analysis(ref_signal, n_fft, hop_length)
    X_mic = stft_analysis(mic_signal, n_fft, hop_length)
    n_frames = X_ref.shape[1]
    
    # Create RLS filter
    rls = RLSBishengMDF(
        NCHAN=1, NBIN=nbin, N_G=N_G,
        N_cnt_loud=1, N_wait=10,
        thr_loud=1e-6, alpha=0.05, beta=0.97,
        bin_lim=nbin, Nrxref=1
    )
    
    # Track convergence
    weight_history = []
    
    for frame_idx in range(n_frames):
        Y_rx = X_ref[:, frame_idx:frame_idx+1]
        Y = X_mic[:, frame_idx:frame_idx+1]
        E = rls.apply(Y, Y_rx)
        
        w_mag_per_tap = np.mean(np.abs(rls.w[0]), axis=0)[:, 0]
        weight_history.append(w_mag_per_tap.copy())
    
    weight_history = np.array(weight_history)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Weight Convergence with Known Delay ({known_delay} samples)', 
                 fontsize=14, fontweight='bold')
    
    frame_axis = np.arange(n_frames)
    tap_axis = np.arange(N_G)
    
    # Plot 1: Weight magnitude over time (all taps)
    ax1 = axes[0, 0]
    for tap in range(N_G):
        ax1.plot(frame_axis, weight_history[:, tap], label=f'Tap {tap}', linewidth=1.5)
    
    # Mark expected tap
    expected_tap = known_delay // hop_length
    ax1.axhline(y=expected_tap + 0.5, color='red', linestyle='--', linewidth=2,
               label=f'Expected (tap {expected_tap})')
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Tap Index')
    ax1.set_title('Weight Magnitude Convergence')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_frames)
    ax1.set_ylim(-0.5, N_G - 0.5)
    
    # Plot 2: Heatmap
    ax2 = axes[0, 1]
    im = ax2.imshow(weight_history.T, aspect='auto', origin='lower',
                   cmap='YlOrRd', extent=[0, n_frames, 0, N_G])
    ax2.axhline(y=expected_tap, color='red', linestyle='--', linewidth=2,
               label=f'Expected tap {expected_tap}')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Tap Index')
    ax2.set_title('Weight Distribution Heatmap')
    ax2.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax2, label='Magnitude')
    
    # Plot 3: Final weight distribution
    ax3 = axes[1, 0]
    ax3.bar(tap_axis, weight_history[-1], color='steelblue', alpha=0.7)
    ax3.axvline(x=expected_tap, color='red', linestyle='--', linewidth=2,
               label=f'Expected (tap {expected_tap})')
    ax3.set_xlabel('Tap Index')
    ax3.set_ylabel('Weight Magnitude')
    ax3.set_title('Final Weight Distribution')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(tap_axis)
    
    # Plot 4: Peak tap over time
    ax4 = axes[1, 1]
    peak_taps = np.argmax(weight_history, axis=1)
    ax4.plot(frame_axis, peak_taps, linewidth=2, color='darkblue')
    ax4.axhline(y=expected_tap, color='red', linestyle='--', linewidth=2,
               label=f'Expected (tap {expected_tap})')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Peak Tap Index')
    ax4.set_title('Peak Tap Location Over Time')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.5, N_G - 0.5)
    ax4.set_yticks(tap_axis)
    
    plt.tight_layout()
    plt.savefig('rls_delay_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: rls_delay_convergence.png")
    plt.close()
    
    print(f"\nPeak tap at end: {peak_taps[-1]} (expected: {expected_tap})")
    print("=" * 70)


def main():
    """Generate all visualizations"""
    print("=" * 70)
    print("RLS Bisheng MDF - Weight Convergence Visualization Suite")
    print("=" * 70)
    
    # Visualization 1: Basic convergence (no delay)
    visualize_weight_convergence()
    
    # Visualization 2: Known delay convergence
    visualize_known_delay_convergence()
    
    print("\n" + "=" * 70)
    print("All visualizations generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. rls_weight_convergence.png - Comprehensive multi-panel view")
    print("  2. rls_convergence_sequence.png - Time progression sequence")
    print("  3. rls_weight_surface_3d.png - 3D surface visualization")
    print("  4. rls_delay_convergence.png - Known delay convergence")
    print("=" * 70)


if __name__ == "__main__":
    main()

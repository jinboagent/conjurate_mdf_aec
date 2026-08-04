"""
Block-based Echo Cancellation using Time-Domain NLMS and Frequency-Domain RLS Bisheng MDF

This implementation properly handles the echo cancellation scenario where:
- Reference signal (x): Original speech from loudspeaker
- Microphone signal (d): Echo of loudspeaker signal (in single-talk scenario)
- Error signal (e): Residual after echo cancellation (should approach silence)

For a single-talk scenario (no near-end speech):
- Loudspeaker plays reference signal x
- Microphone picks up only the echo (delayed + attenuated version of x)
- Adaptive filter estimates the echo path and cancels it
- Output (error signal) should approach silence

System Parameters:
- Sampling frequency: 16000 Hz
- Echo path length: 250 ms (4000 samples)
- FFT size: 256
- Overlap: 75% (step size = 64 samples)
- N_G (delay blocks): 64 blocks (covers 64 * 64 = 4096 samples = 256 ms)

Supported Algorithms:
- NLMS: Normalized Least Mean Squares (time-domain, simpler)
- RLS: RLS Bisheng MDF (frequency-domain multi-delay filter bank)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import RLS Bisheng MDF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rls_bisheng_mdf import RLSBishengMDF, RLS_PARAMS_BALANCED


class BlockRLSBishengMDF:
    """Block-based frequency-domain RLS Bisheng MDF adaptive filter

    This implementation properly interfaces with the RLS Bisheng MDF algorithm.
    The RLS filter maintains its own internal state for the multi-delay filter bank.
    We process sample-by-sample through the RLS filter in the frequency domain.
    """

    def __init__(self, filter_length, sr, mu=0.03, overlap=0.75, fft_size=512, n_g=64):
        """
        Initialize block RLS Bisheng MDF multi-delay filter

        Parameters:
            filter_length: Filter length in samples (echo path to cover)
            sr: Sample rate (16000 Hz)
            mu: Step size for FDAF (default 0.01 for stability)
            overlap: Overlap ratio (0.75 = 75% overlap, step=64 samples)
            fft_size: FFT size (256)
            n_g: Number of delay blocks in multi-delay filter bank
        """
        self.filter_length = filter_length
        self.sr = sr
        self.overlap = overlap
        self.fft_size = fft_size
        self.n_g = n_g
        self.mu = mu

        # Calculate step size (advance per block)
        self.step_size = int(self.fft_size * (1 - overlap))

        # Number of frequency bins
        self.nbin = self.fft_size // 2 + 1

        # Calculate echo path coverage
        self.echo_coverage = self.n_g * self.step_size  # samples
        self.echo_coverage_ms = self.echo_coverage / sr * 1000

        print(f"  - FFT size: {self.fft_size}")
        print(f"  - Number of delay blocks (N_G): {self.n_g}")
        print(f"  - Step size: {self.step_size} samples")
        print(f"  - Echo path coverage: {self.echo_coverage} samples ({self.echo_coverage_ms:.1f} ms)")

        # Initialize FDAF with the specified mu
        alpha = mu    # Use mu as the step size
        beta = 0.97   # Forgetting factor
        bin_lim = self.nbin

        # N_G = number of delay blocks in the multi-delay filter bank
        self.rls_filter = RLSBishengMDF(
            NCHAN=1,
            NBIN=self.nbin,
            N_G=self.n_g,
            alpha=alpha,
            beta=beta,
            bin_lim=bin_lim,
            Nrxref=1
        )


    def process_block(self, x_block, d_block):
        """
        Process a block of samples using RLS Bisheng MDF multi-delay filter.

        Uses overlap-save: fft_size input samples produce step_size valid output samples.
        The RLS filter maintains its own internal multi-delay buffer — this wrapper
        simply converts between time-domain frames and frequency-domain RLS.

        Parameters:
            x_block: Reference signal block (fft_size samples)
            d_block: Desired signal block (fft_size samples)

        Returns:
            e_block: Error signal block (step_size valid output samples)
        """
        # FFT of reference and microphone signals
        X = np.fft.rfft(x_block)
        D = np.fft.rfft(d_block)

        # Reshape for RLS filter (nbin, 1)
        X = X.reshape(-1, 1)
        D = D.reshape(-1, 1)

        # Apply RLS echo cancellation
        E = self.rls_filter.apply(D, X)

        # Convert error back to time domain
        e_time = np.fft.irfft(E[:, 0], n=self.fft_size)

        # Overlap-save: return last step_size samples (valid linear convolution output)
        return e_time[-self.step_size:]


class BlockNLMS:
    """Sample-by-sample time-domain NLMS adaptive filter
    
    This is a standard NLMS filter that processes samples one at a time.
    The 'block' processing is just for API compatibility - we process
    sample-by-sample internally.
    """

    def __init__(self, filter_length, mu=0.5):
        """
        Initialize NLMS filter

        Parameters:
            filter_length: Filter length in samples (echo path to cover)
            mu: Step size parameter (0 < mu < 2 for NLMS stability, 0.5 is good)
        """
        self.filter_length = filter_length
        self.mu = mu

        # Initialize filter weights in time domain
        self.w = np.zeros(self.filter_length)

        # Input buffer for filter history (last filter_length - 1 samples)
        self.x_history = np.zeros(self.filter_length - 1)

    def process_block(self, x_block, d_block):
        """
        Process a block of samples using time-domain NLMS

        Parameters:
            x_block: Reference signal block (loudspeaker signal)
            d_block: Desired signal block (microphone signal with echo)

        Returns:
            e_block: Error signal block (echo-cancelled output)
        """
        block_len = len(x_block)
        e_block = np.zeros(block_len)

        for n in range(block_len):
            # Build filter input: [current sample, previous filter_length-1 samples]
            x_window = np.concatenate([[x_block[n]], self.x_history])

            # Calculate output (echo estimate)
            y = np.dot(self.w, x_window)

            # Calculate error
            e = d_block[n] - y
            e_block[n] = e

            # NLMS weight update
            x_power = np.dot(x_window, x_window) + 1e-6
            self.w = self.w + self.mu * e * x_window / x_power

            # Update history (shift and add new sample)
            self.x_history = x_window[:-1]

        return e_block


def create_echo(signal, sr, delay_ms=50, decay=0.5):
    """
    Create echo signal

    Parameters:
        signal: Original signal
        sr: Sample rate
        delay_ms: Delay time (milliseconds)
        decay: Decay factor

    Returns:
        echo_signal: Signal with echo
        delay_samples: The actual delay in samples
    """
    # Calculate delay in samples
    delay_samples = int(sr * delay_ms / 1000)

    # Create echo signal
    echo_signal = np.zeros_like(signal)
    echo_signal[delay_samples:] = decay * signal[:-delay_samples]

    # Superpose original signal and echo
    return signal + echo_signal, delay_samples


def test_echo_cancellation(algorithm='lms'):
    """Test echo cancellation with different algorithms

    Parameters:
        algorithm: Algorithm to use ('lms' or 'rls')
    """
    print(f"\n{'='*60}")
    print(f"Testing Echo Cancellation with {algorithm.upper()} algorithm")
    print(f"{'='*60}\n")

    # System parameters
    TARGET_SR = 16000  # Target sample rate: 16 kHz
    ECHO_PATH_MS = 250  # Echo path length: 250 ms
    FFT_SIZE = 256  # FFT size
    OVERLAP = 0.75  # 75% overlap
    STEP_SIZE = int(FFT_SIZE * (1 - OVERLAP))  # 64 samples
    N_G = 64  # Number of delay blocks (covers 64 * 64 = 4096 samples = 256 ms)

    # Load pre-generated test files (reference and microphone with echo)
    # Files created by create_test_files.py from LibriSpeech sample
    ref_file = os.path.join(os.path.dirname(__file__), 'reference.wav')
    mic_file = os.path.join(os.path.dirname(__file__), 'microphone.wav')
    
    if os.path.exists(ref_file) and os.path.exists(mic_file):
        print(f"Loading test files...")
        ref_signal, sr = sf.read(ref_file)
        mic_signal, _ = sf.read(mic_file)
        print(f"  - {ref_file}")
        print(f"  - {mic_file}")
    else:
        # Fallback: generate from librosa sample
        print(f"Test files not found, generating from LibriSpeech sample...")
        y_orig, sr_orig = librosa.load(librosa.ex('libri2'), duration=10, sr=None)
        ref_signal = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
        sr = TARGET_SR
        
        # Create echo signal
        delay_samples = int(sr * 50 / 1000)  # 50ms delay
        decay = 0.4  # -8dB
        mic_signal = np.zeros_like(ref_signal)
        mic_signal[delay_samples:] = 1 * ref_signal[:-delay_samples]

    print(f"Sample rate: {sr} Hz")
    print(f"Signal length: {len(ref_signal)} samples ({len(ref_signal)/sr:.2f} s)")
    
    # Calculate actual echo parameters from loaded signals
    # Cross-correlation to find actual delay
    corr = np.correlate(ref_signal, mic_signal, mode='full')
    peak_idx = np.argmax(np.abs(corr))
    actual_delay = peak_idx - (len(ref_signal) - 1)
    if actual_delay > 0 and np.abs(mic_signal[actual_delay]) > 0.01:
        estimated_decay = mic_signal[actual_delay] / ref_signal[0] if ref_signal[0] != 0 else 0.4
    else:
        estimated_decay = 0.4
    
    print(f"Echo delay: {actual_delay} samples ({actual_delay / sr * 1000:.1f} ms)")
    print(f"Echo attenuation: {estimated_decay:.2f} (-{20*np.log10(max(abs(1/estimated_decay), 0.01)):.1f} dB)")
    print(f"\nSystem Configuration:")
    print(f"  - FFT size: {FFT_SIZE}")
    print(f"  - Overlap: {OVERLAP*100:.0f}%")
    print(f"  - Step size: {int(FFT_SIZE * (1-OVERLAP))} samples")
    print(f"  - N_G (delay blocks): {N_G}")
    print(f"  - Echo path coverage: {N_G * int(FFT_SIZE * (1-OVERLAP))} samples ({N_G * int(FFT_SIZE * (1-OVERLAP))/sr*1000:.1f} ms)")

    # Create adaptive filter based on selected algorithm
    filter_length = int(ECHO_PATH_MS * sr / 1000)  # 250 ms in samples

    if algorithm.lower() == 'lms':
        mu = 0.5  # Good step size for NLMS
        adaptive_filter = BlockNLMS(filter_length, mu)
        print(f"\nAlgorithm: Block NLMS (Time-domain)")
        print(f"  - Step size (mu): {mu}")
        print(f"  - Filter length: {filter_length}")
        print(f"  - Filter covers: {filter_length/sr*1000:.1f} ms")

    elif algorithm.lower() == 'rls':
        # For RLS Bisheng MDF (FDAF), use N_G=32 delay blocks
        # The FDAF is a frequency-domain filter that works on FFT frames
        # Use higher mu since it gets divided by N_G internally
        adaptive_filter = BlockRLSBishengMDF(
            filter_length, sr, overlap=0.75, fft_size=512, n_g=N_G, mu=0.03
        )
        print(f"\nAlgorithm: Block FDAF (Frequency-domain)")
        print(f"  - FFT size: {adaptive_filter.fft_size}")
        print(f"  - Frequency bins: {adaptive_filter.nbin}")
        print(f"  - Block size: {adaptive_filter.fft_size}")
        print(f"  - Step size (advance): {adaptive_filter.step_size}")
        print(f"  - Number of delay blocks (N_G): {adaptive_filter.n_g}")
        print(f"  - Echo path coverage: {adaptive_filter.echo_coverage} samples ({adaptive_filter.echo_coverage_ms:.1f} ms)")
        print(f"  - Using FDAF with mu=0.03 (conservative)")

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'lms' or 'rls'")

    # Process signal
    n_samples = len(ref_signal)
    processed_signal = np.zeros(n_samples)

    # Process signal based on algorithm type
    if algorithm.lower() == 'lms':
        # NLMS: Process sample-by-sample (using block API for efficiency)
        print(f"\nProcessing {n_samples} samples...\n")
        
        # Process in chunks for efficiency
        chunk_size = 256
        for start_idx in tqdm(range(0, n_samples, chunk_size), desc="Processing (NLMS)"):
            end_idx = min(start_idx + chunk_size, n_samples)
            x_block = ref_signal[start_idx:end_idx]
            d_block = mic_signal[start_idx:end_idx]
            e_block = adaptive_filter.process_block(x_block, d_block)
            processed_signal[start_idx:end_idx] = e_block

    elif algorithm.lower() == 'rls':
        # FDAF with overlap-save: each frame takes fft_size input samples
        # (with step_size advance between frames) and produces step_size
        # valid output samples from the tail of the IFFT.
        fft_size = adaptive_filter.fft_size
        step_size = adaptive_filter.step_size

        # First valid output sample index (overlap-save discards leading
        # fft_size - step_size samples contaminated by circular convolution)
        output_start = fft_size - step_size
        n_frames = (n_samples - output_start) // step_size
        print(f"\nProcessing {n_frames} frames (step={step_size}, FFT={fft_size})...\n")

        for i in tqdm(range(n_frames), desc="Processing (FDAF)"):
            # Input window for this frame
            in_start = i * step_size
            in_end = in_start + fft_size

            x_block = ref_signal[in_start:in_end]
            d_block = mic_signal[in_start:in_end]
            e_block = adaptive_filter.process_block(x_block, d_block)

            # Overlap-save: valid output maps to the tail of the input window
            out_start = output_start + i * step_size
            processed_signal[out_start:out_start + step_size] = e_block

            if i < 3:
                print(f"  Frame {i}: input=[{in_start}:{in_end}], output=[{out_start}:{out_start+step_size}]")

    # Print some debug info about filter convergence
    print(f"\nFilter weight statistics:")
    if algorithm.lower() == 'lms':
        # NLMS has simple time-domain weights
        print(f"  Max weight: {np.max(np.abs(adaptive_filter.w)):.6f}")
        if delay_samples < len(adaptive_filter.w):
            print(f"  Weight at echo delay: {adaptive_filter.w[delay_samples]:.6f} (should be ≈ {decay})")
        print(f"  Sum of weights: {np.sum(adaptive_filter.w):.6f}")
    elif algorithm.lower() == 'rls':
        # FDAF has frequency-domain weights
        w_fdaf = adaptive_filter.rls_filter.w[0]  # Shape: [nbin, nchan]
        w_max = np.max(np.abs(w_fdaf))
        w_mean = np.mean(np.abs(w_fdaf))
        print(f"  Max frequency weight magnitude: {w_max:.6f}")
        print(f"  Mean frequency weight magnitude: {w_mean:.6f}")
        print(f"  Weight shape: {w_fdaf.shape} (freq_bins x channels)")
    
    # Skip the transient region (first 2*filter_length samples) for evaluation
    # Need extra time for filter to converge
    transient_length = 4 * filter_length  # More time for multi-delay filter
    mic_eval = mic_signal[transient_length:]
    processed_eval = processed_signal[transient_length:]

    # Calculate echo cancellation performance metrics
    # In single-talk scenario:
    # - mic_signal = echo only (what we start with)
    # - After cancellation: processed_signal should ≈ 0 (silence, echo removed)
    # - The error signal is our estimate of the near-end speech (which is 0 in single-talk)

    # Echo component power (mic_signal IS the echo in single-talk)
    echo_power = np.sum(mic_eval**2)

    # Residual error after processing (should approach zero)
    residual_power = np.sum(processed_eval**2)

    # ERLE (Echo Return Loss Enhancement)
    # Measures how much the echo was reduced
    mic_power = np.sum(mic_eval**2)
    processed_power = np.sum(processed_eval**2)

    # ERLE = 10 * log10(mic_power / residual_power)
    # Higher is better - means more echo was cancelled
    echo_cancellation_db = 10 * np.log10((mic_power + 1e-10) / (residual_power + 1e-10))

    print(f"\n{'='*60}")
    print(f"Echo Cancellation Performance ({algorithm.upper()})")
    print(f"{'='*60}")
    print(f"Microphone signal power:   {mic_power:.4f} (echo only)")
    print(f"Processed signal power:    {processed_power:.4f} (should approach 0)")
    print(f"Residual error power:      {residual_power:.4f}")
    print(f"Echo cancellation (ERLE):  {echo_cancellation_db:.2f} dB")
    print(f"{'='*60}\n")

    # Check if processed signal is actually close to silence
    processed_rms = np.sqrt(np.mean(processed_eval**2))
    mic_rms = np.sqrt(np.mean(mic_eval**2))
    print(f"RMS before cancellation:   {mic_rms:.6f}")
    print(f"RMS after cancellation:    {processed_rms:.6f}")
    print(f"Echo reduction achieved:   {echo_cancellation_db:.2f} dB")

    if echo_cancellation_db > 15:
        print(f"✓✓ EXCELLENT: Echo was reduced by {echo_cancellation_db:.2f} dB (>15 dB target)")
    elif echo_cancellation_db > 10:
        print(f"✓ GOOD: Echo was reduced by {echo_cancellation_db:.2f} dB (>10 dB)")
    elif echo_cancellation_db > 0:
        print(f"✓ PARTIAL: Echo was reduced by {echo_cancellation_db:.2f} dB")
    else:
        print(f"✗ FAILED: Echo cancellation made it worse")
    print()

    # =========================================================================
    # ALGORITHM VERIFICATION: Compare estimated echo with true echo
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Algorithm Verification ({algorithm.upper()})")
    print(f"{'='*60}")

    # True echo component (in single-talk, mic_signal IS the echo)
    true_echo = mic_signal[transient_length:]
    estimated_echo = (mic_signal - processed_signal)[transient_length:]

    # Correlation between true echo and estimated echo
    echo_correlation = np.corrcoef(true_echo, estimated_echo)[0, 1]
    print(f"Echo estimation correlation: {echo_correlation:.4f}")

    # Echo estimation accuracy
    echo_est_error = true_echo - estimated_echo
    echo_est_accuracy = 10 * np.log10(np.sum(true_echo**2) / (np.sum(echo_est_error**2) + 1e-10))
    print(f"Echo estimation accuracy:    {echo_est_accuracy:.2f} dB")

    # Check if filter learned the echo path
    if algorithm.lower() == 'lms':
        # For NLMS, check the filter weights directly
        expected_delay = delay_samples
        expected_decay = 0.4

        # Find the peak in the filter weights
        peak_idx = np.argmax(np.abs(adaptive_filter.w))
        peak_val = adaptive_filter.w[peak_idx]

        print(f"\nFilter weight analysis:")
        print(f"  Expected echo delay: {expected_delay} samples")
        print(f"  Filter peak location: {peak_idx} samples")
        print(f"  Expected decay: {expected_decay}")
        print(f"  Filter peak value: {np.abs(peak_val):.4f}")
        print(f"  Delay error: {abs(peak_idx - expected_delay)} samples ({abs(peak_idx - expected_delay)/sr*1000:.2f} ms)")

        # Check if filter approximately learned the echo path
        delay_accuracy = 1 - min(abs(peak_idx - expected_delay) / filter_length, 1.0)
        decay_accuracy = 1 - min(abs(np.abs(peak_val) - expected_decay) / expected_decay, 1.0)

        print(f"\n  Delay accuracy: {delay_accuracy*100:.2f}%")
        print(f"  Decay accuracy: {decay_accuracy*100:.2f}%")

        if delay_accuracy > 0.9 and decay_accuracy > 0.7:
            print(f"\n✓✓ VERIFIED: Filter correctly learned the echo path!")
        elif delay_accuracy > 0.8:
            print(f"\n✓ PARTIAL: Filter approximately learned the echo path")
        else:
            print(f"\n✗ NOT VERIFIED: Filter did not learn the echo path correctly")

    elif algorithm.lower() == 'rls':
        # For FDAF, analyze frequency domain weights
        w_fdaf = adaptive_filter.rls_filter.w[0]  # Shape: [nbin, nchan]
        w_mag = np.mean(np.abs(w_fdaf), axis=1)  # Average over channels

        print(f"\nFilter weight analysis (frequency domain):")
        print(f"  Number of frequency bins: {len(w_mag)}")
        print(f"  Max weight magnitude: {np.max(w_mag):.4f}")
        print(f"  Mean weight magnitude: {np.mean(w_mag):.4f}")

        # For verification, check if echo estimation is good
        if echo_correlation > 0.9 and echo_est_accuracy > 20:
            print(f"\n✓✓ VERIFIED: FDAF successfully estimated the echo!")
        elif echo_correlation > 0.7 and echo_est_accuracy > 10:
            print(f"\n✓ GOOD: FDAF shows strong echo estimation capability")
        elif echo_correlation > 0.5:
            print(f"\n✓ PARTIAL: FDAF shows some echo estimation capability")
        else:
            print(f"\n✗ NOT VERIFIED: FDAF did not estimate echo correctly")

    print(f"{'='*60}\n")

    # Save audio files
    sf.write('original_speech.wav', ref_signal, sr)
    sf.write('echo_signal.wav', mic_signal, sr)
    sf.write(f'processed_signal_{algorithm}.wav', processed_signal, sr)

    # Calculate echo estimate (echo that was removed)
    echo_estimate = mic_signal - processed_signal

    # Visualization
    plt.figure(figsize=(12, 12))

    # Reference signal (loudspeaker) waveform
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(ref_signal, sr=sr)
    plt.title('Reference signal (loudspeaker)')

    # Microphone signal (echo only in single-talk)
    plt.subplot(4, 1, 2)
    librosa.display.waveshow(mic_signal, sr=sr)
    plt.title('Microphone signal (echo only - single-talk)')

    # Processed signal (should be silence)
    plt.subplot(4, 1, 3)
    librosa.display.waveshow(processed_signal, sr=sr)
    plt.title(f'Processed signal after echo cancellation ({algorithm.upper()})')

    # Estimated echo waveform
    plt.subplot(4, 1, 4)
    librosa.display.waveshow(echo_estimate, sr=sr)
    plt.title(f'Estimated echo removed by {algorithm.upper()}')

    plt.tight_layout()
    plt.savefig(f'echo_cancellation_waveforms_{algorithm}.png')
    plt.close()

    # Spectral analysis
    plt.figure(figsize=(12, 12))

    # Reference signal spectrogram
    plt.subplot(4, 1, 1)
    D = librosa.stft(ref_signal)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.f dB')
    plt.title('Reference signal spectrogram (loudspeaker)')

    # Microphone signal spectrogram
    plt.subplot(4, 1, 2)
    D_echo = librosa.stft(mic_signal)
    S_db_echo = librosa.amplitude_to_db(np.abs(D_echo), ref=np.max)
    librosa.display.specshow(S_db_echo, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.f dB')
    plt.title('Microphone signal spectrogram (echo only)')

    # Processed signal spectrogram
    plt.subplot(4, 1, 3)
    D_processed = librosa.stft(processed_signal)
    S_db_processed = librosa.amplitude_to_db(np.abs(D_processed), ref=np.max)
    librosa.display.specshow(S_db_processed, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.f dB')
    plt.title(f'Processed signal spectrogram after echo cancellation ({algorithm.upper()})')

    # Estimated echo spectrogram
    plt.subplot(4, 1, 4)
    D_echo_est = librosa.stft(echo_estimate)
    S_db_echo_est = librosa.amplitude_to_db(np.abs(D_echo_est), ref=np.max)
    librosa.display.specshow(S_db_echo_est, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.f dB')
    plt.title(f'Estimated echo spectrogram ({algorithm.upper()})')

    plt.tight_layout()
    plt.savefig(f'echo_cancellation_spectrograms_{algorithm}.png')
    plt.close()
    
    print(f"Saved output files:")
    print(f"  - processed_signal_{algorithm}.wav")
    print(f"  - echo_cancellation_waveforms_{algorithm}.png")
    print(f"  - echo_cancellation_spectrograms_{algorithm}.png")


if __name__ == "__main__":
    import sys

    # Allow command-line argument for algorithm selection
    if len(sys.argv) > 1:
        algorithm = sys.argv[1].lower()
    else:
        # Default: test RLS (more advanced algorithm)
        algorithm = 'rls'

    # Test specific algorithm
    test_echo_cancellation(algorithm)

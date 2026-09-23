"""
Create test WAV files for echo cancellation testing.

Uses the built-in LibriSpeech sample from librosa to create:
- reference.wav: Original clean speech (loudspeaker signal)
- microphone.wav: Echo signal (microphone pickup with echo)

Both files are:
- 16kHz sample rate
- 10 seconds duration
- Mono channel
"""

import os

import numpy as np
import librosa
import soundfile as sf


def create_test_files(output_dir='audio', duration=10, delay_ms=50, decay=0.4,
                      speech_file=None):
    """
    Create reference and microphone (echo) test files.

    Args:
        output_dir: output directory (default 'audio'); created if missing
        duration: duration in seconds
        delay_ms: echo delay in milliseconds (default 50)
        decay: echo attenuation factor (default 0.4 = -8 dB)
        speech_file: optional WAV to use as the far-end speech (default
            None -> LibriSpeech sample via librosa)
    """
    TARGET_SR = 16000  # 16kHz

    os.makedirs(output_dir, exist_ok=True)

    if speech_file is not None:
        print(f"Loading speech file: {speech_file}")
        y_orig, sr_orig = sf.read(speech_file, dtype='float32')
        if sr_orig != TARGET_SR:
            print(f"Resampling from {sr_orig} Hz to {TARGET_SR} Hz...")
            y = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
        else:
            y = y_orig
        y = y[:int(duration * TARGET_SR)]
    else:
        print(f"Loading LibriSpeech sample from librosa...")
        y_orig, sr_orig = librosa.load(librosa.ex('libri2'), duration=duration, sr=None)
        if sr_orig != TARGET_SR:
            print(f"Resampling from {sr_orig} Hz to {TARGET_SR} Hz...")
            y = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=TARGET_SR)
        else:
            y = y_orig
    
    sr = TARGET_SR
    
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y) / sr:.2f} seconds")
    print(f"Samples: {len(y)}")
    
    # Create echo signal (simulate microphone picking up loudspeaker echo)
    delay_samples = int(sr * delay_ms / 1000)
    mic_signal = np.zeros_like(y)
    
    if delay_samples == 0:
        # Direct path only (for debugging)
        mic_signal = decay * y
    else:
        # Echo: delayed and attenuated version
        if delay_samples < len(y):
            mic_signal[delay_samples:] = decay * y[:-delay_samples]
    
    # Reference signal (what goes to loudspeaker)
    reference = y.copy()
    
    # Normalize to prevent clipping
    ref_max = np.max(np.abs(reference))
    mic_max = np.max(np.abs(mic_signal))
    
    if ref_max > 0.95:
        reference = reference * 0.95 / ref_max
    if mic_max > 0.95:
        mic_signal = mic_signal * 0.95 / mic_max
    
    # Save files
    ref_file = f'{output_dir}/reference.wav'
    mic_file = f'{output_dir}/microphone.wav'
    
    print(f"\nSaving files...")
    sf.write(ref_file, reference.astype(np.float32), sr)
    print(f"  - {ref_file} ({len(reference) / sr:.2f}s, {sr}Hz)")
    
    sf.write(mic_file, mic_signal.astype(np.float32), sr)
    print(f"  - {mic_file} ({len(mic_signal) / sr:.2f}s, {sr}Hz)")
    
    # Print file info
    ref_size = os.path.getsize(ref_file) / 1024  # KB
    mic_size = os.path.getsize(mic_file) / 1024  # KB
    
    print(f"\nFile sizes:")
    print(f"  - reference.wav: {ref_size:.1f} KB")
    print(f"  - microphone.wav: {mic_size:.1f} KB")
    print(f"  - Total: {(ref_size + mic_size) / 1024:.2f} MB")
    
    print(f"\nEcho parameters:")
    print(f"  - Delay: {delay_samples} samples ({delay_ms} ms)")
    print(f"  - Decay: {decay} (-{20 * np.log10(1/decay):.1f} dB)")
    
    # Calculate echo power ratio
    ref_power = np.sum(reference ** 2)
    mic_power = np.sum(mic_signal ** 2)
    echo_ratio = 10 * np.log10(mic_power / ref_power) if ref_power > 0 else 0
    
    print(f"  - Echo power ratio: {echo_ratio:.2f} dB")
    
    print(f"\n✓ Test files created successfully!")
    
    return ref_file, mic_file


if __name__ == '__main__':
    import sys

    # Defaults: the canonical test pair (50 ms delay, -8 dB echo).
    # NOTE: the accidental identity pair (delay=0, decay=1) previously
    # shipped as reference/microphone was created by running this script
    # bare with the old defaults; the defaults below are the intended ones.
    delay_ms = 50
    decay = 0.4
    speech_file = None
    duration = 10

    if len(sys.argv) > 1:
        delay_ms = float(sys.argv[1])
    if len(sys.argv) > 2:
        decay = float(sys.argv[2])
    if len(sys.argv) > 3:
        speech_file = sys.argv[3]
    if len(sys.argv) > 4:
        duration = float(sys.argv[4])

    print(f"Creating test files with delay={delay_ms}ms, decay={decay}, "
          f"duration={duration}s")
    create_test_files(duration=duration, delay_ms=delay_ms, decay=decay,
                      speech_file=speech_file)

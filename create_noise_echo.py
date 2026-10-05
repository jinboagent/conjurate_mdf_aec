"""
Create a WHITE-NOISE reference + microphone pair where the echo path is a
LOW-PASS FILTER with 50 ms delay and gain 0.1.

    reference  = white noise (generate_test_signals' built-in noise source)
    echo path  = 2nd-order Butterworth lowpass (project convention:
                 echo_path_generator.py uses signal.butter(2, 0.3, 'low'))
                 placed at the 50 ms delay tap, scaled to gain 0.1
    microphone = conv(reference, echo path)

Composed from existing project code only:
    - ground_truth.generators.generate_test_signals (noise ref + convolve)
    - ground_truth.generators.generate_simple_echo_path (delay-tap pattern)
    - the scipy butter lowpass convention from echo_path_generator.py
    - ground_truth.loaders.save_wav, harness.metrics.spectrum plots

Outputs:
    audio/noise_reference.wav / _microphone.wav / _echo_path.wav
    results/<timestamp>_noise-echo/  spectrograms + metrics.json

Run: .venv/Scripts/python create_noise_echo.py [delay_ms] [gain] [cutoff_frac] [speech_file] [duration_s] [prefix]
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'harness_template'))

from ground_truth.generators import (
    generate_simple_echo_path,
    generate_test_signals,
)
from ground_truth.loaders import save_wav
from harness.metrics.spectrum import plot_spectrograms

SR = 16000
DURATION_S = 5                                    # canonical length (5 s)
SR = 16000


def main():
    delay_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    gain = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    cutoff_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    speech_file = sys.argv[4] if len(sys.argv) > 4 else None
    duration_s = float(sys.argv[5]) if len(sys.argv) > 5 else DURATION_S
    prefix = sys.argv[6] if len(sys.argv) > 6 else 'noise'

    N = int(duration_s * SR)
    delay = int(delay_ms * SR / 1000)

    # --- echo path: delay tap (generate_simple_echo_path pattern) passed
    # through the project's Butterworth lowpass, scaled so the passband
    # (DC) gain equals `gain`
    tap = generate_simple_echo_path(delay=delay, decay=1.0, length=N)
    b, a = signal.butter(2, cutoff_frac, btype='low')   # project convention
    h = signal.lfilter(b, a, tap)
    h *= gain / np.sum(h)                               # DC gain -> gain

    # --- signals: white-noise reference (speech_file=None) or speech
    ref, mic = generate_test_signals(h, signal_length=N, sample_rate=SR,
                                     seed=42, speech_file=speech_file)

    save_wav(os.path.join(HERE, 'audio', f'{prefix}_reference.wav'),
             ref.astype(np.float32), SR)
    save_wav(os.path.join(HERE, 'audio', f'{prefix}_microphone.wav'),
             mic.astype(np.float32), SR)
    save_wav(os.path.join(HERE, 'audio', f'{prefix}_echo_path.wav'),
             h.astype(np.float32), SR)

    # --- verification
    H8 = np.abs(np.fft.rfft(h, n=len(h) * 8))
    freqs = np.fft.rfftfreq(len(h) * 8, 1.0 / SR)
    cutoff_hz = cutoff_frac * SR / 2
    peak = int(np.argmax(np.abs(h)))

    ratio_bb = 10 * np.log10(np.sum(mic ** 2) / np.sum(ref ** 2))
    # passband power ratio: lowpass both signals, compare in-band energy
    bp_ref = signal.lfilter(b, a, ref)
    bp_mic = signal.lfilter(b, a, mic)
    ratio_pb = 10 * np.log10(np.sum(bp_mic ** 2) / np.sum(bp_ref ** 2))
    corr = np.correlate(mic[:len(ref)], ref, 'full')
    lag = int(np.argmax(np.abs(corr)) - (len(ref) - 1))

    print(f"echo path    : butter(2, {cutoff_frac}) lowpass (cutoff {cutoff_hz:.0f} Hz), "
          f"tap at {delay} ({delay_ms:.0f} ms), gain {gain}")
    print(f"IR peak      : sample {peak} ({peak/SR*1000:.2f} ms), amplitude {np.max(np.abs(h)):.4f}")
    print(f"passband gain: DC = {np.sum(h):.4f}, max|H(f)| = {H8.max():.4f} "
          f"(target {gain})")
    print(f"ref  rms     : {np.sqrt(np.mean(ref**2)):.4f}")
    print(f"mic  rms     : {np.sqrt(np.mean(mic**2)):.4f}")
    print(f"broadband mic/ref power : {ratio_bb:+.2f} dB (lowpass removes energy)")
    print(f"passband mic/ref power  : {ratio_pb:+.2f} dB (expect {20*np.log10(gain):.1f})")
    print(f"cross-corr lag          : {lag} samples ({lag/SR*1000:.2f} ms)")

    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(HERE, 'results', f'{ts}_noise-echo')
    os.makedirs(out_dir, exist_ok=True)
    plot_spectrograms(
        {'reference (white noise)': ref,
         'microphone (lowpassed, delayed echo)': mic},
        SR, os.path.join(out_dir, 'spectrograms.png'))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'delay_ms': delay_ms, 'delay_samples': delay,
                   'gain': gain, 'cutoff_frac': cutoff_frac,
                   'cutoff_hz': cutoff_hz,
                   'dc_gain': float(np.sum(h)), 'max_abs_H': float(H8.max()),
                   'broadband_ratio_db': float(ratio_bb),
                   'passband_ratio_db': float(ratio_pb),
                   'cross_corr_lag_samples': lag}, fh, indent=2)
    print(f"wrote audio/{prefix}_reference.wav, audio/{prefix}_microphone.wav, "
          f"audio/{prefix}_echo_path.wav")
    print(f"artifacts in {out_dir}")


if __name__ == '__main__':
    main()

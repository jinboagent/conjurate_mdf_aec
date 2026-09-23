"""
Create a microphone signal with a SIMPLE echo path (single delayed tap,
attenuation < 1), from an existing speech file in audio/.

Composes existing project code only:
    - harness_template/ground_truth/generators.py::generate_simple_echo_path
      (h = single tap: delay + decay, gain < 1)
    - harness_template/ground_truth/generators.py::generate_test_signals
      (speech loading + convolution)
    - harness_template/ground_truth/loaders.py::save_wav
    - harness/metrics/spectrum.py::plot_spectrograms (traditional view)

Defaults follow create_test_files.py conventions: 50 ms delay, decay 0.4
(-8 dB). Sanity printed: the tap gain is < 1, so the microphone power is
BELOW the reference power (expected ratio 10*log10(decay^2) = -8.0 dB).

Outputs:
    audio/simple_echo_reference.wav   clean speech
    audio/simple_echo_microphone.wav  echo-contaminated microphone signal
    audio/simple_echo_path.wav        the true echo path (impulse)
    results/<timestamp>_simple-echo/  spectrograms + metrics.json + log

Run: .venv/Scripts/python create_simple_echo.py [speech.wav] [delay_ms] [decay]
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

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


def main():
    speech = sys.argv[1] if len(sys.argv) > 1 else 'original_speech.wav'
    delay_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    decay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4

    speech_path = speech if os.path.isabs(speech) else \
        os.path.join(HERE, 'audio', speech)

    delay = int(delay_ms * SR / 1000)

    # --- existing generators: simple path + speech convolution
    h = generate_simple_echo_path(delay=delay, decay=decay, length=8000)
    ref, mic = generate_test_signals(
        h, signal_length=160000, sample_rate=SR, speech_file=speech_path)

    # create_test_files.py normalization convention (both scaled alike)
    gain = 0.95 / max(np.max(np.abs(ref)), 1e-9)
    ref, mic = ref * gain, mic * gain

    save_wav(os.path.join(HERE, 'audio', 'simple_echo_reference.wav'),
             ref.astype(np.float32), SR)
    save_wav(os.path.join(HERE, 'audio', 'simple_echo_microphone.wav'),
             mic.astype(np.float32), SR)
    save_wav(os.path.join(HERE, 'audio', 'simple_echo_path.wav'),
             h.astype(np.float32), SR)

    # --- sanity: echo path gain must be < 1 -> mic quieter than speech
    ratio = 10 * np.log10(np.sum(mic ** 2) / np.sum(ref ** 2))
    print(f"speech file      : {speech}")
    print(f"echo path        : single tap, delay {delay} samples "
          f"({delay_ms:.0f} ms), gain {decay:.2f} "
          f"({20*np.log10(decay):.1f} dB)")
    print(f"path peak / energy: {np.max(np.abs(h)):.3f} / {np.sum(h**2):.3f}")
    print(f"ref  rms / peak  : {np.sqrt(np.mean(ref**2)):.4f} / "
          f"{np.max(np.abs(ref)):.3f}")
    print(f"mic  rms / peak  : {np.sqrt(np.mean(mic**2)):.4f} / "
          f"{np.max(np.abs(mic)):.3f}")
    print(f"mic/ref power    : {ratio:.2f} dB "
          f"(expected {10*np.log10(decay**2):.2f} dB) "
          f"{'OK - mic below speech' if ratio < 0 else 'GAIN > 1?!'}")

    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(HERE, 'results', f'{ts}_simple-echo')
    os.makedirs(out_dir, exist_ok=True)
    plot_spectrograms(
        {'reference speech (far end)': ref,
         'microphone signal (echo only)': mic},
        SR, os.path.join(out_dir, 'spectrograms.png'))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'speech_file': speech, 'delay_ms': delay_ms,
                   'decay': decay, 'delay_samples': delay,
                   'path_energy': float(np.sum(h ** 2)),
                   'mic_ref_power_ratio_db': ratio,
                   'ref_rms': float(np.sqrt(np.mean(ref ** 2))),
                   'mic_rms': float(np.sqrt(np.mean(mic ** 2)))},
                  fh, indent=2)
    print(f"wrote audio/simple_echo_reference.wav, "
          f"audio/simple_echo_microphone.wav, audio/simple_echo_path.wav")
    print(f"artifacts in {out_dir}")


if __name__ == '__main__':
    main()

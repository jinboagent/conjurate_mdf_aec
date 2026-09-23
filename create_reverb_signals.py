"""
Generate reverberant reference + microphone signal pairs using the
project's documented room scenarios and existing code only.

Room scenarios are taken verbatim from echo_path_examples.py:
    office            : 500 ms filter, 45 ms delay, RT60 300 ms (seed 42)
    conference (large): 800 ms filter, 60 ms delay, RT60 500 ms (seed 123)
    huddle (small)    : 300 ms filter, 20 ms delay, RT60 150 ms (seed 456)

Composed through the harness:
    ground_truth.generators.generate_room_echo_signals
      (echo_path_generator.RIR + generate_test_signals convolution,
       RIR normalized to unit energy so echo gain <= 1)
    ground_truth.loaders.save_wav for the audio

Speech source: audio/original_speech.wav, 5 s (canonical length).

Outputs:
    audio/reverb_<room>_reference.wav / _microphone.wav (+ _rir.wav)
    results/<timestamp>_reverb-echo/  metrics.json

Run: .venv/Scripts/python create_reverb_signals.py
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'harness_template'))

from echo_path_generator import RoomParameters
from ground_truth.generators import generate_room_echo_signals
from ground_truth.loaders import save_wav

SR = 16000
DURATION_S = 5
N_SAMPLES = DURATION_S * SR
SPEECH = os.path.join(HERE, 'audio', 'original_speech.wav')

# the documented reverberant scenarios (echo_path_examples.py)
SCENARIOS = {
    'office': (dict(sampling_rate=SR, filter_length_ms=500,
                    initial_delay_ms=45.0, reverberation_time_ms=300), 42),
    'conference': (dict(sampling_rate=SR, filter_length_ms=800,
                        initial_delay_ms=60.0, reverberation_time_ms=500), 123),
    'huddle': (dict(sampling_rate=SR, filter_length_ms=300,
                    initial_delay_ms=20.0, reverberation_time_ms=150), 456),
}


def main():
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(HERE, 'results', f'{ts}_reverb-echo')
    os.makedirs(out_dir, exist_ok=True)

    report = {}
    for name, (kwargs, seed) in SCENARIOS.items():
        params = RoomParameters(**kwargs)
        ref, mic, h = generate_room_echo_signals(
            params=params, seed=seed, signal_length=N_SAMPLES,
            sample_rate=SR, speech_file=SPEECH, normalize_rir=True)
        gain = 0.95 / max(np.max(np.abs(ref)), 1e-9)   # create_test_files convention
        ref, mic = ref * gain, mic * gain

        save_wav(os.path.join(HERE, 'audio', f'reverb_{name}_reference.wav'),
                 ref.astype(np.float32), SR)
        save_wav(os.path.join(HERE, 'audio', f'reverb_{name}_microphone.wav'),
                 mic.astype(np.float32), SR)
        save_wav(os.path.join(HERE, 'audio', f'reverb_{name}_rir.wav'),
                 h.astype(np.float32), SR)

        ratio = 10 * np.log10(np.sum(mic ** 2) / np.sum(ref ** 2))
        peak = int(np.argmax(np.abs(h)))
        row = {
            'filter_ms': params.filter_length_ms,
            'direct_delay_samples': int(params.initial_delay_samples),
            'rir_energy': float(np.sum(h ** 2)),
            'rir_peak_sample': peak,
            'mic_ref_power_ratio_db': float(ratio),
            'ref_rms': float(np.sqrt(np.mean(ref ** 2))),
            'mic_rms': float(np.sqrt(np.mean(mic ** 2))),
        }
        report[name] = row
        print(f"{name:<11} filter {params.filter_length_ms} ms, "
              f"delay {params.initial_delay_ms:.0f} ms, "
              f"RT60 {params.reverberation_time_ms} ms | "
              f"RIR energy {row['rir_energy']:.3f} | "
              f"mic/ref {ratio:+.2f} dB "
              f"({'OK' if ratio < 0 else 'GAIN > 1!'})")

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'speech': SPEECH, 'duration_s': DURATION_S,
                   'normalize_rir': True, 'scenarios': report}, fh, indent=2)
    print(f"\nwrote audio/reverb_<room>_reference/_microphone/_rir.wav "
          f"(office, conference, huddle)")
    print(f"artifacts in {out_dir}")


if __name__ == '__main__':
    main()

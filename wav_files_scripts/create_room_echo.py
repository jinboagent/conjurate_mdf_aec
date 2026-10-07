"""
Create a real echo with the project's own room simulation, via the harness.

All functionality comes from existing project code:
    - harness_template/ground_truth/generators.py::generate_room_echo_signals
      (composes echo_path_generator.py's office-room RIR with
      generate_test_signals' speech convolution)
    - echo_path_generator.py::generate_and_plot_echo_path (console report
      + time/frequency plots)
    - harness_template/ground_truth/loaders.py::save_wav (audio writing)

Outputs:
    audio/room_reference.wav   clean far-end speech
    audio/room_microphone.wav  echo-contaminated microphone signal
    audio/room_rir.wav         the room impulse response
    results/<timestamp>_room-echo/  generator's plots + metrics.json + log

Run: .venv/Scripts/python create_room_echo.py [seed]
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

from echo_path_generator import RoomParameters, generate_and_plot_echo_path
from ground_truth.generators import generate_room_echo_signals
from ground_truth.loaders import save_wav


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    speech = sys.argv[2] if len(sys.argv) > 2 else 'reference.wav'

    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(HERE, 'results', f'{ts}_room-echo')
    os.makedirs(out_dir, exist_ok=True)

    # --- the project's echo path generator (its own console report + plots)
    h, H, freqs = generate_and_plot_echo_path(seed=seed, save_plots=True,
                                              output_dir=out_dir)

    # --- echo-contaminated audio through the harness generator
    params = RoomParameters()
    ref, mic, h_sig = generate_room_echo_signals(
        seed=seed, signal_length=160000, sample_rate=params.sampling_rate,
        speech_file=os.path.join(HERE, 'audio', speech))
    gain = 0.95 / max(np.max(np.abs(ref)), 1e-9)      # create_test_files convention
    ref, mic = ref * gain, mic * gain

    save_wav(os.path.join(HERE, 'audio', 'room_reference.wav'),
             ref.astype(np.float32), params.sampling_rate)
    save_wav(os.path.join(HERE, 'audio', 'room_microphone.wav'),
             mic.astype(np.float32), params.sampling_rate)
    save_wav(os.path.join(HERE, 'audio', 'room_rir.wav'),
             (h / (np.max(np.abs(h)) + 1e-30) * 0.9).astype(np.float32),
             params.sampling_rate)

    peak = int(np.argmax(np.abs(h)))
    ratio = 10 * np.log10(np.sum(mic ** 2) / np.sum(ref ** 2))
    print(f"\ndirect path: {peak} samples ({peak/params.sampling_rate*1000:.1f} ms)")
    print(f"echo power ratio mic/ref: {ratio:.2f} dB")
    print("wrote audio/room_reference.wav, audio/room_microphone.wav, "
          "audio/room_rir.wav")

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'seed': seed, 'speech_file': speech,
                   'params': vars(RoomParameters()),
                   'direct_delay_samples': peak,
                   'echo_power_ratio_db': ratio,
                   'rir_energy': float(np.sum(h ** 2))}, fh, indent=2)
    print(f"artifacts in {out_dir}")


if __name__ == '__main__':
    main()

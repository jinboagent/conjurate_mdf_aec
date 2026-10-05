"""
Block-based Echo Cancellation test: NLMS (time-domain) vs CG-MDF (frequency-domain).

Single-talk scenario: the microphone picks up a delayed + attenuated copy of the
loudspeaker signal (audio/reference.wav -> audio/microphone.wav). The adaptive
filter learns the echo path; its output (error signal) should approach silence.

Outputs (written to the current directory):
    processed_signal_<algo>.wav                 residual error signal
    echo_cancellation_spectrograms_<algo>.png   ref / mic / output / estimated echo
    spectral_erle_<algo>.png                    per-frequency ERLE
and prints delay estimation, ERLE, band ERLE, echo-estimation correlation and
learned-path metrics (shared implementations from harness/metrics/).

CLI keys: 'cgmdf' (default; alias 'rls'), 'fdnlms', 'nlms' (alias 'lms').
"""

import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conjugate_mdf import CONJUGATE_MDF, FD_NLMS
from harness.metrics.comparison import (calculate_correlation,
                                        calculate_echo_path_metrics,
                                        calculate_erle)
from harness.metrics.spectrum import (band_erle_db, plot_spectral_erle,
                                      plot_spectrograms, spectral_erle_db)

SR = 16000                       # sample rate of the test files
CHUNK = 256                      # NLMS call granularity (processed per sample)
TRANSIENT = 4 * int(0.25 * SR)   # skip 1 s of filter convergence


# ---------------------------------------------------------------------------
# Adaptive filters
# ---------------------------------------------------------------------------

class NLMS:
    """Sample-by-sample time-domain NLMS."""

    def __init__(self, length, mu=0.5):
        self.w = np.zeros(length)
        self.mu = mu
        self.x_hist = np.zeros(length - 1)

    def process(self, x, d):
        e = np.empty(len(x))
        for n in range(len(x)):
            window = np.concatenate(([x[n]], self.x_hist))
            y = self.w @ window
            e[n] = d[n] - y
            self.w += (self.mu / (window @ window + 1e-6)) * e[n] * window
            self.x_hist = window[:-1]
        return e


class CGMDF:
    """FFT-frame wrapper around CONJUGATE_MDF (overlap-save: one 512-frame in,
    128 valid tail samples out) — the canonical [0;e] MDF with the
    excitation gate; beta is the gradient-averaging factor (0 = exactly
    FD_NLMS's instantaneous update)."""

    def __init__(self, fft_size=512, step=128, n_g=8, mu=1.0, beta=0.0,
                 bin_skip=0, gate_rel=0.3):
        self.fft_size, self.step = fft_size, step
        self.cg = CONJUGATE_MDF(NCHAN=1, NBIN=fft_size // 2 + 1, N_G=n_g,
                                hop=step, mu=mu, beta=beta,
                                bin_skip=bin_skip, gate_rel=gate_rel)

    def process(self, x_frame, d_frame):
        E = self.cg.apply(np.fft.rfft(d_frame).reshape(-1, 1),
                          np.fft.rfft(x_frame).reshape(-1, 1))
        return np.fft.irfft(E[:, 0], n=self.fft_size)[-self.step:]


class FDNLMS:
    """FFT-frame wrapper around FD_NLMS — identical buffer/output geometry
    to CGMDF, instantaneous per-frame NLMS update ([0;e] criterion)."""

    def __init__(self, fft_size=512, step=128, n_g=8, mu=1.0, gate_rel=0.3):
        self.fft_size, self.step = fft_size, step
        self.fdn = FD_NLMS(NCHAN=1, NBIN=fft_size // 2 + 1, N_G=n_g,
                           mu=mu, hop=step, gate_rel=gate_rel)

    def process(self, x_frame, d_frame):
        E = self.fdn.apply(np.fft.rfft(d_frame).reshape(-1, 1),
                           np.fft.rfft(x_frame).reshape(-1, 1))
        return np.fft.irfft(E[:, 0], n=self.fft_size)[-self.step:]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def estimate_echo(ref, mic, sr):
    """Delay and gain of the mic signal w.r.t. the reference (FFT cross-correlation)."""
    corr = signal.correlate(ref, mic, mode='full', method='fft')
    delay = abs(int(np.argmax(np.abs(corr))) - (len(mic) - 1))
    L = min(8000, len(ref) - delay)
    gain = float(ref[:L] @ mic[delay:delay + L] / (ref[:L] @ ref[:L] + 1e-12))
    print(f"Echo delay: {delay} samples ({delay / sr * 1000:.1f} ms), "
          f"attenuation {gain:.2f} ({20 * np.log10(gain):.1f} dB)")
    return delay, gain


def run(algorithm, ref, mic):
    """Process the whole file; returns (residual signal, filter)."""
    out = np.zeros(len(ref))
    if algorithm == 'nlms':
        f = NLMS(int(0.25 * SR))
        for i in tqdm(range(0, len(ref), CHUNK), desc='NLMS'):
            j = min(i + CHUNK, len(ref))
            out[i:j] = f.process(ref[i:j], mic[i:j])
        return out, f

    f = FDNLMS() if algorithm == 'fdnlms' else CGMDF()
    head = f.fft_size - f.step      # overlap-save: discard contaminated head
    for i in tqdm(range((len(ref) - head) // f.step), desc=algorithm.upper()):
        s = i * f.step
        out[s + head:s + f.fft_size] = f.process(ref[s:s + f.fft_size],
                                                 mic[s:s + f.fft_size])
    return out, f


def main(algorithm='cgmdf'):
    algorithm = {'lms': 'nlms', 'rls': 'cgmdf'}.get(algorithm, algorithm)
    print(f"{'=' * 60}\nEcho cancellation test: {algorithm.upper()}\n{'=' * 60}")

    audio_dir = os.path.join(os.path.dirname(__file__), 'audio')
    ref, sr = sf.read(os.path.join(audio_dir, 'reference.wav'))
    mic, _ = sf.read(os.path.join(audio_dir, 'microphone.wav'))
    print(f"{len(ref)} samples ({len(ref) / sr:.1f} s) @ {sr} Hz")
    delay, gain = estimate_echo(ref, mic, sr)

    out, f = run(algorithm, ref, mic)

    # ---- metrics (shared harness implementations) ----
    erle = calculate_erle(mic, out, TRANSIENT)
    corr = calculate_correlation(mic[TRANSIENT:], (mic - out)[TRANSIENT:])
    verdict = 'PASS' if (erle > 15 and corr > 0.9) else 'FAIL'
    print(f"\nERLE (post-transient): {erle:7.2f} dB  (target > 15 dB)")
    print(f"Echo-est correlation:  {corr:7.4f}    (target > 0.9)")
    print(f"RMS: {np.sqrt(np.mean(mic[TRANSIENT:] ** 2)):.5f} -> "
          f"{np.sqrt(np.mean(out[TRANSIENT:] ** 2)):.5f}   [{verdict}]")
    print("Band ERLE (dB): " + ", ".join(
        f"{k}: {v:.1f}" for k, v in
        band_erle_db(mic, out, sr, transient=TRANSIENT).items()))

    # ---- learned-path check ----
    if algorithm == 'nlms':
        true_path = np.zeros(len(f.w))
        true_path[delay] = gain
        m = calculate_echo_path_metrics(f.w, true_path)
        print(f"Path: peak {np.max(np.abs(f.w)):.3f} @ {int(np.argmax(np.abs(f.w)))} "
              f"(true {gain:.2f} @ {delay}); delay error {m['delay_error_samples']} "
              f"samples, amplitude error {m['amplitude_error_db']:.1f} dB, "
              f"NMSE {m['nmse_db']:.1f} dB")
    else:
        w = f.cg.w[0] if algorithm != 'fdnlms' else f.fdn.w[0]
        print(f"{'CG-MDF' if algorithm != 'fdnlms' else 'FD-NLMS'} |W|: "
              f"max {np.abs(w).max():.3f}, mean {np.abs(w).mean():.3f}")

    # ---- artifacts ----
    sf.write(f'processed_signal_{algorithm}.wav', out, sr)
    plot_spectrograms({'Reference (loudspeaker)': ref,
                       'Microphone (echo)': mic,
                       f'Output after {algorithm.upper()}': out,
                       'Estimated echo': mic - out},
                      sr, f'echo_cancellation_spectrograms_{algorithm}.png')
    plot_spectral_erle({algorithm.upper(): spectral_erle_db(mic, out, sr,
                                                            transient=TRANSIENT)},
                       f'spectral_erle_{algorithm}.png')
    print(f"Saved processed_signal_{algorithm}.wav, "
          f"echo_cancellation_spectrograms_{algorithm}.png, "
          f"spectral_erle_{algorithm}.png")
    return erle


if __name__ == '__main__':
    main(sys.argv[1].lower() if len(sys.argv) > 1 else 'cgmdf')

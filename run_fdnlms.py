"""
Run FD_NLMS (or the CONJUGATE_MDF hop mode) on a wav pair — the common
driving loop used by all the session benchmarks, restored as a permanent
script (the scratch_*.py originals were cleaned up; their results live in
results/*/FINDINGS.md).

Geometry: overlap-save, FFT 512 / hop 128 (75% overlap). The class takes
FULL frame spectra in and returns the zero-headed a priori error spectrum;
the wrapper keeps the last `hop` samples of irfft(output) as the 128 new
output samples. `out` IS the residual stream (ERLE = mic/out; the echo
estimate is mic - out).

Metrics: official harness convention (skip first 1 s).

Usage:
    .venv/Scripts/python run_fdnlms.py [pair] [options]
    pair: reference microphone | noise | speechlp | reverb_conference | canonical
      ("reference microphone" loads audio/reference.wav + audio/microphone.wav;
       "noise" -> audio/noise_*; "speechlp" -> audio/speechlp_*;
       "reverb_conference" -> audio/reverb_conference_*;
       "canonical" is an alias of "reference microphone")

    Options:
    --algo fdnlms|cgmdf   (default fdnlms)
    --n-g N               partitions (default 8)
    --mu F                step (default 1.0)
    --beta F              cgmdf gradient averaging (default 0.3)
    --no-constraint       disable the G=[I_hop,0] weight constraint
    --no-gate             disable the reference-excitation gate
    --delay D             synthetic pure-delay pair from the noise reference
    --per-second          also print per-second ERLE
"""

import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from conjugate_mdf import CONJUGATE_MDF, FD_NLMS
from harness.metrics.comparison import calculate_correlation, calculate_erle

FFT, HOP, head = 512, 128, 384
TR = 16000                                   # official metric: skip 1 s


def load_pair(pair):
    if pair in ('canonical', 'reference microphone'):
        files = ('reference', 'microphone')
    elif pair == 'noise':
        files = ('noise_reference', 'noise_microphone')
    elif pair == 'speechlp':
        files = ('speechlp_reference', 'speechlp_microphone')
    elif pair == 'reverb_conference':
        files = ('reverb_conference_reference', 'reverb_conference_microphone')
    else:
        raise SystemExit(f"unknown pair {pair!r}")
    ref, sr = sf.read(os.path.join(HERE, 'audio', f'{files[0]}.wav'))
    mic, _ = sf.read(os.path.join(HERE, 'audio', f'{files[1]}.wav'))
    return ref, sr, mic


def drive(f, ref, mic):
    """The common driving loop. Returns the residual stream."""
    out = np.zeros(len(ref))
    for i in range((len(ref) - FFT) // HOP):
        s = i * HOP
        E = f.apply(np.fft.rfft(mic[s:s+FFT]).reshape(-1, 1),
                    np.fft.rfft(ref[s:s+FFT]).reshape(-1, 1))
        out[s+head:s+FFT] = np.fft.irfft(E[:, 0], n=FFT)[head:]
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pair', nargs='?', default='canonical')
    p.add_argument('--algo', choices=['fdnlms', 'cgmdf'], default='fdnlms')
    p.add_argument('--n-g', type=int, default=8)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=0.0)
    p.add_argument('--no-constraint', action='store_true')
    p.add_argument('--no-gate', action='store_true')
    p.add_argument('--delay', type=int, default=None)
    p.add_argument('--per-second', action='store_true')
    args = p.parse_args()

    cons = not args.no_constraint
    gate = None if args.no_gate else 0.3

    if args.delay is not None:
        ref, sr = sf.read(os.path.join(HERE, 'audio', 'noise_reference.wav'))
        mic = np.concatenate([np.zeros(args.delay), ref])[:len(ref)]
        tag = f'synthetic delay {args.delay}'
    else:
        ref, sr, mic = load_pair(args.pair)
        tag = args.pair

    if args.algo == 'fdnlms':
        f = FD_NLMS(NCHAN=1, NBIN=FFT//2+1, N_G=args.n_g, mu=args.mu,
                    hop=HOP, constraint=cons, gate_rel=gate)
    else:
        f = CONJUGATE_MDF(NCHAN=1, NBIN=FFT//2+1, N_G=args.n_g, hop=HOP,
                          mu=args.mu, beta=args.beta, gate_rel=gate)
    out = drive(f, ref, mic)

    n = min(len(mic), len(out))
    e = calculate_erle(mic[:n], out[:n], TR)
    c = calculate_correlation(mic[TR:n], (mic[:n]-out[:n])[TR:])
    verdict = 'PASS' if (e > 15 and c > 0.9) else 'FAIL'
    print(f"{args.algo} n_g={args.n_g} mu={args.mu}" +
          (f" beta={args.beta}" if args.algo != 'fdnlms' else '') +
          f" constraint={cons} gate={gate}  [{tag}]")
    print(f"  ERLE {e:7.2f} dB (target > 15), corr {c:.4f} (target > 0.9)"
          f"  [{verdict}]")
    if args.per_second:
        secs = [round(10*np.log10(np.sum(mic[s:s+16000]**2) /
                                  np.sum(out[s:s+16000]**2)), 1)
                for s in range(TR, n - 16000, 16000)]
        print(f"  per-second ERLE: {secs}")
    # learned path: per-lag time-domain peak
    w = f.w[0][:, :, 0]                                   # [nbin, n_g]
    taps = np.fft.irfft(w, n=FFT, axis=0)[:HOP, :]        # [hop, n_g] taps
    lag_pk = int(np.argmax(np.abs(taps).max(axis=0)))
    tap_pk = int(np.argmax(np.abs(taps[:, lag_pk])))
    print(f"  learned path peak {np.abs(taps[:, lag_pk]).max():.4f} "
          f"@ lag {lag_pk}, tap {tap_pk} (delay {lag_pk*HOP + tap_pk})")


if __name__ == '__main__':
    main()

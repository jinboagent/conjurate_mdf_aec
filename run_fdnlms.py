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
    --algo fdnlms|cgmdf|pfcg|lalos
                          (default fdnlms; pfcg = PBFDAF-CG, AES 2006;
                          lalos = FD_CG_WIENER, the faithful port of
                          Lalos & Berberidis EUSIPCO 2006 — FD CG on the
                          Wiener normal equations with exact linear
                          statistics; correlation family, no residual in
                          the update. The legacy full-frame toeplitz engine
                          was removed 2026-10-06 — criterion-cliff
                          counterexample, code recoverable from git 2f23d9d)
    --n-g N               partitions (default 8; fdnlms/cgmdf/pfcg only)
    --mu F                step (default 1.0)
    --beta F              cgmdf gradient averaging (default 0.3)
    --gamma F             pfcg gradient-memory averaging (default 0.4)
    --k-max N             pfcg CG iterations per frame (default 1; >1 diverges)
    --beta-method M       pfcg conjugation: hestenes-stiefel (default) |
                          fletcher-reeves | polak-ribiere | dai-yuan
    --lam F               lalos forgetting factor (default 0.99999)
    --reset-period N      lalos CG direction reset period in blocks (default 32)
    --hop N               lalos block size M (default 1024; nfft = 2M, the
                          filter IS one hop — M taps; must cover the path)
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

from FD_NLMS import FD_NLMS, FD_CG_WIENER
from pfdaf_cg import PFDAF_CG
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
    elif pair == 'reverb30':
        files = ('reverb_conf30_reference', 'reverb_conf30_microphone')
    elif pair == 'reverb30n':
        files = ('reverb_conf30n_reference', 'reverb_conf30n_microphone')
    else:
        raise SystemExit(f"unknown pair {pair!r}")
    ref, sr = sf.read(os.path.join(HERE, 'audio', f'{files[0]}.wav'))
    mic, _ = sf.read(os.path.join(HERE, 'audio', f'{files[1]}.wav'))
    return ref, sr, mic


def drive(f, ref, mic, fft=None, hop=None):
    """The common driving loop. Returns the residual stream."""
    fft = FFT if fft is None else fft
    hop = HOP if hop is None else hop
    head = fft - hop
    out = np.zeros(len(ref))
    for i in range((len(ref) - fft) // hop):
        s = i * hop
        E = f.apply(np.fft.rfft(mic[s:s+fft]).reshape(-1, 1),
                    np.fft.rfft(ref[s:s+fft]).reshape(-1, 1))
        out[s+head:s+fft] = np.fft.irfft(E[:, 0], n=fft)[head:]
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pair', nargs='?', default='canonical')
    p.add_argument('--algo', choices=['fdnlms', 'cgmdf', 'pfcg', 'lalos'],
                   default='fdnlms')
    p.add_argument('--n-g', type=int, default=8)
    p.add_argument('--mu', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=None,
                   help='cgmdf gradient averaging (default 0.3)')
    p.add_argument('--gamma', type=float, default=0.4)
    p.add_argument('--k-max', type=int, default=1)
    p.add_argument('--beta-method', default='hestenes-stiefel',
                   choices=['hestenes-stiefel', 'fletcher-reeves',
                            'polak-ribiere', 'dai-yuan'])
    p.add_argument('--lam', type=float, default=0.99999,
                   help='lalos forgetting factor (effective memory '
                        '~1/(1-lam^hop) blocks)')
    p.add_argument('--reset-period', type=int, default=32,
                   help='lalos CG direction reset period (blocks)')
    p.add_argument('--hop', type=int, default=None,
                   help='lalos block size M (default 1024; nfft=2M, filter '
                        '= M taps, must cover the path length)')
    p.add_argument('--no-constraint', action='store_true')
    p.add_argument('--no-gate', action='store_true')
    p.add_argument('--proportionate', type=float, default=0.0,
                   help='FD_NLMS proportionate step rho (0 = uniform, the '
                        'classic update; 0.5-0.9 favors partitions that '
                        'already carry weight — sparse paths converge '
                        'faster)')
    p.add_argument('--preemph', type=float, default=0.0,
                   help='pre-emphasis alpha (0 = off; 0.9-0.97 whitens '
                        'speech). fdnlms/cgmdf: applied INSIDE the class, '
                        'output de-emphasized -> ERLE directly in the raw '
                        'domain. pfcg: pair-level preprocessing, ERLE in '
                        'the whitened domain (de-emphasize for raw).')
    p.add_argument('--delay', type=int, default=None)
    p.add_argument('--per-second', action='store_true')
    args = p.parse_args()

    cons = not args.no_constraint
    gate = None if args.no_gate else 0.3
    if args.beta is None:
        args.beta = 0.3
    FFT, HOP = 512, 128          # geometry (overridden by --algo lalos)

    if args.delay is not None:
        ref, sr = sf.read(os.path.join(HERE, 'audio', 'noise_reference.wav'))
        mic = np.concatenate([np.zeros(args.delay), ref])[:len(ref)]
        tag = f'synthetic delay {args.delay}'
    else:
        ref, sr, mic = load_pair(args.pair)
        tag = args.pair
    # pre-emphasis: fdnlms/cgmdf do it INSIDE the class (output comes back
    # in the raw domain); pfcg has no class option, so pre-filter the pair
    # (metric then in the whitened domain — de-emphasize for raw)
    class_preemph = args.preemph if args.algo in ('fdnlms', 'cgmdf') else 0.0
    if args.preemph and args.algo == 'pfcg':
        a = float(args.preemph)
        ref = np.concatenate([[0.0], ref[1:] - a * ref[:-1]])
        mic = np.concatenate([[0.0], mic[1:] - a * mic[:-1]])
    if args.preemph:
        tag += f' preemph={args.preemph}'

    if args.algo == 'fdnlms':
        f = FD_NLMS(NCHAN=1, NBIN=FFT//2+1, N_G=args.n_g, mu=args.mu,
                    hop=HOP, constraint=cons, rho=args.proportionate,
                    preemph=class_preemph, gate_rel=gate)
    elif args.algo == 'pfcg':
        f = PFDAF_CG(NCHAN=1, NBIN=FFT//2+1, N_G=args.n_g, hop=HOP,
                     gamma=args.gamma, k_max=args.k_max,
                     beta_method=args.beta_method, constrain='full',
                     gate_rel=gate)
    elif args.algo == 'lalos':
        hop = args.hop if args.hop else 1024
        FFT, HOP = 2 * hop, hop
        f = FD_CG_WIENER(NCHAN=1, NBIN=FFT//2+1, N_G=1, hop=HOP,
                         lam=args.lam, reset_period=args.reset_period)
    else:
        f = FD_NLMS(NCHAN=1, NBIN=FFT//2+1, N_G=args.n_g, mu=args.mu,
                    hop=HOP, beta=args.beta, rho=args.proportionate,
                    preemph=class_preemph, gate_rel=gate)
    out = drive(f, ref, mic, FFT, HOP)

    n = min(len(mic), len(out))
    e = calculate_erle(mic[:n], out[:n], TR)
    c = calculate_correlation(mic[TR:n], (mic[:n]-out[:n])[TR:])
    verdict = 'PASS' if (e > 15 and c > 0.9) else 'FAIL'
    if args.algo == 'lalos':
        print(f"lalos hop={HOP} nfft={FFT} lam={args.lam} "
              f"reset={args.reset_period}  [{tag}]")
    else:
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
    if args.algo == 'lalos':
        taps = f.w[:, 0]
        tap_pk = int(np.argmax(np.abs(taps)))
        print(f"  learned path peak {np.abs(taps[tap_pk]):.4f} "
              f"@ tap {tap_pk} (delay {tap_pk})")
    else:
        w = f.w[0][:, :, 0]                               # [nbin, n_g]
        taps = np.fft.irfft(w, n=FFT, axis=0)[:HOP, :]    # [hop, n_g] taps
        lag_pk = int(np.argmax(np.abs(taps).max(axis=0)))
        tap_pk = int(np.argmax(np.abs(taps[:, lag_pk])))
        print(f"  learned path peak {np.abs(taps[:, lag_pk]).max():.4f} "
              f"@ lag {lag_pk}, tap {tap_pk} (delay {lag_pk*HOP + tap_pk})")


if __name__ == '__main__':
    main()

"""
Spectral analysis for AEC test reports.

Spectral metrics (Welch-based per-frequency and per-band ERLE) plus the
traditional librosa spectrogram rendering used by
test_subband_echo_cancellation.py (stft -> amplitude_to_db -> specshow,
log frequency axis).
"""

import numpy as np

DEFAULT_BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000))


def welch_spectrum_db(x, sr, nfft=1024, hop=512, transient=0):
    """Average-periodogram power spectrum in dB. Returns (freqs, power_db)."""
    frames = np.stack([x[i:i + nfft]
                       for i in range(transient, len(x) - nfft, hop)])
    P = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    return f, 10 * np.log10(P + 1e-30)


def spectral_erle_db(mic, e, sr, nfft=1024, hop=512, transient=0):
    """Per-frequency ERLE: 10*log10(P_mic(f) / P_residual(f))."""
    f, P_mic = welch_spectrum_db(mic, sr, nfft, hop, transient)
    _, P_e = welch_spectrum_db(e, sr, nfft, hop, transient)
    return f, P_mic - P_e


def band_erle_db(mic, e, sr, bands=DEFAULT_BANDS, nfft=1024, hop=512,
                 transient=0):
    """Mean spectral ERLE per frequency band. Returns {band: dB}."""
    f, E = spectral_erle_db(mic, e, sr, nfft, hop, transient)
    return {f"{lo}-{hi} Hz": float(np.mean(E[(f >= lo) & (f < hi)]))
            for lo, hi in bands}


def plot_spectrograms(signals, sr, out_path, n_fft=512, hop=128):
    """
    Traditional (librosa) spectrograms, one row per signal.

    signals: ordered dict {title: signal}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, sig) in zip(axes, signals.items()):
        D = librosa.stft(np.asarray(sig, dtype=np.float32))
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        im = librosa.display.specshow(S_db, sr=sr, x_axis='time',
                                      y_axis='log', ax=ax)
        ax.set_title(title)
        plt.colorbar(im, format='%+2.f dB', ax=ax, pad=0.01)
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_spectral_erle(curves, out_path):
    """
    Overlay per-frequency ERLE curves.

    curves: {name: (freqs, erle_db)}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, (f, E) in curves.items():
        ax.plot(f, E, linewidth=1.2, label=name)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('spectral ERLE (dB)')
    ax.set_title('Per-frequency echo suppression (Welch, post-transient)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_paths(paths, out_path, n_show=None, log_y=False):
    """
    Impulse-response comparison.

    paths: {name: h ndarray}; n_show: number of samples to display.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, h in paths.items():
        hh = h[:n_show] if n_show else h
        style = 'k-' if name.lower().startswith('true') else '-'
        ax.plot(hh, style, linewidth=1.1, label=name)
    if log_y:
        ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xlabel('samples')
    ax.set_ylabel('amplitude')
    ax.set_title('Echo path: true vs estimated')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

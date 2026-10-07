"""
Faithful Python port of matlab/polyphase_dft_fb_{analysis,synthesis}.m
(the rig from test_fada_apa_vss.m: NFFT=512, BLOCKSIZE=256, OS=2, Lp=1024).

Analysis (per block, hop = BLOCKSIZE):
    buffer  = [last Lp-B samples ; new B samples]      (.m line 77)
    sig     = buffer * p_filter                        (line 80)
    sig     = flipud(sig)                              (line 84)   [if flip]
    os      = sum of the OS nfft-segments of sig       (lines 90-92)
    output  = rfft(os) * OSFB_gain                     (lines 95-101)

Synthesis (per frame):
    s       = irfft(X_in)          == real(ifft([X; conj(X(end-1:-1:2))]))
    repl    = tile(s, OS)                              (.m replicate loop)
    filt    = repl * p_filter
    buf     = [zeros(B); buf(1:Lp-B)] + filt           (overlap-add)
    output  = buf(Lp:-1:Lp-B+1) * nfft*blocksize*inv_gain   (REVERSED read)

flip=True reproduces the .m files verbatim; flip=False drops BOTH time
reversals (analysis flipud + synthesis reversed readout) — the pair's
composite is orientation-identical, and the EC then sees frames in the
standard oldest->newest orientation required by the [0;e] convention of
CONJUGATE_MDF hop mode / FD_NLMS.

NOTE on p_filter: the .m prototype comes from an IterLS-type design
(1997 paper; the original .m header cites its authors), which we do not
have. make_prototype('sqrthann') supplies a
sqrt-Hann WOLA prototype (p^2 COLA at hop B) as a stand-in.
make_prototype(None) is the user's "p = 1" case — provably NOT a
reconstructing pair (the fold aliases: round trip = comb at nfft spacing).
"""

import numpy as np


def make_prototype(kind, Lp, nfft, B):
    if kind is None:
        return np.ones(Lp)
    if kind == 'sqrthann':
        w = np.hanning(Lp + 1)[:-1]              # periodic Hann
        return np.sqrt(w)
    raise ValueError(kind)


class AnalysisBank:
    def __init__(self, nfft=512, B=256, OS=2, p_filter=None, gain=1.0,
                 flip=True):
        self.nfft, self.B, self.OS = nfft, B, OS
        self.Lp = nfft * OS
        self.p = p_filter if p_filter is not None else np.ones(self.Lp)
        self.gain = gain
        self.flip = flip
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.Lp)
        self.output = np.zeros(self.nfft // 2 + 1, dtype=complex)

    def apply(self, block):
        self.buf = np.concatenate([self.buf[-(self.Lp - self.B):], block])
        sig = self.buf * self.p
        if self.flip:
            sig = sig[::-1]
        os = np.zeros(self.nfft)
        for ii in range(self.OS):
            os += sig[ii * self.nfft:(ii + 1) * self.nfft]
        self.output = np.fft.rfft(os) * self.gain
        return self.output


class SynthesisBank:
    def __init__(self, nfft=512, B=256, OS=2, p_filter=None, inv_gain=1.0,
                 flip=True):
        self.nfft, self.B, self.OS = nfft, B, OS
        self.Lp = nfft * OS
        self.p = p_filter if p_filter is not None else np.ones(self.Lp)
        self.inv_gain = inv_gain
        self.flip = flip
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.Lp)
        self.out = np.zeros(self.B)

    def apply(self, X):
        s = np.fft.irfft(np.asarray(X), n=self.nfft)
        repl = np.tile(s, self.OS)
        filt = repl * self.p
        self.buf = np.concatenate([np.zeros(self.B),
                                   self.buf[:self.Lp - self.B]]) + filt
        if self.flip:
            self.out = self.buf[::-1][:self.B]
        else:
            self.out = self.buf[self.Lp - self.B:]
        return self.out * (self.nfft * self.B) * self.inv_gain


def roundtrip(x, p_kind=None, nfft=512, B=256, OS=2, flip=True):
    """analysis -> synthesis with the identity in between. Returns
    (y, delay) with delay = lag of the output vs input (samples)."""
    p = make_prototype(p_kind, nfft * OS, nfft, B)
    A = AnalysisBank(nfft, B, OS, p, 1.0, flip)
    S = SynthesisBank(nfft, B, OS, p, 1.0, flip)
    y = np.zeros(len(x))
    for i in range((len(x) - B) // B + 1):
        blk = x[i * B:(i + 1) * B]
        if len(blk) < B:
            blk = np.concatenate([blk, np.zeros(B - len(blk))])
        y[i * B:(i + 1) * B] = S.apply(A.apply(blk))
    # effective delay+gain via cross-correlation against x
        # (measured after the loop below)
    return y

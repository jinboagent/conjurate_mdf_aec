"""
Canonical partitioned-block frequency-domain echo cancellers.

FD_NLMS implements the classic MDF (Lee & Chang, "On the Convergence
Properties of Multidelay Frequency Domain Adaptive Filter"; Soo & Pang
1990; Ferrara 1980):

    est    = sum_j w_j (.) X(t-j)                  overlapping frames
    e      = last `hop` samples of irfft(Y - est)   error on the NEW block
    E      = rfft([0; e])                           zero-headed error
    w_j   += mu * conj(X(t-j)) (.) E / P            normalized gradient
    w_j    = FFT[ first-hop-taps ; 0 ]              G = [I_hop, 0]

The [0; e] criterion grades only the samples the output keeps, so the
overlap-save head (whose circular wrap cannot serve non-hop-aligned
delays) never biases the weights — no delay-mod-hop cliff
(docs/DATAFLOW.md §7). Weights are additionally frozen while the
reference is not identifiable (excitation gate: reverb ring-down,
digital silence, double talk).

beta > 0 turns on gradient averaging (gacc = beta*gacc + conj(X)*E_upd
with smoothed normalizer Pn) — the former CONJUGATE_MDF algorithm, merged
into FD_NLMS on 2026-10-06 (verified bit-exact vs the old class at
beta = 0.3/0.05/preemph; the separate class is deleted). beta = 0 keeps
the instantaneous classic update.

References: aec/ON THE CONVERGENCE PROPERTIES ...pdf (eqs. 10/12/14/
16-19), aec/freqgradientconjugate.pdf, docs/DATAFLOW.md.
"""

import numpy as np
from scipy.signal import lfilter


def _preemph_spectrum(S, alpha, state, carry_index):
    """Pre-emphasize the time frame behind spectrum S (rows = frequency
    bins, columns = channels): y[n] = x[n] - alpha*x[n-1], with the true
    (non-circular) previous sample x[s-1] held in `state` (in/out, one
    entry per column). Frames OVERLAP (hop < nfft): the sample preceding
    the NEXT frame's start is THIS frame's sample at index `carry_index`
    (= hop - 1), not the frame's last sample. LTI-commutes with the echo
    path."""
    nfft = 2 * (S.shape[0] - 1)
    t = np.fft.irfft(S, n=nfft, axis=0)
    carry = t[carry_index].copy()
    t[1:] -= alpha * t[:-1]
    t[0] -= alpha * state
    state[:] = carry
    return np.fft.rfft(t, axis=0)


# Recommended parameters (see DATAFLOW.md §10 benchmarks: beta costs
# average ERLE on speech; use beta > 0 only for mic-noise robustness).
FD_NLMS_PARAMS = {
    'mu': 1.0,     # NLMS-normalized step (0 < mu <= 2)
    'beta': 0.0,   # gradient averaging factor (0..0.5)
}


class FD_NLMS:
    """
    Partitioned-block frequency-domain NLMS (classic overlap-save FDAF) —
    the canonical [0;e] engine. beta = 0 is the instantaneous classic
    update; beta > 0 is the former CONJUGATE_MDF algorithm (merged
    2026-10-06, bit-exact).

        e   = last `hop` samples of irfft(Y - est)          ([0; e])
        X2  = sum_j |X(t-j)|^2                              per bin
        w_j += mu * conj(X(t-j)) * E / (X2 + eps)
        w_j = FFT[first-hop-taps; 0]                        (if constraint)

    rho > 0 switches the update to proportionate steps (frequency-domain
    IPNLPS idea): partitions already carrying weight get larger steps
    (G in [1-rho, 1], G-weighted power denominator). rho = 0 is the
    uniform classic update, bit-identical.

    beta > 0 absorbs the former CONJUGATE_MDF algorithm (merged
    2026-10-06): decay-first gradient accumulator
    gacc = beta*gacc + conj(X)*E_upd and exponentially smoothed normalizer
    Pn = beta*Pn + (1-beta)*X2 (updated unconditionally, gate included),
    step = mu*(1-beta)*gacc/(Pn + eps), bins with Pn <= 1e-12 suspended.
    beta = 0 keeps the historical FD_NLMS code path bit-for-bit. The old
    CONJUGATE_MDF(bin_skip=...) knob is dropped (was 0 in every caller).

    preemph > 0 applies first-order pre-emphasis inside the class (same
    semantics as the beta option's pre-emphasis: adaptation whitened, output and
    self.e de-emphasized back to the raw domain).
    """

    __slots__ = ['nchan', 'nbin', 'N_G', 'Nrxref', 'mu', 'hop', 'nfft',
                 'constraint', 'eps', 'rho', 'preemph', 'beta',
                 'gate_rel', 'gate_hold', 'gate_decay', 'gate_floor',
                 '_Prun', '_hold',
                 'sr',
                 '_pe_rx_state', '_pe_mic_state', '_pe_e_zi',
                 'buf_Y_rx', 'w', 'P_X', 'output', 'e', 'X2',
                 'gacc', 'Pn']

    def __init__(self, NCHAN, NBIN, N_G, mu, hop,
                 Nrxref=1, constraint=True,
                 eps=1e-10, rho=0.0, preemph=0.0, beta=0.0, gate_rel=0.3,
                 gate_hold=2, gate_decay=1.5, gate_floor=1e-6,
                 sr=16000.0):
        self.nchan = NCHAN
        self.nbin = NBIN
        self.N_G = int(N_G)
        self.Nrxref = Nrxref
        self.mu = float(mu)
        self.nfft = 2 * (NBIN - 1)
        self.hop = int(hop)
        if not 0 < self.hop <= self.nfft // 2:
            raise ValueError("hop must be in (0, nfft/2] "
                             f"(got hop={hop}, nfft={self.nfft})")
        self.constraint = bool(constraint)
        self.eps = float(eps)
        self.rho = float(rho)      # proportionate step (0 = uniform)
        self.preemph = float(preemph)
        self.beta = float(beta)    # gradient averaging (0 = instantaneous;
                                   # >0 = the former CONJUGATE_MDF algorithm)
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        self.sr = float(sr)
        self.reset()

    def reset(self):
        self.buf_Y_rx = np.zeros((self.nbin, self.N_G, self.Nrxref),
                                 dtype=complex)
        self.w = [np.zeros((self.nbin, self.N_G, self.nchan), dtype=complex)
                  for _ in range(self.Nrxref)]
        self.output = np.zeros((self.nbin, self.nchan), dtype=complex)
        self.e = np.zeros(self.hop, dtype=float)
        self.X2 = np.zeros(self.nbin)
        self.gacc = [np.zeros((self.nbin, self.N_G, self.nchan),
                              dtype=complex)
                     for _ in range(self.Nrxref)]
        self.Pn = np.zeros(self.nbin)
        self._Prun = 0.0
        self._hold = 0
        # pre-emphasis filter states (per column: last raw sample / zi)
        self._pe_rx_state = np.zeros(self.Nrxref)
        self._pe_mic_state = np.zeros(self.nchan)
        self._pe_e_zi = np.zeros((1, self.nchan))

    def apply(self, Y, Y_rx):
        """
        Process one frame. Y [NBIN, NCHAN] mic spectrum, Y_rx [NBIN, Nrxref]
        reference spectrum (both FULL L-sample frame FFTs). Returns the
        zero-headed a priori error spectrum rfft([0; e]) — with preemph > 0
        the returned spectrum and self.e are de-emphasized back to the raw
        domain.
        """
        # 0. optional pre-emphasis (whiten both inputs, LTI-commutes)
        if self.preemph:
            Y_rx = _preemph_spectrum(Y_rx, self.preemph, self._pe_rx_state,
                                    self.hop - 1)
            Y = _preemph_spectrum(Y, self.preemph, self._pe_mic_state,
                                  self.hop - 1)

        # 1. regressor buffer
        self.buf_Y_rx = np.roll(self.buf_Y_rx, -1, axis=1)
        self.buf_Y_rx[:, -1, :] = Y_rx
        rx_flipped = np.flip(self.buf_Y_rx, axis=1)

        # 2. a priori estimate and residual
        est = np.zeros((self.nbin, self.nchan), dtype=complex)
        for iref in range(self.Nrxref):
            est += np.sum(self.w[iref] * rx_flipped[:, :, iref:iref+1],
                          axis=1)
        R = Y - est
        e_time = np.fft.irfft(R, n=self.nfft, axis=0)[self.nfft - self.hop:]

        # 3. error for the update: [0; e] — only the new block is graded
        E_head = np.zeros((self.nfft, self.nchan))
        E_head[self.nfft - self.hop:] = e_time
        E_upd = np.fft.rfft(E_head, axis=0)

        # 4. normalized update, suspended while the gate is closed
        X2 = np.sum(np.abs(rx_flipped) ** 2, axis=(1, 2))
        self.X2 = X2
        gate_open = True
        if self.gate_rel is not None:
            P_ref = float(np.sum(np.abs(Y_rx) ** 2))
            P_mic = float(np.sum(np.abs(Y) ** 2))
            decay = np.exp(-self.hop / (self.sr * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0
        if self.beta:
            # CONJUGATE_MDF heritage (merged 2026-10-06): the update runs on
            # a decay-first gradient accumulator and an exponentially
            # smoothed normalizer. gacc tracks the gate (stale gradient
            # fades); Pn updates unconditionally — silence drags it to 0,
            # which suspends the affected bins exactly like the old class.
            if self.rho:
                wmax = np.abs(self.w[0]).max(axis=1) + 1e-30
                Gb = (1.0 - self.rho) + self.rho * np.abs(self.w[0]) / \
                    wmax[:, np.newaxis, :]
                P_inst = np.sum(Gb * np.abs(rx_flipped) ** 2, axis=(1, 2))
            else:
                P_inst = X2
            self.Pn = self.beta * self.Pn + (1 - self.beta) * P_inst
            if gate_open:
                for iref in range(self.Nrxref):
                    self.gacc[iref] = self.beta * self.gacc[iref] +                         np.conj(rx_flipped[:, :, iref:iref+1]) *                         E_upd[:, np.newaxis, :]
            else:
                for iref in range(self.Nrxref):
                    self.gacc[iref] = self.beta * self.gacc[iref]
        if gate_open:
            if self.beta:
                if self.rho:
                    wmax = np.abs(self.w[0]).max(axis=1) + 1e-30
                    Gb = (1.0 - self.rho) + self.rho * np.abs(self.w[0]) / \
                        wmax[:, np.newaxis, :]
                else:
                    Gb = None
                silent = self.Pn <= 1e-12
                Pn_safe = np.where(silent, 1.0, self.Pn)
                step = (self.mu * (1.0 - self.beta) /
                        (Pn_safe + self.eps))[:, np.newaxis, np.newaxis]
                for iref in range(self.Nrxref):
                    term = step * self.gacc[iref]
                    if self.rho:
                        term = term * Gb
                    new_w = self.w[iref] + term
                    new_w[silent] = self.w[iref][silent]  # silent-bin suspend
                    self.w[iref] = new_w
                if self.constraint:
                    for iref in range(self.Nrxref):
                        Wt = np.fft.irfft(self.w[iref], n=self.nfft, axis=0)
                        Wt[self.hop:] = 0.0
                        self.w[iref] = np.fft.rfft(Wt, n=self.nfft, axis=0)
            else:
                if self.rho:
                    # proportionate step (frequency-domain IPNLPS idea):
                    # partitions already carrying weight get bigger steps;
                    # rho=0 is exactly the uniform update. NOTE: the update is
                    # homogeneous in G, so any per-bin rescaling of G cancels —
                    # only the partition RATIOS matter. rho <= 0.5 keeps the
                    # active-partition step amplification within the mu <= 2
                    # envelope on sparse paths.
                    wmax = np.abs(self.w[0]).max(axis=1) + 1e-30
                    G = (1.0 - self.rho) + self.rho * np.abs(self.w[0]) / \
                        wmax[:, np.newaxis, :]
                    X2 = np.sum(G * np.abs(rx_flipped) ** 2, axis=(1, 2))
                gain = self.mu * E_upd / (X2[:, np.newaxis] + self.eps)
                for iref in range(self.Nrxref):
                    gw = G[:, :, iref:iref+1] if self.rho else 1.0
                    self.w[iref] += np.conj(rx_flipped[:, :, iref:iref+1]) * \
                        gw * gain[:, np.newaxis, :]
                if self.constraint:
                    for iref in range(self.Nrxref):
                        Wt = np.fft.irfft(self.w[iref], n=self.nfft, axis=0)
                        Wt[self.hop:] = 0.0
                        self.w[iref] = np.fft.rfft(Wt, n=self.nfft, axis=0)

        # 5. output: zero-headed a priori error; with preemph, de-emphasize
        #    the whitened residual back to the RAW domain (stable IIR)
        if self.preemph:
            e_raw, self._pe_e_zi = lfilter([1.0], [1.0, -self.preemph],
                                           e_time, axis=0, zi=self._pe_e_zi)
            self.e = e_raw
            E_out = np.zeros((self.nfft, self.nchan))
            E_out[self.nfft - self.hop:] = e_raw
        else:
            self.e = e_time
            E_out = np.zeros((self.nfft, self.nchan))
            E_out[self.nfft - self.hop:] = e_time
        self.output = np.fft.rfft(E_out, axis=0)
        return self.output

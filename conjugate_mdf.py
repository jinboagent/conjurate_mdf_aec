"""
Canonical partitioned-block frequency-domain echo cancellers.

CONJUGATE_MDF implements the classic MDF (Lee & Chang, "On the
Convergence Properties of Multidelay Frequency Domain Adaptive Filter";
Soo & Pang 1990; Ferrara 1980):

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

FD_NLMS is the same structure with an instantaneous gradient — the
baseline CONJUGATE_MDF generalizes (beta = gradient averaging; beta = 0
reproduces FD_NLMS exactly).

References: aec/ON THE CONVERGENCE PROPERTIES ...pdf (eqs. 10/12/14/
16-19), aec/freqgradientconjugate.pdf, docs/DATAFLOW.md.
"""

import numpy as np


# Recommended parameters (see DATAFLOW.md §10 benchmarks: beta costs
# average ERLE on speech; use beta > 0 only for mic-noise robustness).
CONJUGATE_MDF_PARAMS = {
    'mu': 1.0,     # NLMS-normalized step (0 < mu <= 2)
    'beta': 0.0,   # gradient averaging factor (0..0.5)
}


class CONJUGATE_MDF:
    """
    Conjugate Gradient MDF echo canceller (single configuration; run
    several instances and take the minimum-power output for multi-length
    operation).

    Parameters
    ----------
    NCHAN : microphone channels
    NBIN : frequency bins of the frame spectra (frame length L = 2*(NBIN-1))
    N_G : filter length in frames; delay coverage = (N_G + R - 2)*hop
          samples at R = L/hop overlap
    hop : block size M — new samples per frame. Frames advance by `hop`
          and the wrapper keeps the LAST `hop` samples of irfft(output).
    mu : step size
    beta : gradient averaging (0 = instantaneous, = FD_NLMS; 0.05..0.3
           trades speed for smoothing under mic noise)
    bin_skip : freeze the lowest bins at zero weight (normally removed by
               a downstream high-pass filter)
    gate_rel / gate_hold / gate_decay / gate_floor : reference-excitation
        gate. Adaptation runs only while
            P_ref > gate_rel * P_mic    and   P_ref > gate_floor * P_run
        (P_run = running peak of P_ref, decay `gate_decay` seconds),
        held open for `gate_hold` frames. gate_rel = None disables.
    """

    __slots__ = ['nchan', 'nbin', 'N_G', 'Nrxref', 'hop', 'mu', 'beta',
                 'bin_skip', 'gate_rel', 'gate_hold', 'gate_decay',
                 'gate_floor', '_Prun', '_hold',
                 'buf_Y_rx', 'gacc', 'Pn', 'w', 'w_last', 'output', 'e']

    def __init__(self, NCHAN, NBIN, N_G, hop, mu=1.0, beta=0.0,
                 bin_skip=0, Nrxref=1, gate_rel=0.3, gate_hold=2,
                 gate_decay=1.5, gate_floor=1e-6):
        self.nchan = NCHAN
        self.nbin = NBIN
        self.N_G = int(N_G)
        self.Nrxref = Nrxref
        self.hop = int(hop)
        if not 0 < self.hop <= 2 * (NBIN - 1) // 2:
            raise ValueError("hop must be in (0, nfft/2]")
        self.mu = float(mu)
        self.beta = float(beta)
        self.bin_skip = int(bin_skip)
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        self.reset()

    def reset(self):
        nbin, N_G = self.nbin, self.N_G
        self.buf_Y_rx = np.zeros((nbin, N_G, self.Nrxref), dtype=complex)
        # Averaged [0;e] gradient: gacc = beta*gacc + inst (decay-first,
        # so beta = 0 is the pure instantaneous gradient).
        self.gacc = [np.zeros((nbin, N_G, self.nchan), dtype=complex)
                     for _ in range(self.Nrxref)]
        # Smoothed per-bin reference power (sum over partitions), the
        # normalizer scale.
        self.Pn = np.zeros(nbin)
        self.w = [np.zeros((nbin, N_G, self.nchan), dtype=complex)
                  for _ in range(self.Nrxref)]
        self.w_last = [w_.copy() for w_ in self.w]
        for w_ in self.w + self.w_last:
            w_[:self.bin_skip] = 0.0
        self.output = np.zeros((nbin, self.nchan), dtype=complex)
        self.e = np.zeros(self.hop)
        self._Prun = 0.0
        self._hold = 0

    def apply(self, Y, Y_rx):
        """
        Process one frame: Y [NBIN, NCHAN] mic spectrum, Y_rx [NBIN, Nrxref]
        reference spectrum (both FULL L-sample frame FFTs). Returns the
        zero-headed a priori error spectrum rfft([0; e]) — identical frame
        geometry to FD_NLMS; the wrapper keeps the last `hop` samples of
        irfft(output).
        """
        nbin, nchan = self.nbin, self.nchan

        # 1. regressor buffer: index j = frame lag j (newest at -1)
        self.buf_Y_rx = np.roll(self.buf_Y_rx, -1, axis=1)
        self.buf_Y_rx[:, -1, :] = Y_rx
        rx_flipped = np.flip(self.buf_Y_rx, axis=1)

        # 2. a priori estimate and residual
        est = np.zeros((nbin, nchan), dtype=complex)
        for iref in range(self.Nrxref):
            est += np.sum(self.w_last[iref] * rx_flipped[:, :, iref:iref+1],
                          axis=1)
        nfft = 2 * (nbin - 1)
        e_tail = np.fft.irfft(Y - est, n=nfft, axis=0)[nfft - self.hop:]
        E_head = np.zeros((nfft, nchan))
        E_head[nfft - self.hop:] = e_tail
        E_zh = np.fft.rfft(E_head, axis=0)          # [0; e]

        # 3. excitation gate: adapt only while the reference is
        #    identifiable. A pure level threshold cannot separate quiet
        #    speech (adapt) from a reverberant pause (freeze) — the
        #    reference/microphone ratio can, and freezing on
        #    P_ref <= gate_rel*P_mic also covers double talk.
        gate_open = True
        if self.gate_rel is not None:
            P_ref = float(np.sum(np.abs(Y_rx) ** 2))
            P_mic = float(np.sum(np.abs(Y) ** 2))
            decay = np.exp(-self.hop / (16000.0 * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0

        # 4. statistics: the exact [0;e] gradient (the time-gating
        #    operator is linear, so accumulating conj(X)*E_zh needs no
        #    bin-decoupling approximation) and the normalizer power.
        beta = self.beta
        for iref in range(self.Nrxref):
            new_acc = np.conj(rx_flipped[:, :, iref:iref+1]) * E_zh[:, np.newaxis, :]
            if gate_open:
                self.gacc[iref] = beta * self.gacc[iref] + new_acc
            else:
                self.gacc[iref] *= beta      # let the stale gradient fade
            self.Pn = beta * self.Pn + (1 - beta) * np.sum(
                np.abs(rx_flipped[:, :, iref]) ** 2, axis=1)

        # 5. normalized update + G = [I_hop, 0] projection (the canonical
        #    MDF weight definition: each partition is an `hop`-tap causal
        #    filter, so the frequency products stay exact linear
        #    convolutions).
        if gate_open:
            g_scale = 1.0 - beta             # gacc -> exponential average
            for iref in range(self.Nrxref):
                for ibin in range(self.bin_skip, nbin):
                    if self.Pn[ibin] <= 1e-12:      # silent bin: suspend
                        self.w[iref][ibin] = self.w_last[iref][ibin]
                        continue
                    self.w[iref][ibin] = self.w_last[iref][ibin] + \
                        (self.mu * g_scale / (self.Pn[ibin] + 1e-10)) * \
                        self.gacc[iref][ibin]
                Wt = np.fft.irfft(self.w[iref], n=nfft, axis=0)
                Wt[self.hop:] = 0.0
                self.w[iref] = np.fft.rfft(Wt, n=nfft, axis=0)
        else:
            for iref in range(self.Nrxref):
                self.w[iref] = self.w_last[iref].copy()

        # 6. output: zero-headed a priori error (same frame geometry as
        #    FD_NLMS — the classes are interchangeable end-to-end), and
        #    advance the filter state.
        self.output = E_zh.copy()
        self.e = e_tail
        for iref in range(self.Nrxref):
            self.w_last[iref] = self.w[iref].copy()
        return self.output


class FD_NLMS:
    """
    Partitioned-block frequency-domain NLMS (classic overlap-save FDAF) —
    the instantaneous-gradient baseline of CONJUGATE_MDF (identical input
    and output frame geometry; CONJUGATE_MDF with beta = 0 reproduces it
    exactly).

        e   = last `hop` samples of irfft(Y - est)          ([0; e])
        X2  = sum_j |X(t-j)|^2                              per bin
        w_j += mu * conj(X(t-j)) * E / (X2 + eps)
        w_j = FFT[first-hop-taps; 0]                        (if constraint)

    full_frame_error=True reproduces the historical head-scoring
    criterion inside this same class — the controlled ablation that
    demonstrates the criterion cliff (docs/DATAFLOW.md §7).
    """

    __slots__ = ['nchan', 'nbin', 'N_G', 'Nrxref', 'mu', 'hop', 'nfft',
                 'constraint', 'full_frame_error', 'eps',
                 'gate_rel', 'gate_hold', 'gate_decay', 'gate_floor',
                 '_Prun', '_hold',
                 'buf_Y_rx', 'w', 'P_X', 'output', 'e', 'X2']

    def __init__(self, NCHAN, NBIN, N_G, mu, hop,
                 Nrxref=1, constraint=True, full_frame_error=False,
                 eps=1e-10, gate_rel=0.3, gate_hold=2, gate_decay=1.5,
                 gate_floor=1e-6):
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
        self.full_frame_error = bool(full_frame_error)
        self.eps = float(eps)
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        self.reset()

    def reset(self):
        self.buf_Y_rx = np.zeros((self.nbin, self.N_G, self.Nrxref),
                                 dtype=complex)
        self.w = [np.zeros((self.nbin, self.N_G, self.nchan), dtype=complex)
                  for _ in range(self.Nrxref)]
        self.output = np.zeros((self.nbin, self.nchan), dtype=complex)
        self.e = np.zeros(self.hop, dtype=float)
        self.X2 = np.zeros(self.nbin)
        self._Prun = 0.0
        self._hold = 0

    def apply(self, Y, Y_rx):
        """
        Process one frame. Y [NBIN, NCHAN] mic spectrum, Y_rx [NBIN, Nrxref]
        reference spectrum (both FULL L-sample frame FFTs). Returns the
        zero-headed a priori error spectrum rfft([0; e]).
        """
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

        # 3. error for the update: [0; e], or the full-frame ablation
        if self.full_frame_error:
            E_upd = R
        else:
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
            decay = np.exp(-self.hop / (16000.0 * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0
        if gate_open:
            gain = self.mu * E_upd / (X2[:, np.newaxis] + self.eps)
            for iref in range(self.Nrxref):
                self.w[iref] += np.conj(rx_flipped[:, :, iref:iref+1]) * \
                    gain[:, np.newaxis, :]
            if self.constraint:
                for iref in range(self.Nrxref):
                    Wt = np.fft.irfft(self.w[iref], n=self.nfft, axis=0)
                    Wt[self.hop:] = 0.0
                    self.w[iref] = np.fft.rfft(Wt, n=self.nfft, axis=0)

        # 5. output: zero-headed a priori error
        self.e = e_time
        E_out = np.zeros((self.nfft, self.nchan))
        E_out[self.nfft - self.hop:] = e_time
        self.output = np.fft.rfft(E_out, axis=0)
        return self.output

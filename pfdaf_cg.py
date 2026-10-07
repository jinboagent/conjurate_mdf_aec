"""
Partitioned-block frequency-domain adaptive filter, Conjugate-Gradient
update (PBFDAF-CG) — faithful single-channel implementation of:

    L. García Morales, J. A. Beracoechea, S. Torres-Guijarro and
    F. J. Casajús-Quirós, "Conjugate Gradient Techniques for
    Multichannel Acoustic Echo Cancellation in Frequency Domain",
    AES Convention Paper 6713, 120th Convention, Paris, 2006.

Paper structure -> this module:
    eq. (2)-(6)   overlap-save, [0;e] error, a priori output   (as in
                  FD_NLMS.py — same frame contract)
    eq. (8)+(9)   instantaneous gradient conj(X)·E, constrained
    eq. (18)      Φ: gradient estimate AVERAGED over a sliding memory
                  (exponential average `gamma`; the averaged gradient of
                  the [0;e] criterion — exact, the gate is linear)
    eq. (11)-(17) CG machinery: v = -g + β·v_prev, w += α·v,
                  β from Hestenes-Stiefel / Fletcher-Reeves /
                  Polak-Ribière / Dai-Yuan (eqs. 17,19,20,21),
                  clipped: β > 1 -> 1 (paper's stability rule), β < 0 -> 0
                  (restart)
    α (eq. 14)    the paper's no-line-search step (Boray & Srinath 1992,
                  ref. [5]) is unreadable in the PDF; we use the exact
                  line search on the per-bin quadratic model with the
                  SAME memory-averaged Gram matrix
                  R[p,q] = avg conj(X(t-p))·X(t-q)  (eq. 10's R):
                      α = -(g^H v) / (v^H R v)
                  (per-bin approximation of the bin-coupled [0;e]
                  Hessian — used only to scale steps, so the fixed
                  point stays the exact [0;e] optimum; see
                  docs/DATAFLOW.md §7 for the coupling proof).
    inner k_max   per frame, min(N_G, hop) CG iterations on the fixed
                  averaged model (gradient carried by the quadratic
                  update g <- g - α·R v, exact for the model).

The gradient-memory average (Φ) at k_max=1, gamma=0 IS exactly
FD_NLMS's instantaneous steepest descent; gamma>0 + conjugate
directions is the paper's speedup.

PFDAF_CG is the spectra-driven core (same apply(Y, Y_rx) contract as
CONJUGATE_MDF / FD_NLMS — drop-in for the same wrapper); PFDAFCG
keeps a time-block filt/update API for direct experimentation.
"""

import numpy as np


class PFDAF_CG:
    """
    Conjugate-gradient PBFDAF (García Morales et al. 2006), single mic
    channel, Nrxref reference channels.

    Parameters
    ----------
    NCHAN : microphone channels
    NBIN : frequency bins of the frame spectra (frame length L = 2*(NBIN-1))
    N_G : filter length in frames (paper's Q)
    hop : block size K — new samples per frame (paper: FFT 2K, hop K;
          any L = R*hop geometry works, same as the sibling classes)
    gamma : gradient-memory averaging factor (paper's N-block average,
            exponential form; 0 = instantaneous steepest descent = FD_NLMS)
    k_max : CG iterations per frame (paper: min(N, K), MSE-stopped)
    beta_method : 'hestenes-stiefel' (paper default) | 'fletcher-reeves'
                  | 'polak-ribiere' | 'dai-yuan'
    constrain : 'full' (project every partition every frame — paper's
                constrained version), 'partial' (one partition per frame,
                rotating), or 'none'
    delta : relative Tikhonov loading of the line-search denominator
            (paper's stability constant, eq. 22 spirit): the step along v
            is α = -(g^H v)/(v^H R v + delta·P_b·‖v‖²), P_b = per-bin
            averaged regressor power (trace of R). Keeps α bounded when
            R is rank-deficient (v in a near-null direction).
    alpha_max : safety cap on the CG step in units of the (equally
                regularized) steepest-descent step
    gate_rel / gate_hold / gate_decay / gate_floor : reference-excitation
        gate (project addition, not in the paper; identical to the
        sibling classes — freeze while P_ref <= gate_rel*P_mic or
        below the running-peak floor). gate_rel=None disables.
    """

    __slots__ = ['nchan', 'nbin', 'N_G', 'Nrxref', 'hop', 'gamma', 'k_max',
                 'beta_method', 'constrain', 'alpha_max', 'delta', 'nfft',
                 'gate_rel', 'gate_hold', 'gate_decay', 'gate_floor', 'sr',
                 '_Prun', '_hold', '_partial_index',
                 'buf_Y_rx', 'gacc', 'R', 'v_prev', 'g_prev_cg',
                 'w', 'output', 'e']

    def __init__(self, NCHAN, NBIN, N_G, hop, gamma=0.25, k_max=1,
                 beta_method='hestenes-stiefel', constrain='full',
                 alpha_max=2.0, delta=0.5, Nrxref=1, gate_rel=0.3,
                 gate_hold=2, gate_decay=1.5, gate_floor=1e-6,
                 sr=16000.0):
        self.nchan = NCHAN
        self.nbin = NBIN
        self.N_G = int(N_G)
        self.Nrxref = Nrxref
        self.hop = int(hop)
        self.nfft = 2 * (NBIN - 1)
        if not 0 < self.hop <= self.nfft // 2:
            raise ValueError("hop must be in (0, nfft/2]")
        self.gamma = float(gamma)
        self.k_max = max(1, int(k_max))
        self.beta_method = beta_method
        self.constrain = constrain
        self.alpha_max = float(alpha_max)
        self.delta = float(delta)
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        self.sr = float(sr)
        self.reset()

    def reset(self):
        nbin, N_G = self.nbin, self.N_G
        self.buf_Y_rx = np.zeros((nbin, N_G, self.Nrxref), dtype=complex)
        # Φ (eq. 18): memory-averaged [0;e] gradient, per reference
        self.gacc = [np.zeros((nbin, N_G, self.nchan), dtype=complex)
                     for _ in range(self.Nrxref)]
        # per-bin averaged Gram of the partition regressors (eq. 10):
        # R[b][p, q] = avg conj(X_{t-p}[b]) X_{t-q}[b]
        self.R = [np.zeros((nbin, N_G, N_G), dtype=complex)
                  for _ in range(self.Nrxref)]
        # CG direction & previous (inner-iteration) gradient, per reference
        self.v_prev = [np.zeros((nbin, N_G, self.nchan), dtype=complex)
                       for _ in range(self.Nrxref)]
        self.g_prev_cg = [np.zeros((nbin, N_G, self.nchan), dtype=complex)
                          for _ in range(self.Nrxref)]
        self.w = [np.zeros((nbin, N_G, self.nchan), dtype=complex)
                  for _ in range(self.Nrxref)]
        self.output = np.zeros((nbin, self.nchan), dtype=complex)
        self.e = np.zeros(self.hop)
        self._Prun = 0.0
        self._hold = 0
        self._partial_index = 0

    # ------------------------------------------------------------------
    def apply(self, Y, Y_rx):
        """
        Process one frame: Y [NBIN, NCHAN] mic spectrum, Y_rx [NBIN, Nrxref]
        reference spectrum (both FULL L-sample frame FFTs). Returns the
        zero-headed a priori error spectrum rfft([0; e]) — identical frame
        geometry to FD_NLMS / CONJUGATE_MDF.
        """
        nbin, nchan, nfft = self.nbin, self.nchan, self.nfft
        hop, gam = self.hop, self.gamma

        # 1. regressor buffer (index j = frame lag, newest at -1)
        self.buf_Y_rx = np.roll(self.buf_Y_rx, -1, axis=1)
        self.buf_Y_rx[:, -1, :] = Y_rx
        rx_flipped = np.flip(self.buf_Y_rx, axis=1)     # oldest first

        # 2. a priori estimate + [0;e] error (eq. 4-6)
        est = np.zeros((nbin, nchan), dtype=complex)
        for iref in range(self.Nrxref):
            est += np.sum(self.w[iref] * rx_flipped[:, :, iref:iref+1],
                          axis=1)
        e_tail = np.fft.irfft(Y - est, n=nfft, axis=0)[nfft - hop:]
        E_head = np.zeros((nfft, nchan))
        E_head[nfft - hop:] = e_tail
        E_zh = np.fft.rfft(E_head, axis=0)

        # 3. excitation gate (project convention; freezes adaptation)
        gate_open = True
        if self.gate_rel is not None:
            P_ref = float(np.sum(np.abs(Y_rx) ** 2))
            P_mic = float(np.sum(np.abs(Y) ** 2))
            decay = np.exp(-hop / (self.sr * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0

        # 4. Φ (eq. 18): average the instantaneous [0;e] gradient
        #    G = conj(X)·E (eq. 8) and the Gram R (eq. 10) over the
        #    memory (exponential form of the paper's N-block sum).
        for iref in range(self.Nrxref):
            rx = rx_flipped[:, :, iref]                          # [nbin, N_G]
            g_inst = np.conj(rx[:, :, np.newaxis]) * E_zh[:, np.newaxis, :]
            if gate_open:
                self.gacc[iref] = gam * self.gacc[iref] + (1 - gam) * g_inst
            else:
                self.gacc[iref] *= gam            # let it fade, no new data
                # restart the CG machinery after a freeze: conjugating the
                # fresh gradient against a stale direction is unsafe — the
                # zeroed direction makes the next open frame take a plain
                # steepest-descent step
                self.v_prev[iref][:] = 0.0
                self.g_prev_cg[iref][:] = 0.0
            R_inst = np.einsum('bp,bq->bpq', np.conj(rx), rx)
            self.R[iref] = gam * self.R[iref] + (1 - gam) * R_inst

        # 5. CG iterations on the per-bin quadratic model (eqs. 11-17).
        #    g carries across the inner loop by the exact model update
        #    g <- g - alpha·R v; the averaged gacc seeds it each frame.
        if gate_open:
            for iref in range(self.Nrxref):
                g = self.gacc[iref].copy()
                v = self.v_prev[iref]
                g_prev = self.g_prev_cg[iref]
                R = self.R[iref]
                for _ in range(self.k_max):
                    # search direction (eq. 16, descent convention): gacc
                    # is the DESCENT direction (= -G of the paper's
                    # eq. 8), so v = g + beta*v_prev; first step or
                    # restart: plain steepest descent
                    if not np.any(v):
                        v_dir = g
                    else:
                        beta = self._beta(g, g_prev, v)
                        v_dir = g + beta * v
                    # constrained version (eq. 9): the direction is a
                    # valid partitioned filter
                    v_dir = self._project(v_dir)
                    # regularized exact line search (see `delta`): the
                    # loading keeps alpha bounded when v lies in a
                    # near-null direction of the (early, rank-deficient)
                    # per-bin Gram
                    P_b = np.maximum(
                        np.trace(R, axis1=1, axis2=2).real / self.N_G,
                        1e-30)                                   # [nbin]
                    nv2 = np.sum(np.abs(v_dir) ** 2, axis=(1, 2))
                    ng2 = np.sum(np.abs(g) ** 2, axis=(1, 2))
                    Rv = np.einsum('bpq,bqc->bpc', R, v_dir)
                    Rg = np.einsum('bpq,bqc->bpc', R, g)
                    den = np.real(np.sum(np.conj(v_dir) * Rv, axis=(1, 2))) \
                        + self.delta * P_b * nv2
                    num = np.real(np.sum(np.conj(g) * v_dir, axis=(1, 2)))
                    alpha = np.where(den > 0, num / np.maximum(den, 1e-30),
                                     0.0)
                    sd = ng2 / (np.real(np.sum(np.conj(g) * Rg,
                                               axis=(1, 2)))
                                + self.delta * P_b * ng2 + 1e-30)
                    alpha = np.clip(alpha, 0.0, self.alpha_max * sd)
                    self.w[iref] += alpha[:, np.newaxis, np.newaxis] * v_dir
                    # model gradient at the new point
                    g_prev = g
                    g = g - alpha[:, np.newaxis, np.newaxis] * Rv
                    v = v_dir
                self.v_prev[iref] = v
                self.g_prev_cg[iref] = g_prev
                if self.constrain == 'full':
                    self.w[iref] = self._project(self.w[iref])
                elif self.constrain == 'partial':
                    p = self._partial_index
                    self.w[iref][:, p, :] = self._project(
                        self.w[iref][:, p: p + 1, :])[:, 0, :]
        self._partial_index = (getattr(self, '_partial_index', 0) + 1) \
            % self.N_G if self.constrain == 'partial' else 0

        # 6. output: zero-headed a priori error (same geometry as FD_NLMS)
        self.output = E_zh.copy()
        self.e = e_tail
        return self.output

    # ------------------------------------------------------------------
    def _beta(self, g, g_prev, v_prev):
        """β per bin (eqs. 17/19/20/21), with the paper's stability
        clip (β > 1 -> 1) and a restart floor (β < 0 -> 0)."""
        gH = np.sum(np.conj(g) * g, axis=(1, 2)).real
        if self.beta_method == 'fletcher-reeves':
            num, den = gH, np.sum(np.conj(g_prev) * g_prev,
                                  axis=(1, 2)).real
        else:
            gd = g - g_prev
            gdH_g = np.sum(np.conj(gd) * g, axis=(1, 2)).real
            if self.beta_method in ('polak-ribiere',):
                num, den = gdH_g, np.sum(np.conj(g_prev) * g_prev,
                                         axis=(1, 2)).real
            elif self.beta_method == 'dai-yuan':
                num, den = gH, np.sum(np.conj(v_prev) * gd, axis=(1, 2)).real
            else:                                   # hestenes-stiefel
                num = gdH_g
                den = np.sum(np.conj(v_prev) * gd, axis=(1, 2)).real
        beta = np.where(den > 1e-30, num / np.maximum(den, 1e-30), 0.0)
        return np.clip(beta, 0.0, 1.0)[:, np.newaxis, np.newaxis]

    def _project(self, W):
        """G = [I_hop, 0] projection (eq. 9): each partition weight stays
        the spectrum of a hop-tap causal filter."""
        Wt = np.fft.irfft(W, n=self.nfft, axis=0)
        Wt[self.hop:] = 0.0
        return np.fft.rfft(Wt, n=self.nfft, axis=0)


class PFDAFCG:
    """Time-block wrapper around PFDAF_CG (paper's K-block API:
    filt(x_block, d_block) -> e_block, update(e, d))."""

    def __init__(self, N, winlen, mu=None, nchan=1, k_max=1,
                 beta_method='hestenes-stiefel', constrain='full',
                 gamma=0.25, gate_rel=0.3):
        # mu kept for API compatibility; the CG step is line-searched
        self.M = int(winlen)
        self.N = int(N)
        self.core = PFDAF_CG(NCHAN=nchan, NBIN=winlen + 1, N_G=N,
                             hop=winlen, gamma=gamma, k_max=k_max,
                             beta_method=beta_method, constrain=constrain,
                             Nrxref=1, gate_rel=gate_rel)
        self.x_old = np.zeros(winlen)
        self.d_old = np.zeros(winlen)

    def filt(self, x, d):
        assert len(x) == self.M
        e = self.process_block(x, d)
        return e

    def process_block(self, x, d):
        """One K-block: filter + update; returns the error block."""
        x_now = np.concatenate((self.x_old, x))
        d_now = np.concatenate((self.d_old, d))
        self.x_old = x.copy()
        self.d_old = d.copy()
        E = self.core.apply(np.fft.rfft(d_now).reshape(-1, 1),
                            np.fft.rfft(x_now).reshape(-1, 1))
        return np.fft.irfft(E[:, 0], n=2 * self.M)[self.M:]

    def update(self, e, d=None):
        """Compatibility no-op: the CG update already ran in filt()."""
        pass


def pfdaf_cg(x, d, N=4, M=64, mu=None, nchan=1, k_max=1,
             beta_method='hestenes-stiefel', constrain='full', gamma=0.25):
    """Block-driver convenience function (returns the residual stream)."""
    ft = PFDAFCG(N, M, mu, nchan, k_max, beta_method, constrain, gamma)
    num_block = min(len(x), len(d)) // M
    e = np.zeros(num_block * M)
    for n in range(num_block):
        e[n * M:(n + 1) * M] = ft.process_block(
            x[n * M:(n + 1) * M], d[n * M:(n + 1) * M])
    return e

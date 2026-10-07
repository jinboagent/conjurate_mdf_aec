"""
CONJUGATE_FB_TOEPLITZ — the Toeplitz/CG update running inside a WOLA
filterbank: route 3 of "Toeplitz without the cliff".

Why this should be cliff-free
-----------------------------
The legacy full-frame engine (conjugate_toeplitz.py, deleted 2026-10-06;
code recoverable from git 2f23d9d, then inside the module now named
FD_NLMS.py) cliffed because
its criterion
was the FULL-frame overlap-save LS: the head of every frame is circular
wrap for sub-hop delays, and the full-frame Wiener solve converges to
the optimum of that WRONG criterion (canonical 5.94 dB vs 23.77 for
[0;e]). The Toeplitz structure itself was never the problem — it is
BOUND to the full-frame criterion because full-frame lag correlations
are only Toeplitz under windowed-frame stationarity.

In a WOLA filterbank there is no head at all: the
error is the per-bin subband difference
    E_b[m] = D_b[m] - sum_j w_b[j] X_b[m-j],
a clean LS over decimated subband samples. Under subband wide-sense
stationarity the normal matrix of THAT criterion is Toeplitz in the
subband lag j — so the legacy correlation machinery (autoR/rcross +
one CG step on toeplitz(autoR) w = rcross) is the legitimate
second-order solver of the head-free criterion:

    autoR_b  = beta*(autoR_b  + alpha*conj(X_b[m-j]) X_b[m])   subband lags
    rcross_b = beta*(rcross_b + alpha*conj(X_b[m-j]) D_b[m])
    g0     = rcross - toeplitz(autoR) @ w          normal-eq residual
    alf    = 0.999 <g0,g0>/<g0, T g0>              (+ the 1e-30 guard)
    w      = w + alf * g0                          one CG step per block

What can still floor it (measured, self-test + results/2026-10-06_*
diagnostics): the filterbank REPRESENTATION (not criterion) — the
analysis window breaks shift invariance, so a pure delay is exactly
representable only on the subband grid (d = 0 mod hop). With beta = 1
the ON-GRID case has NO floor: ERLE deepens at the clean 1/N
correlation-estimate rate (+2.9 dB per data doubling: 20.1 dB at 2 s ->
25.9 dB at 8 s, white noise). OFF-GRID the slope decays (34% off-grid
+1.2, 50% off-grid +0.7 dB/doubling) toward a representation floor
(~16-17 dB at n_g=8) — but that is a soft slope, not the legacy
criterion cliff (legacy: aligned 18.8 -> sub-hop 5.5 dB, a -13 dB
collapse; here aligned 14.4 -> 50% off-grid 11.9, -2.5 dB on the same
2 s budget). The correlation-window noise bounds STEADY depth on
short data (beta sweep: 0.9 -> 7.2, 0.97 -> 14.4, 0.999 -> 20.1 dB);
more partitions HURT (estimation dimensions grow: n_g 4/8/16/32 ->
14.7/11.2/7.0/2.8 dB) — take the smallest n_g covering the path. The
legacy 'sd' solver step is measured-best (PR neutral, FR diverges).

Skeleton/prototype/driver geometry mirror the WOLA-NLMS sibling
(conjugate_full.py, removed 2026-10-06; its _sqrt_hann is vendored
below); _herm_toeplitz_matvec (verified bit-exact vs scipy when the
legacy full-frame engine still existed) is vendored below too.

API is TIME-BLOCK:
    aec = CONJUGATE_FB_TOEPLITZ()             # nfft=512, hop=256
    e_block = aec.process(x_block, d_block)   # hop in, hop out
Driver: fb_toeplitz_aec(x, d) -> residual stream.
"""

import numpy as np

__all__ = ['CONJUGATE_FB_TOEPLITZ', 'fb_toeplitz_aec',
           'FB_TOEPLITZ_PARAMS_BALANCED']


def _sqrt_hann(n):
    """Periodic sqrt-Hann prototype: q^2 = Hann satisfies COLA at
    hop = n/k for any integer k >= 2."""
    return np.sqrt(np.hanning(n + 1)[:-1])


def _herm_toeplitz_matvec(r, v):
    """
    Exact vectorized (toeplitz(r) @ v) under scipy's Hermitian convention:
    T[i,j] = r[i-j] for i >= j, T[i,j] = conj(r[j-i]) for j > i.

    r: [..., N] correlation vectors; v: [..., N, K] weight stacks. The
    last axis of r indexes the lag; the last axis of v is contracted.
    """
    n = r.shape[-1]
    out = np.zeros_like(v)
    for i in range(n):
        oi = out[..., i, :]
        for j in range(i + 1):
            oi += r[..., i - j, None] * v[..., j, :]
        for j in range(i + 1, n):
            oi += np.conj(r[..., j - i, None]) * v[..., j, :]
    return out


def _tmatvec(r, v):
    """_herm_toeplitz_matvec for channel-less 2-D weights [nbin, n_g]."""
    return _herm_toeplitz_matvec(r, v[:, :, np.newaxis])[:, :, 0]

# alpha cancels in the solve (both correlations share the window); beta is
# the real knob: effective correlation window ~1/(1-beta) blocks.
FB_TOEPLITZ_PARAMS_BALANCED = {'alpha': 0.05, 'beta': 0.97}


class CONJUGATE_FB_TOEPLITZ:
    """
    WOLA filterbank subband AEC with the legacy Toeplitz-CG solver
    (single channel, single reference).

    Parameters
    ----------
    nfft, hop : analysis geometry (hop = nfft/2 canonical sqrt-Hann
        COLA). Subband grid = hop: a delay d is exactly representable
        when d = 0 mod hop; n_g partitions cover n_g*hop samples.
    n_g : subband partitions (filter taps per bin at the subband rate).
    alpha, beta : correlation accumulation scale / forgetting factor —
        the legacy presets. beta trades correlation-estimate noise
        (large beta -> longer window -> deeper steady ERLE) against
        tracking of nonstationary speech.
    constraint : across-bin G=[I_hop,0] projection after the solve (the
        conjugate_full convention). Default False: the per-bin solve is
        already the subband optimum and the projection can fight it.
    gate_rel / gate_hold / gate_decay / gate_floor / sr : the shared
        reference-excitation gate (accumulation and solve both freeze
        on closed frames; correlations keep decaying so stale
        statistics fade).
    """

    __slots__ = ['nfft', 'hop', 'n_g', 'alpha', 'beta', 'beta_method',
                 'delta', 'k_max', 'reset_every', 'gradient', 'gamma', 'internal',
                 'eps_rel', 'constraint', 'gate_rel', 'gate_hold',
                 'gate_decay', 'gate_floor', 'sr', 'q', '_syn', '_Prun',
                 '_hold', '_frame', 'solver', 'lam',
                 'x_hist', 'd_hist', 'acc', 'buf_X', 'w',
                 'autoR', 'rcross', 'phi', 'v', 'g_prev', '_cg_live',
                 'P', 'e']

    def __init__(self, nfft=512, hop=256, n_g=8, alpha=0.05, beta=0.97,
                 beta_method='sd', delta=0.1, k_max=1, reset_every=None,
                 gradient='correlation', gamma=0.25, internal='model',
                 constraint=False, gate_rel=0.3, gate_hold=2,
                 gate_decay=1.5, gate_floor=1e-6, sr=16000.0,
                 solver='cg', lam=0.999):
        if nfft % hop or hop * 2 > nfft:
            raise ValueError("hop must divide nfft and nfft >= 2*hop")
        self.nfft = int(nfft)
        self.hop = int(hop)
        self.n_g = int(n_g)
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Gradient source. 'correlation': g = rcross - T w (the paper's
        # CG1 residual on the windowed normal equations). 'error': the
        # gamma-averaged INSTANTANEOUS error gradient (PAES 2006 form,
        # phi = gamma*phi + (1-gamma)*conj(X(m-j)) E(m)) — unbiased at
        # the true path for ANY delay (the windowed rcross carries the
        # +-lag inconsistency and the beta^-k shift bias; the error
        # gradient has neither), with T kept ONLY as the line-search
        # curvature. Measured: CG2 on 'correlation' converges into the
        # windowed system's noise (delay probes get WORSE with k_max
        # while delay-0 reaches 326 dB) — the noise is in the system,
        # not the solver; 'error' removes it from the fixed point.
        self.gradient = str(gradient).lower()
        if self.gradient not in ('correlation', 'error'):
            raise ValueError(f"unknown gradient {gradient!r}")
        self.gamma = float(gamma)
        # relative normalizer floor for the error-gradient curvature
        # (the WOLA-NLMS sibling's eps_rel)
        self.eps_rel = 0.1
        # Relative curvature floor for the line search (the eps_rel trick
        # from conjugate_full, same role as PBFDAF's regularized alpha):
        # den = <v,Tv> + delta*Pbar*||v||^2 with Pbar = frame-mean zero-lag
        # subband power. At 4x oversampling the adjacent-lag correlation is
        # ~0.75 and the per-bin T is near-singular in spectral holes — the
        # unregularized Rayleigh step exploded there (|w| -> 1e11 on the
        # canonical pair). delta scales with the AVERAGE power, so quiet
        # bins are damped relative to the frame level (a per-bin floor
        # would vanish exactly where it is needed). Fixed point unchanged
        # (g = 0 at the solution); only oversized steps are capped.
        self.delta = float(delta)
        # Direction memory for the per-bin solve, Chang & Willson (TSP
        # 2000) CG1/CG2 forms with the complex Hermitian inner products
        # verified in pfdaf_cg.py. 'sd' = single steepest-descent
        # Rayleigh step (p = g, the legacy engine's step). 'fr'/'pr'/
        # 'dy'/'hs' = conjugate directions (FR (6); PR = paper eq. (53),
        # numerator <g-g_prev, g> over ||g_prev||^2; Dai-Yuan;
        # Hestenes-Stiefel), all clipped to [0, 1] as in pfdaf_cg.
        # Measured hazards: 'fr' diverges in this noisy per-bin setting;
        # at 4x oversampling 'pr' with a short correlation window
        # (beta <= 0.97) diverged before the periodic reset existed.
        self.beta_method = str(beta_method).lower()
        if self.beta_method not in ('sd', 'fr', 'pr', 'dy', 'hs'):
            raise ValueError(f"unknown beta_method {beta_method!r}")
        # CG1 vs CG2 (paper section III): k_max = 1 is CG1 — ONE CG step
        # per frame with cross-frame direction memory (the "degenerated"
        # scheme; the paper recommends periodically resetting the
        # direction to the fresh gradient — reset_every, in frames).
        # k_max > 1 is CG2 — the correlations are FROZEN for k internal
        # iterations (direction restarts from the fresh gradient each
        # frame, residual updated by the recursion g <- g - a*T*v), the
        # paper's eigenvalue-spread-independent mode.
        self.k_max = max(1, int(k_max))
        self.reset_every = None if reset_every is None else max(1, int(reset_every))
        # CG2 internal-iteration gradient source. 'model': the paper's
        # frozen-R recursion g <- g - a*T*v (valid when T IS the Hessian
        # of the minimized cost — correlation mode at beta=1). 'true':
        # recompute the instantaneous error gradient from the UPDATED w
        # each internal iteration (error mode needs this — the frozen-T
        # model recursion diverges there, k=4: -342 dB at 2x, because
        # the error surface's curvature is the per-frame Gram, not T).
        self.internal = str(internal).lower()
        if self.internal not in ('model', 'true'):
            raise ValueError(f"unknown internal {internal!r}")
        # Solver. 'cg': the Chang & Willson gradient/line-search block
        # below. 'rls': per-bin exponentially-windowed recursive least
        # squares (Haykin ch. 9) — P = R^-1 of the TRUE per-bin Gram
        # maintained by Sherman-Morrison (no explicit inversion), gain
        # K = P conj(u) / (lam + u^H P u), update w += K xi (xi = a-priori
        # subband error). At n_g = 8 the O(n_g^2) cost that historically
        # made CG preferable is 128 complex MACs per bin — negligible.
        # Measured (2026-10-06, scratch_fb_toeplitz_rls*/dissect2/_
        # profile): RLS beats the CG champion by 11-14 dB on every speech
        # pair (canonical 43.6-44.7 vs 29.8). DISSECTION VERDICT: the
        # Toeplitz projection is NOT the cause (|T - R|/|R| = 0.4%,
        # cond equal ~220; frozen exact solve of T w = rcross = 45.4 dB
        # = the full-Gram solve 46.9 = RLS 46.7). The difference is
        # TRAJECTORY quality: per-frame exact re-solve with a constant
        # relative ridge explodes during low-excitation stretches
        # (w jumps 0.5-0.95, error -3 dB -> stream ERLE 8.5), while
        # RLS's P(0) = I/delta fading loading + recursive smoothing keep
        # the trajectory stable (jumps <= 0.02). The CG one-step is
        # trajectory-robust (delta=1 current-power floor, delta<1 hurts
        # or diverges) but slow (29.8 at 5 s, 33.9 at 20 s, still
        # climbing). delta doubles as the RLS loading scale P(0) =
        # I/delta (contributes delta*lam^n*I, fades as data accumulates);
        # sweet spot 0.1-0.3, lambda insensitive on 0.995-0.9995.
        self.solver = str(solver).lower()
        if self.solver not in ('cg', 'rls'):
            raise ValueError(f"unknown solver {solver!r}")
        self.lam = float(lam)
        self.constraint = bool(constraint)
        self.gate_rel = None if gate_rel is None else float(gate_rel)
        self.gate_hold = max(1, int(gate_hold))
        self.gate_decay = float(gate_decay)
        self.gate_floor = float(gate_floor)
        self.sr = float(sr)
        self.q = _sqrt_hann(self.nfft)
        # WOLA synthesis normalization: analysis q + synthesis q gives an
        # OLA sum of k/2 for the periodic sqrt-Hann (sum_m q^2[n - m*hop]
        # = k * mean(q^2) = k/2). At k = 2 that is exactly 1 (why the bug
        # was invisible); at k = 4 the output stream came out DOUBLED
        # (PR probe read 0 dB = (2d - d)^2 == d^2). Scale synthesis only;
        # analysis and the subband-domain adaptation are gain-free.
        self._syn = 2.0 / (self.nfft // self.hop)
        self.reset()

    def reset(self):
        nfft, hop, n_g = self.nfft, self.hop, self.n_g
        nbin = nfft // 2 + 1
        self.x_hist = np.zeros(nfft - hop)      # analysis frame history
        self.d_hist = np.zeros(nfft - hop)
        self.acc = np.zeros(nfft)               # synthesis accumulator
        self.buf_X = np.zeros((nbin, n_g), dtype=complex)
        self.w = np.zeros((nbin, n_g), dtype=complex)
        # Zero-init correlations (the legacy lesson): rcross == autoR
        # elementwise for d = ref, so w = e0 solves the system exactly,
        # and T -> 0 in silence where the |den| guard self-suspends.
        self.autoR = np.zeros((nbin, n_g), dtype=complex)
        self.rcross = np.zeros((nbin, n_g), dtype=complex)
        self.phi = np.zeros((nbin, n_g), dtype=complex)   # error-gradient
        self.v = np.zeros((nbin, n_g), dtype=complex)      # CG direction
        self.g_prev = np.zeros((nbin, n_g), dtype=complex)
        self._cg_live = False                              # v valid?
        # RLS inverse-Gram state (solver='rls'): P(0) = I/delta, i.e.
        # R_eff(0) = delta*I — the loading decays as delta*lam^n once
        # data accumulates (soft regularized startup, no tuning of a
        # separate ridge).
        self.P = np.broadcast_to(np.eye(n_g, dtype=complex) / self.delta,
                                 (nbin, n_g, n_g)).copy()
        self._frame = 0
        self.e = np.zeros(hop)
        self._Prun = 0.0
        self._hold = 0

    # ------------------------------------------------------------------
    def _rls_step(self, rx, E):
        """Per-bin exponentially-windowed RLS (solver='rls').

        Exact recursive solve of R_n w = b_n with R_n = lam*R_{n-1} +
        conj(u)u^T (u = rx, the TRUE per-bin Gram — no Toeplitz
        projection), b_n = lam*b_{n-1} + conj(u)*D, via Sherman-Morrison:
        K = P conj(u) / (lam + u^H P u), w += K*xi with xi the a-priori
        subband error E, P <- (P - K (u^T P))/lam. P(0) = I/delta gives a
        loading delta*lam^n that fades as data accumulates. Frozen when
        the gate is closed (no P inflation in silence).
        """
        v = np.conj(rx)
        Pv = np.einsum('bij,bj->bi', self.P, v)
        den = (self.lam +
               np.real(np.einsum('bi,bij,bj->b', rx, self.P, v)) + 1e-30)
        K = Pv / den[:, np.newaxis]
        self.w = self.w + K * E[:, np.newaxis]
        uP = np.einsum('bi,bij->bj', rx, self.P)
        self.P = (self.P - K[:, :, np.newaxis] * uP[:, np.newaxis, :]) \
            / self.lam
        # re-Hermitianize against roundoff drift (same cost as the update)
        self.P = 0.5 * (self.P + np.conj(np.swapaxes(self.P, 1, 2)))

    def process(self, x_block, d_block):
        """One hop of audio: x_block/d_block [hop] in, error [hop] out
        (a priori). Causal WOLA latency = nfft - hop samples — same
        convention as CONJUGATE_FULL.process."""
        nfft, hop = self.nfft, self.hop

        # ---- analysis: the only windows in the chain
        x_frame = np.concatenate([self.x_hist, x_block])
        d_frame = np.concatenate([self.d_hist, d_block])
        X = np.fft.rfft(self.q * x_frame)
        D = np.fft.rfft(self.q * d_frame)
        self.x_hist = x_frame[-(nfft - hop):]
        self.d_hist = d_frame[-(nfft - hop):]

        # ---- regressor shift register (partition j = subband lag j)
        self.buf_X = np.roll(self.buf_X, -1, axis=1)
        self.buf_X[:, -1] = X
        rx = self.buf_X[:, ::-1]                        # rx[:, j] = X(m-j)

        # ---- THE ERROR: plain per-bin subband difference — no [0;e],
        #      no head, the criterion problem does not exist here.
        E = D - np.sum(self.w * rx, axis=1)

        # ---- excitation gate (project convention)
        gate_open = True
        if self.gate_rel is not None:
            P_ref = float(np.sum(np.abs(X) ** 2))
            P_mic = float(np.sum(np.abs(D) ** 2))
            decay = np.exp(-hop / (self.sr * self.gate_decay))
            self._Prun = max(P_ref, decay * self._Prun)
            if (P_ref > self.gate_rel * P_mic and
                    P_ref > self.gate_floor * self._Prun):
                self._hold = self.gate_hold
            else:
                self._hold -= 1
            gate_open = self._hold > 0

        # ---- correlations (the legacy accumulation, on subband
        #      samples): accumulate when identifiable, always forget.
        if gate_open:
            self.autoR += self.alpha * (np.conj(rx) * X[:, np.newaxis])
            self.rcross += self.alpha * (np.conj(rx) * D[:, np.newaxis])
        self.autoR *= self.beta
        self.rcross *= self.beta

        # ---- the CG solve on toeplitz(autoR) w = rcross, Chang &
        #      Willson (2000) Table I. The fresh residual
        #      g = rcross - T w_last is mathematically the paper's (52)
        #      INCLUDING the new-sample correction (our correlations
        #      already contain this frame's rank-1 update; one matvec).
        #      Line search (50)/(3): a = 0.999*<g,v>/<v,Tv> — the 0.999
        #      is the paper's eta damping (same constant as the MATLAB
        #      reference), + delta*Pbar*||v||^2 regularized curvature
        #      (needed because the ESTIMATED T is near-singular; the
        #      paper assumes exact R). Guard: suspend unless Re>1e-30.
        if gate_open and self.solver == 'rls':
            self._rls_step(rx, E)
        elif gate_open:
            self._frame += 1
            if self.gradient == 'error':
                # gamma-averaged instantaneous error gradient (PAES):
                # E was formed from the a-priori estimate above — exact
                # stochastic gradient of the subband LS at w_last.
                self.phi = (self.gamma * self.phi +
                            (1.0 - self.gamma) *
                            np.conj(rx) * E[:, np.newaxis])
                g = self.phi
                # Curvature floor from the CURRENT frame's regressor
                # power: the long-window T is near-null in spectral
                # holes the current frame may excite, where a Pbar-
                # (window-average) floored step explodes — measured
                # -655 dB. With the current-power floor the T-null
                # limit degrades to an exact NLMS step of mu = 1/delta
                # (the same eps_rel trick as the WOLA-NLMS sibling),
                # while T-excited directions keep the second-order
                # curvature. delta is therefore 1/mu_eff: 1.0 = NLMS
                # mu 1 worst case.
                X2 = np.sum(np.abs(rx) ** 2, axis=1)
                P_floor = X2 + self.eps_rel * np.mean(X2) + 1e-30
            else:
                g = self.rcross - _tmatvec(self.autoR, self.w)
                P_floor = max(float(self.autoR[:, 0].real.mean()), 0.0)
            Pbar = P_floor

            def _line_search(g_, v_):
                tv = _tmatvec(self.autoR, v_)
                num = 0.999 * np.einsum('bj,bj->b', np.conj(g_), v_).real
                vnorm2 = np.einsum('bj,bj->b', np.conj(v_), v_).real
                den = (np.einsum('bj,bj->b', np.conj(v_), tv).real +
                       self.delta * Pbar * vnorm2)
                alf = np.zeros_like(num)
                ok = (den > 1e-30) & (num > 0)
                np.divide(num, den, out=alf, where=ok)
                return alf, tv

            def _beta(g_, g_prev_, v_prev_):
                # complex Hermitian beta, clipped to [0,1] (pfdaf_cg
                # forms; 'pr' = paper eq. (53) Polak-Ribiere)
                gg = np.einsum('bj,bj->b', np.conj(g_), g_).real
                y = g_ - g_prev_
                if self.beta_method == 'fr':
                    num, den = gg, np.einsum(
                        'bj,bj->b', np.conj(g_prev_), g_prev_).real
                elif self.beta_method == 'pr':
                    num = np.einsum('bj,bj->b', np.conj(y), g_).real
                    den = np.einsum('bj,bj->b',
                                    np.conj(g_prev_), g_prev_).real
                elif self.beta_method == 'dy':
                    num = gg
                    den = np.einsum('bj,bj->b', np.conj(v_prev_), y).real
                else:                       # 'hs'
                    num = np.einsum('bj,bj->b', np.conj(y), g_).real
                    den = np.einsum('bj,bj->b', np.conj(v_prev_), y).real
                bcg = np.where(den > 1e-30,
                               num / np.maximum(den, 1e-30), 0.0)
                return np.clip(bcg, 0.0, 1.0)

            if self.k_max == 1:
                # CG1: one step per frame, cross-frame direction memory,
                # optional periodic reset to the fresh gradient (paper:
                # "important to periodically reset ... for sample-by-
                # sample processing"; a true SD step at every reset).
                if (self.beta_method == 'sd' or not self._cg_live or
                        (self.reset_every is not None and
                         self._frame % self.reset_every == 0)):
                    v = g
                else:
                    bcg = _beta(g, self.g_prev, self.v)
                    v = g + bcg[:, np.newaxis] * self.v
                alf, _ = _line_search(g, v)
                self.w = self.w + alf[:, np.newaxis] * v
                self.v = v
                self.g_prev = g
                self._cg_live = True
            else:
                # CG2: correlations frozen for k_max internal iterations;
                # restart from the fresh gradient each frame ("goto
                # Start"). internal='model': residual by the frozen-model
                # recursion g <- g - a*T*v (paper (52) minus the
                # new-sample term, which enters next frame).
                # internal='true': recompute the instantaneous error
                # gradient from the updated w each iteration (error-mode
                # CG2; each step then minimizes the TRUE current cost,
                # no model drift).
                g_prev = None
                for k in range(self.k_max):
                    if self.internal == 'true' and k > 0:
                        E_k = D - np.sum(self.w * rx, axis=1)
                        g = np.conj(rx) * E_k[:, np.newaxis]
                        v = g
                    elif k == 0 or g_prev is None:
                        v = g
                    else:
                        bcg = _beta(g, g_prev, v)
                        v = g + bcg[:, np.newaxis] * v
                    alf, tv = _line_search(g, v)
                    self.w = self.w + alf[:, np.newaxis] * v
                    g_prev = g
                    if self.internal == 'model':
                        g = g - alf[:, np.newaxis] * tv
                self._cg_live = False     # CG2 restarts every frame
            if self.constraint:
                Wt = np.fft.irfft(self.w, n=nfft, axis=0)
                Wt[hop:] = 0.0
                self.w = np.fft.rfft(Wt, n=nfft, axis=0)
        else:
            self._cg_live = False      # stale direction after a freeze

        # ---- synthesis: window + overlap-add (k/2-normalized); emit
        #      the oldest hop
        self.acc = np.concatenate([self.acc[hop:], np.zeros(hop)])
        self.acc += self._syn * self.q * np.fft.irfft(E, n=nfft)
        self.e = self.acc[:hop].copy()
        return self.e


def fb_toeplitz_aec(x, d, nfft=512, hop=256, n_g=8, alpha=0.05, beta=0.97,
                    beta_method='sd', delta=0.1, k_max=1, reset_every=None,
                    gradient='correlation', gamma=0.25, internal='model',
                    constraint=False, gate_rel=0.3, solver='cg', lam=0.999):
    """Block driver: full signals in, residual stream out (aligned to
    the input; the first nfft-hop samples are 0 = causal WOLA
    latency)."""
    n = min(len(x), len(d)) // hop * hop
    aec = CONJUGATE_FB_TOEPLITZ(nfft=nfft, hop=hop, n_g=n_g, alpha=alpha,
                                beta=beta, beta_method=beta_method,
                                delta=delta, k_max=k_max,
                                reset_every=reset_every, gradient=gradient,
                                gamma=gamma, internal=internal,
                                constraint=constraint,
                                gate_rel=gate_rel, solver=solver, lam=lam)
    lag_blocks = nfft // hop - 1          # completed-samples age
    e = np.zeros(n)
    for i in range(n // hop):
        blk = aec.process(x[i*hop:(i+1)*hop], d[i*hop:(i+1)*hop])
        j = i - lag_blocks
        if j >= 0:
            e[j*hop:(j+1)*hop] = blk
    return e


if __name__ == '__main__':
    import os
    import time

    # -- 1. PR passthrough (zero weights -> E = D passthrough) ---------
    rng = np.random.default_rng(0)
    d = rng.standard_normal(32000)
    aec = CONJUGATE_FB_TOEPLITZ()
    blocks = [aec.process(np.zeros(aec.hop), d[s:s+aec.hop])
              for s in range(0, len(d) - aec.hop, aec.hop)]
    y = np.concatenate(blocks[aec.nfft // aec.hop - 1:])
    core = slice(2000, 20000)
    pr = 10 * np.log10(np.sum((y[core] - d[core]) ** 2) / np.sum(d[core] ** 2))
    print(f"PR passthrough (zero weights): {pr:.1f} dB")
    aec4 = CONJUGATE_FB_TOEPLITZ(nfft=1024, hop=256)      # k = 4 (OSFB)
    blocks = [aec4.process(np.zeros(aec4.hop), d[s:s+aec4.hop])
              for s in range(0, len(d) - aec4.hop, aec4.hop)]
    y4 = np.concatenate(blocks[aec4.nfft // aec4.hop - 1:])
    pr4 = 10 * np.log10(np.sum((y4[core] - d[core]) ** 2) / np.sum(d[core] ** 2))
    print(f"PR passthrough 1024/256 (k=4):  {pr4:.1f} dB")

    # -- 2. the grid question: aligned vs sub-grid delays ---------------
    x = rng.standard_normal(32000) * 0.1
    tr = 16000
    print("\ndelay sweep (white noise, alpha=0.05 beta=0.97, n_g=8):")
    for delay in (512, 768, 600, 384):
        dd = np.concatenate([np.zeros(delay), x])[:32000]
        e = fb_toeplitz_aec(x, dd)
        er = 10 * np.log10(np.sum(dd[tr:]**2) / np.sum(e[tr:]**2))
        print(f"  delay {delay:4d} (mod hop = {delay % 256:3d}, "
              f"{(delay % 256) / 256:.0%} off-grid): ERLE {er:7.2f} dB")

    # -- 3. does more partition length soften the sub-grid floor? ------
    print("\nsub-grid floor vs n_g (delay 600 = 88/256 off-grid):")
    dd = np.concatenate([np.zeros(600), x])[:32000]
    for n_g in (4, 8, 16, 32):
        e = fb_toeplitz_aec(x, dd, n_g=n_g)
        er = 10 * np.log10(np.sum(dd[tr:]**2) / np.sum(e[tr:]**2))
        print(f"  n_g {n_g:2d}: ERLE {er:7.2f} dB")

    # -- 4. correlation-window trade-off (beta) on aligned delay -------
    print("\nbeta sweep (aligned delay 512):")
    dd = np.concatenate([np.zeros(512), x])[:32000]
    for beta in (0.9, 0.97, 0.99, 0.999):
        e = fb_toeplitz_aec(x, dd, beta=beta)
        er = 10 * np.log10(np.sum(dd[tr:]**2) / np.sum(e[tr:]**2))
        print(f"  beta {beta:5.3f}: ERLE {er:7.2f} dB")

    # -- 5. solver direction: sd (legacy single step) vs true CG -------
    print("\nbeta_method on aligned delay 512 (4 s noise):")
    x4 = rng.standard_normal(64000) * 0.1
    dd4 = np.concatenate([np.zeros(512), x4])[:64000]
    for bm in ('sd', 'fr', 'pr'):
        e = fb_toeplitz_aec(x4, dd4, beta_method=bm)
        r1 = 10*np.log10(np.sum(dd4[16000:32000]**2) / np.sum(e[16000:32000]**2))
        r2 = 10*np.log10(np.sum(dd4[48000:64000]**2) / np.sum(e[48000:64000]**2))
        print(f"  {bm}: ERLE 1-2 s {r1:7.2f} | 3-4 s {r2:7.2f} dB")

    # -- 6. convergence speed vs the NLMS sibling (1 s / 3 s ERLE) -----
    from conjugate_full import wola_aec
    print("\nconvergence, aligned delay 512 (Toeplitz-PR vs WOLA-NLMS):")
    eT = fb_toeplitz_aec(x4, dd4)
    eN = wola_aec(x4, dd4)
    for t in (16000, 48000):
        rT = 10*np.log10(np.sum(dd4[t:t+16000]**2) / np.sum(eT[t:t+16000]**2))
        rN = 10*np.log10(np.sum(dd4[t:t+16000]**2) / np.sum(eN[t:t+16000]**2))
        print(f"  t = {t/16000:.0f} s: Toeplitz {rT:7.2f} | NLMS(mu=0.1) {rN:7.2f} dB")

    # -- 6. canonical pair (official metrics, skip 1 s) -----------------
    here = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(here, 'audio', 'reference.wav')
    mic_path = os.path.join(here, 'audio', 'microphone.wav')
    if os.path.exists(ref_path):
        import soundfile as sf
        ref, sr = sf.read(ref_path)
        mic, _ = sf.read(mic_path)
        print("\ncanonical pair (speech, delay 800 = 32/256 off-grid):")
        for beta in (0.97, 0.99, 0.999):
            t0 = time.time()
            e = fb_toeplitz_aec(ref, mic, beta=beta)
            er = 10 * np.log10(np.sum(mic[sr:]**2) / np.sum(e[sr:]**2))
            print(f"  beta {beta:5.3f}: ERLE {er:6.2f} dB  ({time.time()-t0:.1f} s)")
        eN = wola_aec(ref, mic)
        rN = 10 * np.log10(np.sum(mic[sr:]**2) / np.sum(eN[sr:]**2))
        print(f"  WOLA-NLMS sibling:  ERLE {rN:6.2f} dB")
    else:
        print("\n(audio/reference.wav not found — skipping pair benchmark)")

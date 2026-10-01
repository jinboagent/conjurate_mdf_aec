"""
Adapters for connecting existing algorithms to the harness interface.

These adapters wrap algorithm implementations to implement the
AdaptiveFilter interface required by the test harness.

Engine contract (harness.core.runner._process_signal): filt() receives
FULL fft_size time-domain frames and returns the full-frame time error;
the engine itself keeps the last step_size samples (overlap-save). The
frequency-domain classes below satisfy this with one rfft -> apply ->
irfft round trip; their zero-headed a-priori output makes the kept tail
exactly the new-block error.
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ..core.interfaces import AdaptiveFilter


class FreqDomainBlockAdapter(AdaptiveFilter):
    """
    Shared adapter for the frequency-domain block classes in
    conjugate_mdf.py (CONJUGATE_MDF hop mode, FD_NLMS). The wrapped
    filter adapts internally in apply(); update() is a no-op.

    Parameters
    ----------
    fft_size, step : frame length L and hop M (L = R*M geometry)
    n_g : number of partitions (delay coverage (n_g + R - 2)*M samples)
    filter_factory : zero-arg callable returning the wrapped filter
    """

    def __init__(self, fft_size: int, step: int, n_g: int, filter_factory):
        self.fft_size = fft_size
        self.step = step
        self.n_g = n_g
        self.nbin = fft_size // 2 + 1
        self.f = filter_factory()

    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        E = self.f.apply(np.fft.rfft(d).reshape(-1, 1),
                         np.fft.rfft(x).reshape(-1, 1))
        return np.fft.irfft(E[:, 0], n=self.fft_size)

    def update(self, e: np.ndarray) -> None:
        pass  # adaptation happens inside apply()

    def reset(self) -> None:
        self.f.reset()

    def get_echo_path(self) -> np.ndarray:
        """Assemble the time-domain path from the partition weights:
        lag j's first `step` taps occupy delays [j*step, (j+1)*step)."""
        w = self.f.w[0][:, :, 0]                      # [nbin, n_g]
        h = np.zeros(self.n_g * self.step)
        for j in range(self.n_g):
            h[j*self.step:(j+1)*self.step] = \
                np.fft.irfft(w[:, j], n=self.fft_size)[:self.step]
        return h


class CGMDFAdapter(FreqDomainBlockAdapter):
    """CONJUGATE_MDF — the canonical [0;e] MDF (G-constraint built in,
    reference-excitation gate). Same input/output frame geometry as
    FD_NLMS; only the weight update differs (beta = gradient averaging)."""

    def __init__(self, n_g: int = 64, fft_size: int = 512, step: int = 128,
                 mu: float = 1.0, beta: float = 0.0, gate_rel=0.3):
        from conjugate_mdf import CONJUGATE_MDF
        super().__init__(
            fft_size, step, n_g,
            lambda: CONJUGATE_MDF(NCHAN=1, NBIN=fft_size // 2 + 1, N_G=n_g,
                                  hop=step, mu=mu, beta=beta,
                                  gate_rel=gate_rel))


class FDNLMSAdapter(FreqDomainBlockAdapter):
    """FD_NLMS: classic partitioned-block FDAF baseline ([0;e] + G)."""

    def __init__(self, n_g: int = 64, fft_size: int = 512, step: int = 128,
                 mu: float = 1.0, constraint: bool = True,
                 full_frame_error: bool = False, gate_rel=0.3):
        from conjugate_mdf import FD_NLMS
        super().__init__(
            fft_size, step, n_g,
            lambda: FD_NLMS(NCHAN=1, NBIN=fft_size // 2 + 1, N_G=n_g,
                            mu=mu, hop=step, constraint=constraint,
                            full_frame_error=full_frame_error,
                            gate_rel=gate_rel))


# Historical name kept as an alias (the class was renamed RLSBishengMDF ->
# CONJUGATE_MDF; the adapter now wraps the renamed class).
RLSBishengMDFAdapter = CGMDFAdapter


class PFADFMDFCGAdapter(AdaptiveFilter):
    """Adapter for pfadf_mdf_cg.PFADFMDFCG (thin wrapper — that class
    already implements the block interface)."""

    def __init__(self, n: int = 64, winlen: int = 256, mu: float = 0.03):
        from pfadf_mdf_cg import PFADFMDFCG
        self.filter = PFADFMDFCG(N=n, M=winlen, mu=mu)

    def filt(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        return self.filter.filt(x, d)

    def update(self, e: np.ndarray) -> None:
        self.filter.update(e)

    def reset(self) -> None:
        self.filter.reset()

    def get_echo_path(self) -> np.ndarray:
        return self.filter.get_echo_path()


def create_standard_rlsmdf_adapter(n_g: int = 64, fft_size: int = 512,
                                   step: int = 128, mu: float = 1.0
                                   ) -> CGMDFAdapter:
    """Standard CG-MDF adapter with recommended (hop-mode) parameters."""
    return CGMDFAdapter(n_g=n_g, fft_size=fft_size, step=step, mu=mu)


def create_standard_pfadf_mdf_cg_adapter(n: int = 64, winlen: int = 256,
                                         mu: float = 0.03
                                         ) -> PFADFMDFCGAdapter:
    return PFADFMDFCGAdapter(n=n, winlen=winlen, mu=mu)

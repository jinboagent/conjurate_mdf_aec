"""
Generic DSP -> TensorFlow wrapper framework, with a 3-contract comparison.

Goal: train the CG-MDF *weight function* W [nbin x N_G] inside a machine
learning framework and compare the result against the untouched numpy
reference (FD_NLMS.py::FD_NLMS, the former CONJUGATE_MDF; imported only — never modified).

The three integration contracts:

  A (exact)   DSP re-expressed in pure TF ops; GradientTape differentiates
              the filtering automatically.                        -> conjugate_mdf_tf.py (TF port)
  B (exact)   numpy DSP forward + hand-written VJP registered with
              @tf.custom_gradient. Because partitioned filtering is LINEAR
              in the weights, the weight gradient has a closed form:
              dL/dW = -sum_b conj(Xp[b]) * rfft([0, dL/de_b])      -> this file
  C (black box)  tf.py_function black box + unbiased minibatch straight-
                 through gradient, in the trainable_dsp.py architectural
                 style: single complex W tensor, @tf.custom_gradient seam,
                 numpy complex-Adam manager. batch_m = 1 is the classic
                 noisy (LMS-like) baseline; batch_m = B gives the exact
                 gradient. Default 64.                          -> this file

A gradient cannot flow through a plain numpy call or tf.py_function — that
boundary is the fundamental blocker. Contracts A and B remove it in exact
ways from opposite sides; C keeps the DSP fully black-box and pays for it
in gradient variance only.

Experiment design (per project decision):
  - canonical speech input (audio/reference.wav, 10 s) with a short
    synthetic echo path (~5 taps inside 4 blocks)
  - N_G = 4, M = 256, FFT = 512, step = 128 (75% overlap — the geometry the
    reference is proven stable in); echo path coverage 4*128 = 512 samples
  - adaptation parameters FIXED at stable values (alpha = 0.03, beta = 0.97);
    the ONLY trainable in every method is W
  - identical protocol for A/B/C: W starts at zero, Adam, same loss
    (MSE of the residual), same step budget
  - metrics via harness/metrics/comparison.py; ERLE reported both with the
    harness 0.25 s transient and after 4 s (post-convergence) because the
    reference's startup transient is much longer than the harness default

Run:  .venv/Scripts/python neural_train/trainable_dsp_wrapper_copy.py
Artifacts land in results/<timestamp>_dsp-hybrid/.
"""

import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from FD_NLMS import FD_NLMS                          # numpy reference (untouched;
                                                           # CONJUGATE_MDF merged into
                                                           # FD_NLMS(beta) 2026-10-06)
from conjugate_mdf_tf import partitioned_filter_tf, run_cg_mdf_tf
from harness.metrics.comparison import (
    calculate_echo_path_metrics,
    calculate_erle,
    compare_outputs,
)

# ---------------------------------------------------------------------------
# Configuration — small and fixed so the only moving part is W
# ---------------------------------------------------------------------------
N_G = 4            # partitions (per decision: short, keeps runs quick)
M = 256            # window half-length
FFT = 2 * M        # 512
STEP = 128         # frame advance (75% overlap, canonical reference geometry)
NBIN = FFT // 2 + 1
ALPHA = 0.03       # fixed adaptation step (never trained)
BETA = 0.97        # fixed forgetting factor (never trained)
SR = 16000
TRANSIENT = 4000       # harness ERLE convention (0.25 s)
TRANSIENT_CONV = 64000 # post-convergence ERLE window (4 s)
SEED = 7
ADAM_LR = 0.03
TRAIN_STEPS = 400

# synthetic echo path: 5 sparse taps inside the 512-sample coverage
H_TRUE = np.zeros(1024)
for delay, amp in [(150, 0.50), (220, -0.25), (300, 0.18), (380, 0.12), (450, 0.08)]:
    H_TRUE[delay] = amp

# rfft endpoint weights for the exact adjoint of irfft-then-tail
# (c_k = 2 for interior bins, 1 for DC and Nyquist)
_C_K = np.full(NBIN, 2.0, dtype=np.float64)
_C_K[0] = 1.0
if FFT % 2 == 0:
    _C_K[-1] = 1.0

# contract C: number of blocks averaged in the minibatch straight-through
# gradient (1 = classic noisy baseline, B = exact full-batch gradient)
C_BATCH_M = 64


def make_dataset():
    """Canonical speech reference + single-talk synthetic echo."""
    import soundfile as sf
    x, sr = sf.read(os.path.join(REPO, 'audio', 'reference.wav'))
    assert sr == SR
    x = x.astype(np.float32)
    d = np.convolve(x, H_TRUE)[:len(x)].astype(np.float32)
    return x, d


def block_spectra(sig, n_blocks):
    """Sliding FFT windows sig[b*STEP : b*STEP + FFT] -> [B, nbin]."""
    frames = np.stack([sig[b * STEP:b * STEP + FFT] for b in range(n_blocks)])
    return np.fft.rfft(frames, axis=1).astype(np.complex64)


def partition_spectra(X_spec):
    """Xp[b, p] = spectrum of the window p steps earlier; zeros before start."""
    B = X_spec.shape[0]
    Xp = np.zeros((B, N_G, NBIN), dtype=np.complex64)
    for p in range(N_G):
        Xp[p:, p] = X_spec[:B - p]
    return Xp


def block_tails(sig, n_blocks):
    """Valid (new) samples of each window: d[b*STEP+FFT-STEP : b*STEP+FFT]."""
    return np.stack([sig[b * STEP + FFT - STEP:b * STEP + FFT]
                     for b in range(n_blocks)]).astype(np.float32)


def numpy_partitioned_filter(Xp, W, d_tail):
    """numpy forward (the 'DSP side' for contracts B and C): e = d - y."""
    Y = np.sum(Xp * W[None, :, :].transpose(0, 2, 1), axis=1)   # [B, nbin]
    y_tail = np.fft.irfft(Y, n=FFT, axis=1)[:, -STEP:]
    return d_tail - y_tail


def extract_path(W):
    """
    Composite echo-path estimate by impulse probing: filter a unit impulse
    (placed at the overlap-save latency FFT-STEP so the returned IR starts
    at delay 0) through the trained weights.
    """
    n_probe = 10
    delta = np.zeros(n_probe * STEP + FFT, dtype=np.float32)
    delta[FFT - STEP] = 1.0
    Xp_d = partition_spectra(block_spectra(delta, n_probe))
    e = numpy_partitioned_filter(Xp_d, W.astype(np.complex64),
                                 np.zeros((n_probe, STEP), np.float32))
    return (-e).reshape(-1)     # with d_tail = 0, e = -y, so y = -e is the IR


def to_timeline(e_blocks):
    """Place block tails on the absolute sample axis (tail starts at FFT-STEP)."""
    out = np.zeros(len(x_global), dtype=np.float32)
    flat = e_blocks.reshape(-1)
    start = FFT - STEP
    out[start:start + len(flat)] = flat
    return out


# ---------------------------------------------------------------------------
# Contract B: numpy forward + closed-form custom gradient (exact)
# ---------------------------------------------------------------------------
def make_bridge_filter(Xp_np, d_tail_np):
    """
    Wrap a numpy filtering forward in @tf.custom_gradient.

    The VJP uses the fact that the map W -> e is linear:
        e_b = d_b - tail(irfft(sum_p Xp[b,p] * W[p]))
        dL/dW[p,k] = -sum_b conj(Xp[b,p,k]) * Ghat_b[k],
        Ghat_b = c_k/Nfft * rfft([0, dL/de_b])  (adjoint of irfft-then-tail)
    """
    Xp_t = tf.constant(Xp_np)                                   # [B, N_G, nbin]
    d_tail_t = tf.constant(d_tail_np)
    B = Xp_np.shape[0]
    c_k = _C_K

    @tf.custom_gradient
    def bridged(Wr, Wi):
        W = (Wr.numpy() + 1j * Wi.numpy()).astype(np.complex64)  # numpy DSP boundary
        e_np = numpy_partitioned_filter(Xp_np, W, d_tail_np)     # [B, STEP] float32

        def grad(dy):                                            # dy = dL/de [B, STEP]
            g = dy.numpy().astype(np.float64)
            pad = np.zeros((B, FFT), dtype=np.float64)
            pad[:, FFT - STEP:] = g
            Ghat = (c_k / FFT) * np.fft.rfft(pad, axis=1)        # adjoint, [B, nbin]
            gW = -np.einsum('bpk,bk->kp', Xp_np.conj(), Ghat)    # [nbin, N_G]
            return (tf.constant(gW.real, tf.float32),
                    tf.constant(gW.imag, tf.float32))

        return tf.constant(e_np), grad

    return bridged, Xp_t, d_tail_t


# ---------------------------------------------------------------------------
# Contract C: black-box forward + unbiased minibatch straight-through VJP
# (trainable_dsp.py architectural style — consolidated from wrapper2)
# ---------------------------------------------------------------------------
def make_trainable_dsp_c_step(batch_m):
    """
    Build the C-contract training step for a given gradient batch size.

    batch_m lives in a closure, NOT in the decorated signature — TF counts
    decorated-function arguments it can convert to tensors, and a scalar in
    the signature makes the expected gradient count ambiguous.

    batch_m blocks are averaged in the straight-through gradient:
    batch_m=1 is the classic noisy baseline; batch_m=B is the exact
    full-batch gradient.
    """
    batch_m = int(batch_m)

    @tf.custom_gradient
    def trainable_dsp_c_step(Xp, d_tail, weights):
        """
        Args:
            Xp:      [B, N_G, nbin] complex partition spectra (constant)
            d_tail:  [B, STEP] mic tail samples (constant)
            weights: [nbin, N_G] complex64 — the trainable weight function

        Returns:
            e: [B, STEP] residual
        """
        B = int(Xp.shape[0])

        def forward_np(xp_np, dt_np, w_np):
            # ANY numpy DSP can live here (black box); args may arrive as
            # EagerTensors through the custom_gradient eager path — convert
            xp = xp_np.numpy() if hasattr(xp_np, 'numpy') else np.asarray(xp_np)
            dt = dt_np.numpy() if hasattr(dt_np, 'numpy') else np.asarray(dt_np)
            w = w_np.numpy() if hasattr(w_np, 'numpy') else np.asarray(w_np)
            return numpy_partitioned_filter(xp, w.astype(np.complex64), dt)

        e = tf.py_function(forward_np, [Xp, d_tail, weights], tf.float32)
        e.set_shape([B, STEP])

        def grad(dy):
            # unbiased Monte-Carlo estimate of dL/dW (TF complex convention):
            # exact gradient = -sum_b f(dy_b); averaging m blocks drawn
            # WITHOUT replacement gives gW = -(B/m) * sum_{i in idx} f(dy_i)
            idx = np.random.default_rng().choice(B, size=batch_m, replace=False)
            g = dy.numpy()[idx].astype(np.float64) * (B / float(batch_m))
            pad = np.zeros((batch_m, FFT), dtype=np.float64)
            pad[:, FFT - STEP:] = g
            Ghat = (_C_K / FFT) * np.fft.rfft(pad, axis=1)
            gW = -np.einsum('mpk,mk->kp', Xp.numpy()[idx].conj(), Ghat)
            return None, None, tf.constant(gW.astype(np.complex64))

        return e, grad

    return trainable_dsp_c_step


class DSPParameterManager:
    """
    Holds the weights as ONE complex64 numpy array (single source of truth)
    and performs the Adam step in numpy.

    Why not Keras Adam: Keras 3's optimizer silently DROPS the imaginary
    part of complex gradients (verified in this TF build: a constant
    (1+0.5j) gradient moved only the real axis). Numpy Adam below applies
    the full complex update — m stays complex, v tracks |g|^2 (real).

    The forward pass only ever sees a plain tensor — no tf.Variable and no
    real/imaginary split crosses the DSP seam.
    """

    def __init__(self, w0, learning_rate=ADAM_LR,
                 beta1=0.9, beta2=0.999, eps=1e-7):
        self.weights = np.asarray(w0, dtype=np.complex64)
        self.lr = learning_rate
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros(self.weights.shape, dtype=np.float64)
        self.t = 0

    def get_params(self):
        """Current weights as a plain (watchable) complex64 tensor."""
        return tf.convert_to_tensor(self.weights)

    def update(self, grad):
        """Apply one Adam step from a complex gradient tensor."""
        g = grad.numpy().astype(np.complex64)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * np.abs(g) ** 2
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        self.weights = (self.weights
                        - self.lr * mhat / (np.sqrt(vhat) + self.eps)
                        ).astype(np.complex64)


def train_c_with_parameter_manager(Xp, d_tail, batch_m=C_BATCH_M,
                                   steps=TRAIN_STEPS, lr=ADAM_LR):
    """Train the complex weight function W through the C-contract seam."""
    pm = DSPParameterManager(np.zeros((NBIN, N_G), np.complex64), lr)
    step_fn = make_trainable_dsp_c_step(batch_m)
    Xp_t = tf.constant(Xp)
    d_t = tf.constant(d_tail)
    losses = []
    for step in range(steps):
        weights = pm.get_params()                 # plain tensor, numpy-sourced
        with tf.GradientTape() as tape:
            tape.watch(weights)
            e = step_fn(Xp_t, d_t, weights)
            loss = tf.reduce_mean(tf.square(e))
        grad = tape.gradient(loss, weights)       # ONE complex64 gradient
        pm.update(grad)
        losses.append(float(loss.numpy()))
        if (step + 1) % 100 == 0:
            print(f"    [C m={batch_m}] step {step+1}/{steps}  "
                  f"loss {losses[-1]:.3e}", flush=True)
    return pm.weights, losses


# ---------------------------------------------------------------------------
# Training loop — identical protocol for all three contracts
# ---------------------------------------------------------------------------
def train_weights(mode, Xp_t, d_tail_t, Xp_np, d_tail_np, c_batch_m=C_BATCH_M):
    """
    Adam-train W [nbin, N_G] from zeros; returns (W, loss_history).

    Modes A/B train the re/im variable pair with Keras Adam (exact tape /
    bridge gradients). Mode C delegates to the trainable_dsp.py-style
    trainer: single complex weight tensor through the black-box seam with
    an unbiased minibatch straight-through gradient (c_batch_m blocks).
    """
    if mode == 'C':
        return train_c_with_parameter_manager(Xp_np, d_tail_np,
                                              batch_m=c_batch_m)

    Wr = tf.Variable(tf.zeros((NBIN, N_G), tf.float32))
    Wi = tf.Variable(tf.zeros((NBIN, N_G), tf.float32))
    opt = tf.keras.optimizers.Adam(ADAM_LR)
    losses = []

    bridge_b = None
    if mode == 'B':
        bridge_b, _, _ = make_bridge_filter(Xp_np, d_tail_np)

    for step in range(TRAIN_STEPS):
        with tf.GradientTape() as tape:
            if mode == 'A':
                y = partitioned_filter_tf(Xp_t, tf.complex(Wr, Wi), FFT, STEP)
                e = d_tail_t - y
            else:
                e = bridge_b(Wr, Wi)
            loss = tf.reduce_mean(tf.square(e))
        grads = tape.gradient(loss, [Wr, Wi])
        opt.apply_gradients(zip(grads, [Wr, Wi]))
        losses.append(float(loss.numpy()))
        if (step + 1) % 100 == 0:
            print(f"    [{mode}] step {step+1}/{TRAIN_STEPS}  loss {losses[-1]:.3e}",
                  flush=True)
    return (Wr.numpy() + 1j * Wi.numpy()), losses


# ---------------------------------------------------------------------------
# Reference: the untouched numpy DSP, driven over the same blocks
# ---------------------------------------------------------------------------
def run_reference(X_spec, D_spec, n_blocks):
    np.random.seed(SEED)
    # canonical [0;e] MDF (hop mode); the legacy Toeplitz criterion was
    # removed from the class — this reference now runs the canonical mode
    filt = FD_NLMS(NCHAN=1, NBIN=NBIN, N_G=N_G, mu=1.0, hop=STEP,
                   beta=0.0, gate_rel=None, Nrxref=1)
    w_init = filt.w[0][:, :, 0].astype(np.complex64)        # capture for TF port
    e_ref = np.zeros((n_blocks, STEP), dtype=np.float32)
    for b in range(n_blocks):
        E = filt.apply(D_spec[b].reshape(-1, 1), X_spec[b].reshape(-1, 1))
        e_ref[b] = np.fft.irfft(E[:, 0], n=FFT)[-STEP:]
    w_ref = filt.w[0][:, :, 0]
    return e_ref, w_ref, w_init


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
x_global = None   # set in main(); used by to_timeline


def main():
    t0 = time.time()
    global x_global

    x, d = make_dataset()
    x_global = x
    n_blocks = (len(x) - FFT) // STEP + 1
    X_spec = block_spectra(x, n_blocks)
    D_spec = block_spectra(d, n_blocks)
    d_tail = block_tails(d, n_blocks)
    Xp = partition_spectra(X_spec)
    Xp_t = tf.constant(Xp)
    d_tail_t = tf.constant(d_tail)

    print(f"geometry: N_G={N_G}, M={M}, FFT={FFT}, step={STEP}, nbin={NBIN}, "
          f"{n_blocks} blocks ({len(x)/SR:.1f} s), alpha={ALPHA}, beta={BETA}")
    print(f"h_true: 5 taps, delays {[int(i) for i in np.nonzero(H_TRUE)[0]]}\n")

    # --- gradient sanity: contract B's hand VJP vs contract A's tape gradient
    r1 = np.random.default_rng(1)
    r2 = np.random.default_rng(2)
    Wr0 = tf.constant(0.01 * r1.standard_normal((NBIN, N_G)).astype(np.float32))
    Wi0 = tf.constant(0.01 * r2.standard_normal((NBIN, N_G)).astype(np.float32))
    bridge_check, _, _ = make_bridge_filter(Xp, d_tail)
    with tf.GradientTape() as tp:
        tp.watch(Wr0); tp.watch(Wi0)
        y = partitioned_filter_tf(Xp_t, tf.complex(Wr0, Wi0), FFT, STEP)
        lossA = tf.reduce_mean(tf.square(d_tail_t - y))
    gA = tp.gradient(lossA, [Wr0, Wi0])
    with tf.GradientTape() as tp:
        tp.watch(Wr0); tp.watch(Wi0)
        eB = bridge_check(Wr0, Wi0)
        lossB = tf.reduce_mean(tf.square(eB))
    gB = tp.gradient(lossB, [Wr0, Wi0])
    rel = np.max([np.abs(gA[i].numpy() - gB[i].numpy()) /
                  (np.abs(gA[i].numpy()).max() + 1e-30) for i in (0, 1)])
    print(f"gradient cross-check B vs A: max relative diff = {rel:.2e} "
          f"({'OK' if rel < 1e-3 else 'MISMATCH'})\n")

    # --- reference (untouched numpy DSP)
    print("--- reference: numpy CONJUGATE_MDF (adaptation, alpha/beta fixed)")
    e_ref, w_ref, w_init = run_reference(X_spec, D_spec, n_blocks)

    # --- contract A extra validation: TF recurrence port tracks the reference
    E_tf, _ = run_cg_mdf_tf(tf.constant(D_spec), tf.constant(X_spec),
                            ALPHA, BETA, w_init=w_init)
    e_arec = tf.signal.irfft(E_tf, fft_length=[FFT])[:, -STEP:].numpy()
    rec_diff = float(np.max(np.abs(e_arec - e_ref)))
    print(f"contract A recurrence port vs numpy reference: "
          f"max |e_A - e_ref| = {rec_diff:.2e} "
          f"({'OK (float32)' if rec_diff < 1e-2 else 'DIVERGES'})\n")

    # --- train the weight function under the three contracts
    results = {}
    curves = {}
    for mode in ('A', 'B', 'C'):
        label = f'C m={C_BATCH_M}' if mode == 'C' else mode
        print(f"--- training weights, contract {label} (Adam lr={ADAM_LR}, "
              f"{TRAIN_STEPS} steps)")
        W, losses = train_weights(mode, Xp_t, d_tail_t, Xp, d_tail)
        results[mode] = W
        curves[label] = losses

    # --- metrics
    e_ref_abs = to_timeline(e_ref)
    table = {}

    def output_metrics(name, W):
        e_blocks = numpy_partitioned_filter(Xp, W.astype(np.complex64), d_tail)
        e_abs = to_timeline(e_blocks)
        h_est = extract_path(W)
        row = {
            'erle_db_transient': float(calculate_erle(d, e_abs, TRANSIENT)),
            'erle_db_converged': float(calculate_erle(d, e_abs, TRANSIENT_CONV)),
            'erle_db_last2s': float(calculate_erle(d, e_abs, 8 * SR)),
        }
        pm = calculate_echo_path_metrics(h_est, H_TRUE)
        row.update({k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else v)
                    for k, v in pm.items() if not isinstance(v, (bool, np.bool_))})
        row['passed'] = bool(pm.get('passed', False))
        row['compare_vs_ref'] = str(compare_outputs(e_abs, e_ref_abs, d))
        row['weight_nmse_db_vs_ref'] = float(10 * np.log10(
            np.sum(np.abs(W - w_ref) ** 2) / (np.sum(np.abs(w_ref) ** 2) + 1e-30)))
        table[name] = row
        return h_est, e_abs

    # reference row
    h_ref = extract_path(w_ref)
    ref_row = {
        'erle_db_transient': float(calculate_erle(d, e_ref_abs, TRANSIENT)),
        'erle_db_converged': float(calculate_erle(d, e_ref_abs, TRANSIENT_CONV)),
        'erle_db_last2s': float(calculate_erle(d, e_ref_abs, 8 * SR)),
    }
    pm = calculate_echo_path_metrics(h_ref, H_TRUE)
    ref_row.update({k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else v)
                    for k, v in pm.items() if not isinstance(v, (bool, np.bool_))})
    ref_row['passed'] = bool(pm.get('passed', False))
    table['reference (numpy DSP)'] = ref_row

    # per-second local ERLE profile (diagnostic: pauses vs transients)
    def sec_profile(e_abs):
        out = []
        for i in range(len(d) // SR):
            seg = slice(i * SR, (i + 1) * SR)
            dp = np.sum(d[seg] ** 2) + 1e-30
            out.append(10 * np.log10(dp / (np.sum(e_abs[seg] ** 2) + 1e-30)))
        return np.array(out)

    profiles = {'reference': sec_profile(e_ref_abs)}

    paths = {'reference (numpy DSP)': h_ref, 'h_true': H_TRUE}
    for mode in ('A', 'B', 'C'):
        label = f'C m={C_BATCH_M}' if mode == 'C' else mode
        h_est, e_abs = output_metrics(f"{label} (trained)", results[mode])
        paths[f"{label} (trained)"] = h_est
        profiles[label] = sec_profile(e_abs)

    print("\nper-second local ERLE (dB):")
    for name, prof in profiles.items():
        print(f"  {name:<10} " + " ".join(f"{v:6.1f}" for v in prof))

    # --- report
    print("\n================ COMPARISON ================")
    print(f"{'method':<22} {'ERLE@.25s':>9} {'ERLE@4s':>8} {'ERLE@8s':>8} "
          f"{'corr':>7} {'NMSE dB':>8} {'delay':>6} {'wNMSE dB':>9}")
    for name, row in table.items():
        print(f"{name:<22} {row['erle_db_transient']:>9.2f} "
              f"{row['erle_db_converged']:>8.2f} {row['erle_db_last2s']:>8.2f} "
              f"{row.get('correlation', 0):>7.3f} {row.get('nmse_db', 0):>8.2f} "
              f"{row.get('delay_error_samples', 0):>6.0f} "
              f"{row.get('weight_nmse_db_vs_ref', float('nan')):>9.2f}")

    # artifacts (methodology: results/<timestamp>_dsp-hybrid/)
    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(REPO, 'results', f'{ts}_dsp-hybrid')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({
            'config': {'N_G': N_G, 'M': M, 'fft': FFT, 'step': STEP, 'nbin': NBIN,
                       'alpha': ALPHA, 'beta': BETA, 'adam_lr': ADAM_LR,
                       'steps': TRAIN_STEPS, 'seed': SEED, 'c_batch_m': C_BATCH_M,
                       'transient': TRANSIENT, 'transient_conv': TRANSIENT_CONV,
                       'gradient_check_B_vs_A': rel,
                       'recurrence_port_maxdiff': rec_diff},
            'results': {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                            for kk, vv in v.items()} for k, v in table.items()},
            'loss_curves': curves,
        }, fh, indent=2, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, h in paths.items():
        ax1.plot(h, 'k-' if name == 'h_true' else '-', label=name, linewidth=1.2)
    ax1.set_title('echo path: true vs estimated (impulse probe, N_G=4)')
    ax1.set_xlabel('samples'); ax1.legend(fontsize=7)
    for mode, curve in curves.items():
        ax2.semilogy(curve, '-', label=mode)
    ax2.set_title('training loss (MSE of residual)')
    ax2.set_xlabel('Adam step'); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison.png'), dpi=120)
    print(f"\nwall {time.time()-t0:.1f}s | artifacts in {out_dir}")


if __name__ == '__main__':
    main()

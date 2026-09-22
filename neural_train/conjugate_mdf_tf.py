"""
Contract A — TensorFlow functional port of the Conjugate Gradient MDF.

This is a faithful TF (complex64) reimplementation of the recurrence in
conjugate_mdf.py::CONJUGATE_MDF.apply(), restructured so TensorFlow can
differentiate through it:

- functional state passing (state in -> new state out, no in-place mutation)
- all nbin Hermitian Toeplitz matrices built with a single gather:
  T[b, q, p] = r[b, |q-p|]
  (scipy.linalg.toeplitz called with one argument builds the symmetric /
  Hermitian Toeplitz matrix, whose entries depend only on |q-p|)
- the same one-step CG update with the clamped data-driven step size
- magnitude-rescaled weight clamp (the numpy reference's np.clip on complex
  weights would raise; it never triggers for stable parameters)

Geometry (Soo-Pang MDF): 2M-point FFT frames [x_prev, x_cur] advancing M
samples per block, N_G partitions -> echo path coverage N_G * M samples.

Two entry points:
- cg_mdf_tf_step(): one apply() recurrence step (adaptation active)
- partitioned_filter_tf(): filtering only with FIXED weights W — the
  trainable-weights forward pass (linear in W, vectorized over all blocks)

The numpy reference (conjugate_mdf.py) is never modified; this file only
mirrors its math.
"""

import numpy as np
import tensorflow as tf

# |q-p| index matrices for the vectorized Hermitian Toeplitz build
_TOEPLITZ_IDX_CACHE = {}


def _toeplitz_idx(n_g):
    """
    Index tensors for scipy.linalg.toeplitz(c) semantics:
    T[q,p] = c[q-p] for q>=p (lower), conj(c[p-q]) for q<p (upper)
    -> Hermitian, matching the numpy reference exactly.
    """
    if n_g not in _TOEPLITZ_IDX_CACHE:
        i = np.arange(n_g)
        diff = i[:, None] - i[None, :]
        _TOEPLITZ_IDX_CACHE[n_g] = (
            np.abs(diff).astype(np.int32),                    # |q-p|
            (diff >= 0).astype(np.bool_),                     # lower-triangle mask
        )
    return _TOEPLITZ_IDX_CACHE[n_g]


def initial_state(nbin, n_g, alpha, beta, w_init=None, dtype=tf.complex64):
    """Create the functional state dict (all tensors, Nrxref = nchan = 1)."""
    if w_init is None:
        w_init = np.zeros((nbin, n_g), dtype=np.complex64)
    rd = tf.cast(np.asarray(w_init), dtype)
    return {
        'buf': tf.zeros((nbin, n_g), dtype),      # reference spectra shift-register
        'autoR': tf.cast(tf.fill((nbin, n_g), 1e-4), dtype),
        'rcross': tf.zeros((nbin, n_g), dtype),
        'w': rd,
        'w_last': rd,
        'alpha': tf.constant(alpha, tf.float32),
        'beta': tf.constant(beta, tf.float32),
    }


def cg_mdf_tf_step(state, D, X):
    """
    One frame of the CONJUGATE_MDF.apply() recurrence, in pure TF ops.

    Args:
        state: dict from initial_state()
        D: mic spectrum [nbin] (complex)
        X: reference spectrum [nbin] (complex)

    Returns:
        (E, new_state): error spectrum [nbin] and the updated state dict
    """
    dt = state['w'].dtype
    D = tf.cast(D, dt)
    X = tf.cast(X, dt)
    n_g = state['w'].shape[1]
    alpha = tf.cast(state['alpha'], dt)
    beta = tf.cast(state['beta'], dt)

    # 1. roll the partition buffer, insert the new frame (newest last)
    buf = tf.concat([state['buf'][:, 1:], X[:, tf.newaxis]], axis=1)
    rx_flipped = tf.reverse(buf, axis=[1])        # newest first (delay j*blocks)

    # 2. a-priori filtering with w_last
    micest = tf.reduce_sum(state['w_last'] * rx_flipped, axis=1)
    out_apriori = D - micest

    # 3. correlation updates (add, then forget — same order as the reference)
    new_R1 = rx_flipped * tf.math.conj(X)[:, tf.newaxis]
    new_rc = rx_flipped * tf.math.conj(D)[:, tf.newaxis]
    autoR = (state['autoR'] + alpha * new_R1) * beta
    rcross = (state['rcross'] + alpha * new_rc) * beta

    # 4. one-step CG on the per-bin Hermitian Toeplitz normal equations
    #    diagonal loading on the first column, exactly like the reference
    first = autoR[:, :1] + tf.cast(1e-30, dt)
    r_reg = tf.concat([first, autoR[:, 1:]], axis=1)          # [nbin, n_g]
    idx, lower = _toeplitz_idx(int(n_g))
    G = tf.gather(r_reg, tf.constant(idx), axis=1)            # [nbin, n_g, n_g]
    T = tf.where(tf.constant(lower)[tf.newaxis, :, :],        # Hermitian:
                 G, tf.math.conj(G))                           # conj upper triangle

    w_last = state['w_last']
    g0 = rcross - tf.linalg.matmul(T, w_last[..., tf.newaxis])[..., 0]
    p = g0
    rp = tf.linalg.matmul(T, p[..., tf.newaxis])[..., 0]
    num = 0.999 * tf.reduce_sum(tf.math.real(tf.math.conj(p) * g0), axis=1)
    den = tf.reduce_sum(tf.math.real(tf.math.conj(p) * rp), axis=1) + 1e-30
    alf = tf.math.divide_no_nan(num, den)
    alf = tf.clip_by_value(alf, -1.0, 1.0)
    w = w_last + tf.cast(alf, dt)[:, tf.newaxis] * p
    # magnitude clamp (reference: dormant np.clip branch)
    w_abs = tf.abs(w)
    w = tf.where(w_abs > 10.0, w * tf.cast(10.0 / (w_abs + 1e-30), dt), w)

    # 5. a-posteriori filtering with the updated weights
    micest2 = tf.reduce_sum(w * rx_flipped, axis=1)
    out = D - micest2

    return out, {
        'buf': buf, 'autoR': autoR, 'rcross': rcross,
        'w': w, 'w_last': w,
        'alpha': state['alpha'], 'beta': state['beta'],
    }


def run_cg_mdf_tf(frames_D, frames_X, alpha, beta, w_init=None):
    """
    Drive the TF recurrence over all blocks (adaptation active).

    Args:
        frames_D: [B, nbin] complex mic spectra
        frames_X: [B, nbin] complex reference spectra
        alpha, beta: fixed adaptation parameters
        w_init: optional [nbin, n_g] initial weights (else zeros)

    Returns:
        (E_all [B, nbin], final state dict)
    """
    nbin = int(frames_X.shape[1])
    n_g = w_init.shape[1] if w_init is not None else 4
    state = initial_state(nbin, n_g, alpha, beta, w_init,
                          dtype=tf.as_dtype(tf.convert_to_tensor(frames_X).dtype))
    outs = []
    for b in range(int(frames_X.shape[0])):
        E, state = cg_mdf_tf_step(state, frames_D[b], frames_X[b])
        outs.append(E)
    return tf.stack(outs, axis=0), state


def partitioned_filter_tf(Xp, W, fft_size, block_size):
    """
    Filtering-only forward pass with FIXED weights (trainable-weights mode).

    Linear in W and fully vectorized over blocks — this is what the tape
    differentiates in contract A's training loop.

    Args:
        Xp: [B, N_G, nbin] complex partition spectra (Xp[b, p] = frame b-p)
        W:  [nbin, N_G] complex weight tensor
        fft_size, block_size: 2M and M

    Returns:
        e_tail: [B, block_size] real residual (d_tail - y_tail assumed done
                by the caller); here we return the echo estimate tail
                y_tail [B, block_size]
    """
    Y = tf.reduce_sum(Xp * tf.transpose(W)[tf.newaxis, :, :], axis=1)  # [B, nbin]
    y = tf.signal.irfft(Y, fft_length=[fft_size])
    return y[:, fft_size - block_size:]

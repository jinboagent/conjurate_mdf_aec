"""
Contract C implemented in the trainable_dsp.py architectural style.

The trainable_dsp.py pattern, mapped onto the CG-MDF weight-training problem:

  trainable_dsp.py element          -> here
  --------------------------------------------------------------------
  lms_filter (pure TF forward)      -> numpy partitioned filter, called
                                       through tf.py_function (any DSP fits
                                       at this seam, numpy stays numpy)
  trainable_lms_step               -> trainable_dsp_c_step: @tf.custom_gradient
                                       with a hand-written grad; the ONLY place
                                       autodiff meets the DSP
  ParameterManager (numpy params)   -> DSPParameterManager: the complex weight
                                       array W is ONE complex64 numpy array,
                                       enters the forward as a plain watched
                                       tensor, Adam updates land through a
                                       hidden synced tf.Variable
  train_lms_model                   -> train_c_with_parameter_manager

Weights: a single complex64 tensor [nbin, N_G] — no real/imaginary split.
TF's tape.gradient uses the Wirtinger (descent) convention for complex
parameters, and the custom_gradient seam returns that same complex gradient,
so the math is identical to the split version (verified to 3.5e-7 in the
wrapper_copy comparison run).

The straight-through gradient averages its estimate over `batch_m` random
blocks: batch_m=1 is classic C (unbiased, very noisy); batch_m=B makes it
the exact full-batch gradient. Adam on complex64 variables is supported
(verified in this TF build).

Run: .venv/Scripts/python neural_train/trainable_dsp_wrapper2.py
"""

import numpy as np
import tensorflow as tf

from trainable_dsp_wrapper_copy import (   # shared experiment machinery
    FFT, NBIN, N_G, STEP, SEED, ADAM_LR, TRAIN_STEPS,
    make_dataset, block_spectra, partition_spectra, block_tails,
    numpy_partitioned_filter, extract_path, H_TRUE,
    partitioned_filter_tf,
)

# rfft endpoint weights for the exact adjoint of irfft-then-tail
_C_K = np.full(NBIN, 2.0, dtype=np.float64)
_C_K[0] = 1.0
if FFT % 2 == 0:
    _C_K[-1] = 1.0


# ---------------------------------------------------------------------------
# The custom-gradient seam (analog of trainable_lms_step)
# ---------------------------------------------------------------------------
def make_trainable_dsp_c_step(batch_m):
    """
    Build the C-contract training step for a given gradient batch size.

    batch_m lives in a closure, NOT in the decorated signature — TF counts
    decorated-function arguments it can convert to tensors, and a scalar in
    the signature makes the expected gradient count ambiguous.

    batch_m blocks are averaged in the straight-through gradient:
    batch_m=1 is classic C (unbiased, noisy); batch_m=B is the exact
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
            # ANY numpy DSP can live here; this one is the CG-MDF basis
            # filter. (args may arrive as EagerTensors through the
            # custom_gradient eager path — always convert first)
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


# ---------------------------------------------------------------------------
# External parameter manager (analog of ParameterManager — working version)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Training loop (analog of train_lms_model)
# ---------------------------------------------------------------------------
def train_c_with_parameter_manager(Xp, d_tail, batch_m,
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
# Demo: gradient check (batch_m = B -> exact) + training comparison
# ---------------------------------------------------------------------------
def main():
    x, d = make_dataset()
    n_blocks = (len(x) - FFT) // STEP + 1
    X_spec = block_spectra(x, n_blocks)
    d_tail = block_tails(d, n_blocks)
    Xp = partition_spectra(X_spec)
    Xp_t = tf.constant(Xp)
    d_t = tf.constant(d_tail)
    B = Xp.shape[0]
    print(f"data: {n_blocks} blocks, W shape [{NBIN}, {N_G}] complex64\n")

    # --- gradient check: full-batch straight-through == TF tape gradient
    rng = np.random.default_rng(1)
    W0 = tf.constant((rng.standard_normal((NBIN, N_G)) * 0.01).astype(np.complex64))
    step_exact = make_trainable_dsp_c_step(B)
    with tf.GradientTape() as tp:
        tp.watch(W0)
        y = partitioned_filter_tf(Xp_t, W0, FFT, STEP)
        eA = d_t - y
        lossA = tf.reduce_mean(tf.square(eA))
    gA = tp.gradient(lossA, W0).numpy()
    with tf.GradientTape() as tp:
        tp.watch(W0)
        eC = step_exact(Xp_t, d_t, W0)
        lossC = tf.reduce_mean(tf.square(eC))
    gC = tp.gradient(lossC, W0).numpy()
    # triangulation: the expected VJP computed manually from the residuals
    dyA = (2.0 * eA.numpy() / (B * STEP)).astype(np.float64)
    padA = np.zeros((B, FFT), dtype=np.float64)
    padA[:, FFT - STEP:] = dyA
    gM = -np.einsum('bpk,bk->kp', Xp.conj(),
                    (_C_K / FFT) * np.fft.rfft(padA, axis=1))
    fwd = float(np.max(np.abs(eA.numpy() - eC.numpy())))
    print(f"forward match vs TF: max |eA - eC| = {fwd:.2e}")
    print(f"|gA|inf={np.abs(gA).max():.3e}  |gC|inf={np.abs(gC).max():.3e}  "
          f"|gM|inf={np.abs(gM).max():.3e}")
    print(f"rel(gC, gM) = {np.max(np.abs(gC - gM)) / (np.abs(gM).max() + 1e-30):.2e}"
          f"   rel(gA, gM) = {np.max(np.abs(gA - gM)) / (np.abs(gM).max() + 1e-30):.2e}"
          f"   rel(gA, gC) = {np.max(np.abs(gA - gC)) / (np.abs(gA).max() + 1e-30):.2e}\n")

    # --- training comparison: exact baseline vs classic C vs minibatched C
    # (all three share the same numpy complex-Adam manager; only the
    # gradient SOURCE differs: TF tape vs the custom-gradient seam)
    results = {}
    print("--- baseline A (pure TF forward, exact tape gradient)")
    pm = DSPParameterManager(np.zeros((NBIN, N_G), np.complex64))
    losses_A = []
    for step in range(TRAIN_STEPS):
        W = pm.get_params()
        with tf.GradientTape() as tp:
            tp.watch(W)
            y = partitioned_filter_tf(Xp_t, W, FFT, STEP)
            loss = tf.reduce_mean(tf.square(d_t - y))
        pm.update(tp.gradient(loss, W))
        losses_A.append(float(loss.numpy()))
        if (step + 1) % 100 == 0:
            print(f"    [A] step {step+1}/{TRAIN_STEPS}  loss {losses_A[-1]:.3e}",
                  flush=True)
    results['A exact (baseline)'] = (pm.weights, losses_A[-1])
    print(f"    final loss {results['A exact (baseline)'][1]:.3e}\n")

    for m in (1, 64):
        print(f"--- C, trainable_dsp.py style (single complex W, batch_m={m})")
        W, losses = train_c_with_parameter_manager(Xp, d_tail, batch_m=m)
        results[f'C style, batch_m={m}'] = (W, losses[-1])
        print(f"    final loss {losses[-1]:.3e}\n")

    # --- path recovery check on the best C run
    h_est = extract_path(results['C style, batch_m=64'][0])
    corr = float(np.corrcoef(h_est[:1024], H_TRUE)[0, 1])
    peak = int(np.argmax(np.abs(h_est[:1024])))
    print(f"path recovery (C batch_m=64): corr = {corr:.3f}, "
          f"peak at {peak} (true {int(np.argmax(np.abs(H_TRUE)))})")


if __name__ == '__main__':
    main()

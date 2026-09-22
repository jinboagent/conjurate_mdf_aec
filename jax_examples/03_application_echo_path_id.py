"""Application to this repo's domain: echo-path identification in JAX.

Synthetic scene: far-end x(n) -> unknown FIR echo path h -> microphone d(n).
Three ways to recover h from the pair (x, d):

    A. NLMS online adaptation          -- written as ONE jax.lax.scan
    B. Full-batch gradient training    -- the ../neural_train GradientTape idea:
                                          jit(value_and_grad) + Adam on the MSE
    C. Meta-gradient through NLMS      -- differentiate NLMS's final MSE with
                                          respect to its own step size mu

Run:  .venv/Scripts/python.exe jax_examples/03_application_echo_path_id.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

N = 64          # echo path length in taps
T = 8_000       # number of samples

# --- synthetic data ---------------------------------------------------------
rng = np.random.default_rng(0)
h_true = np.exp(-np.arange(N) / 16.0) * rng.standard_normal(N)     # decaying IR
x = rng.standard_normal(T)                                         # far-end signal
d = np.convolve(x, h_true)[:T] + 1e-3 * rng.standard_normal(T)     # mic = echo + noise

# regressor rows u[n] = [x[n], x[n-1], ..., x[n-N+1]], so that d[n] = u[n] @ h_true
windows = sliding_window_view(x, N)                 # row j = x[j : j+N]
X = jnp.asarray(np.ascontiguousarray(windows[: T - N + 1, ::-1]))
d_win = jnp.asarray(d[N - 1:])
x_j, d_j, h_j = map(jnp.asarray, (x, d, h_true))
print(f"{N}-tap echo path, {T} samples; irreducible MSE floor = 1e-06 (mic noise)")


def erle(d, e):
    """Echo-return-loss enhancement in dB (bigger is better)."""
    return 10.0 * jnp.log10(jnp.sum(d * d) / jnp.sum(e * e))


# ---------------------------------------------------------------------------
print("\n=== A. NLMS as a single lax.scan over the whole signal ===")

def nlms(mu, x, d):
    """Per-sample normalized-LMS; returns the error signal e[n]."""
    def step(carry, inputs):
        w, u = carry                        # u = [x[n], ..., x[n-N+1]]
        x_n, d_n = inputs
        u = jnp.concatenate([x_n[None], u[:-1]])          # shift in the new sample
        e = d_n - u @ w
        w = w + mu * e * u / (u @ u + 1e-6)               # normalized update
        return (w, u), e
    _, errors = jax.lax.scan(step, (jnp.zeros(N), jnp.zeros(N)), (x, d))
    return errors

for mu in (0.1, 0.3, 0.8):
    e = jax.jit(nlms)(mu, x_j, d_j)
    tail = slice(T // 2, T)                  # ERLE after convergence
    print(f"  mu={mu:.1f}: converged ERLE = {float(erle(d_j[tail], e[tail])):6.2f} dB")

# ---------------------------------------------------------------------------
print("\n=== B. full-batch gradient training (the GradientTape experiment, in JAX) ===")

def mse(w, X, d):
    e = X @ w - d
    return jnp.mean(e * e)

def train_adam(w0, X, d, steps=500, lr=3e-2):
    """Adam on the batch MSE; the entire training loop is one compiled scan."""
    def step(carry, _):
        w, m, v, t = carry
        loss, g = jax.value_and_grad(mse)(w, X, d)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        w = w - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        return (w, m, v, t + 1), loss
    carry0 = (w0, jnp.zeros(N), jnp.zeros(N), 1)
    (w, *_), losses = jax.lax.scan(step, carry0, length=steps)
    return w, losses

w_hat, losses = jax.jit(train_adam)(jnp.zeros(N), X, d_win)
e = X @ w_hat - d_win
print(f"  loss: {float(losses[0]):.3e} -> {float(losses[-1]):.3e} in {len(losses)} steps")
print(f"  final ERLE = {float(erle(d_win, e)):6.2f} dB, "
      f"||h_hat - h_true|| = {float(jnp.linalg.norm(w_hat - h_j)):.2e}")

# ---------------------------------------------------------------------------
print("\n=== C. meta-gradient: differentiate NLMS w.r.t. its own step size ===")

def nlms_tail_mse(mu, x, d):
    e = nlms(mu, x, d)
    return jnp.mean(e[len(e) // 2:] ** 2)    # MSE after convergence

mu0 = 0.3
g_mu = jax.grad(nlms_tail_mse)(mu0, x_j, d_j)
eps = 1e-5
fd = (nlms_tail_mse(mu0 + eps, x_j, d_j) - nlms_tail_mse(mu0 - eps, x_j, d_j)) / (2 * eps)
print(f"  d/dmu [converged NLMS MSE] at mu={mu0}: "
      f"jax.grad = {float(g_mu):+.4e}, finite diff = {float(fd):+.4e}")
print("  -> one call backprops through all 8,000 adaptation steps; this is exactly")
print("     the mechanism behind 'trainable DSP' / learned optimizers.")

print("\nA is online (single pass, no gradient); B is offline batch training;")
print("C shows the two worlds compose: the adaptive algorithm itself is differentiable.")

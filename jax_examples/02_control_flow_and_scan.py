"""Control flow in JAX: Python loops unroll under jit -- use lax loops instead.

    lax.fori_loop   counted loop with an accumulator
    lax.scan        counted loop with carried state AND per-step outputs
                    (the workhorse for RNNs and streaming DSP)
    lax.cond        traced if/else
    lax.while_loop  data-dependent termination (no reverse-mode gradients)

Run:  .venv/Scripts/python.exe jax_examples/02_control_flow_and_scan.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
print("=== 1. fori_loop: a counted loop that compiles to ONE loop ===")

xs = jnp.arange(10.0)

def sum_squares(xs):
    def body(i, acc):
        return acc + xs[i] * xs[i]
    return jax.lax.fori_loop(0, len(xs), body, 0.0)

print("fori_loop:", float(jax.jit(sum_squares)(xs)), "  numpy:", float(np.sum(xs**2)))
# A Python `for` would also work, but inside jit it gets *unrolled*:
# 10,000 iterations => a 10,000-operation graph and a very slow compile.

# ---------------------------------------------------------------------------
print("\n=== 2. scan: carried state + stacked per-step outputs ===")

def cumsum_scan(xs):
    def body(carry, x):            # (carry, x) -> (new carry, output)
        carry = carry + x
        return carry, carry
    _, out = jax.lax.scan(body, 0.0, xs)
    return out

print("scan  :", cumsum_scan(xs).tolist())
print("numpy :", np.cumsum(xs).tolist())
assert np.allclose(cumsum_scan(xs), np.cumsum(xs))

# ---------------------------------------------------------------------------
print("\n=== 3. streaming DSP: a one-pole IIR filter IS a scan ===")
# y[n] = a * x[n] + (1 - a) * y[n-1]   -- a loop with state, i.e. a scan

def one_pole(alpha, xs):
    def step(y_prev, x):
        y = alpha * x + (1.0 - alpha) * y_prev
        return y, y
    _, ys = jax.lax.scan(step, 0.0, xs)
    return ys

rng = np.random.default_rng(0)
signal = jnp.asarray(rng.standard_normal(20_000))
alpha = 0.05
ys = jax.jit(one_pole)(alpha, signal)

# sanity check against the analytic noise gain of the filter
power_in = float(jnp.mean(signal**2))
power_out = float(jnp.mean(ys**2))
analytic_db = 10.0 * np.log10(alpha**2 / (1.0 - (1.0 - alpha) ** 2))
print(f"output/input power: {10.0 * np.log10(power_out / power_in):+.2f} dB "
      f"(analytic {analytic_db:+.2f} dB)")

# ---------------------------------------------------------------------------
print("\n=== 4. gradients THROUGH the loop: the differentiable-DSP superpower ===")

def output_energy(alpha, xs):
    return jnp.mean(one_pole(alpha, xs) ** 2)

g = jax.grad(output_energy)(alpha, signal)
eps = 1e-5
fd = (output_energy(alpha + eps, signal) - output_energy(alpha - eps, signal)) / (2 * eps)
print(f"d/d(alpha) mean(y^2) at alpha=0.05:  jax.grad = {float(g):+.6e}   "
      f"finite diff = {float(fd):+.6e}")
print("-> backpropagation ran through 20,000 filter steps in one compiled pass.")

# ---------------------------------------------------------------------------
print("\n=== 5. cond and while_loop (brief) ===")

safe_sqrt = jax.jit(
    lambda v: jax.lax.cond(v >= 0.0, jnp.sqrt, lambda u: jnp.full_like(u, float("nan")), v)
)
print("cond: sqrt(4) =", float(safe_sqrt(4.0)), "  sqrt(-1) =", float(safe_sqrt(-1.0)))

def newton_sqrt(a):
    def cond(s):
        return jnp.abs(s * s - a) > 1e-12
    def body(s):
        return 0.5 * (s + a / s)
    return jax.lax.while_loop(cond, body, a)

print("while_loop: newton sqrt(9) =", float(newton_sqrt(9.0)))
# while_loop's trip count is data-dependent, so jax.grad does NOT support it;
# if you need gradients, restructure the loop as scan/fori_loop.

# Extra: lax.scan(f, ..., unroll=k) trades compile time for code size, and
# jax.checkpoint(f) (rematerialization) trades compute for activation memory.

print("\nnext: 03_application_echo_path_id.py -- NLMS, Adam-trained filters, and a meta-gradient")

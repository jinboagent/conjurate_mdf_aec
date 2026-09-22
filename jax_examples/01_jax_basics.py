"""JAX basics: the five ideas everything else builds on.

    1. jax.numpy feels like NumPy, but arrays are immutable
    2. jit   -- compile a pure function with XLA
    3. grad  -- automatic differentiation of pure functions
    4. vmap  -- automatic vectorization; transformations compose freely
    5. explicit PRNG keys and pytrees (nested containers of arrays)

Run:  .venv/Scripts/python.exe jax_examples/01_jax_basics.py
"""

import jax

# JAX defaults to float32; enable float64 when you want NumPy-level numerics.
# This must run before the first array is created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
banner("1. arrays: numpy-like, but immutable")
x = jnp.linspace(0.0, 1.0, 5)
print("x               =", x)

# x[0] = 99.0                 # TypeError: JAX arrays don't support item assignment
x2 = x.at[0].set(99.0)        # functional update: returns a NEW array
print("x.at[0].set(99) =", x2)
print("x is unchanged  =", x)

A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
print("matmul/sin/sum all work as in numpy:", (A @ A).tolist(), float(jnp.sin(0.5)), float(A.sum()))

# ---------------------------------------------------------------------------
banner("2. jit: compile a pure function with XLA")

def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))

gelu_compiled = jax.jit(gelu)
z = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("gelu(z) =", gelu_compiled(z))   # first call traces the function, then compiles

# You can inspect the traced computation graph (the "jaxpr"):
print("its jaxpr:")
print(jax.make_jaxpr(gelu)(z))

# jit rules of thumb:
#   * the function must be pure: no side effects, no printing, no global state
#   * shapes & dtypes are part of the compiled artifact: each new shape recompiles

# ---------------------------------------------------------------------------
banner("3. grad: differentiate pure functions")

def mse_loss(w, X, d):
    """Plain math, no framework constructs -- JAX differentiates this directly."""
    e = X @ w - d
    return jnp.mean(e * e)

rng = np.random.default_rng(0)
X = jnp.asarray(rng.standard_normal((200, 4)))
w_true = jnp.array([2.0, -1.0, 0.5, 3.0])
d = X @ w_true

w = jnp.zeros(4)
loss_and_grad = jax.jit(jax.value_and_grad(mse_loss))   # transformations compose
for _ in range(60):
    loss, g = loss_and_grad(w, X, d)
    w = w - 0.1 * g
print(f"gradient descent on MSE: loss {float(loss):.3e}, "
      f"w -> {np.round(np.asarray(w), 3).tolist()} (true {w_true.tolist()})")

g = jax.grad(lambda t: t**3)(2.0)
print(f"sanity check: d/dx x^3 at 2.0 = {float(g)} (analytic: 12.0)")
# argnums selects which argument to differentiate; jax.grad(f, argnums=(0, 2)) for several

# ---------------------------------------------------------------------------
banner("4. vmap: automatic vectorization")

def weighted_dot(u, v, scale):     # written for ONE pair of vectors + a scalar
    return scale * (u @ v)

batched = jax.vmap(weighted_dot, in_axes=(0, 0, None))   # batch axes: 0, 0, shared
rows = jnp.arange(12.0).reshape(4, 3)
print("per-row scores:", batched(rows, rows, 10.0))

# THE selling point: transformations compose into one compiled kernel
batched_grads = jax.jit(jax.vmap(jax.grad(lambda v: jnp.sum(v * v))))
print("batched grad of sum(v^2):", batched_grads(rows).tolist(), " (== 2 * each row)")

# ---------------------------------------------------------------------------
banner("5. randomness: explicit PRNG keys, no global seed")

key = jax.random.key(42)
key, k_init, k_noise = jax.random.split(key, 3)     # split BEFORE using each part

w0 = 0.01 * jax.random.normal(k_init, (3, 5))
same = jax.random.normal(k_init, (2,))
print("random init (first row):", w0[0].tolist())
print("reusing a key reproduces the same numbers:",
      bool(jnp.array_equal(same, jax.random.normal(k_init, (2,)))))
# This explicitness is what lets randomness work inside jit and under vmap.

# ---------------------------------------------------------------------------
banner("6. pytrees: parameters as nested dicts/lists of arrays")

params = {
    "layer0": {"w": jnp.ones((3, 4)), "b": jnp.zeros(4)},
    "layer1": {"w": jnp.ones((4, 1)), "b": jnp.zeros(1)},
}
params = jax.tree.map(lambda p: 0.1 * p, params)    # apply a function to every leaf

def l2_norm(tree):
    return sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(tree))

grad_tree = jax.grad(l2_norm)(params)               # gradient has the SAME structure
print("d(l2)/d(layer0.b) =", grad_tree["layer0"]["b"].tolist(), " (== 2 * 0.1)")

# Flax/Equinox models and Optax optimizers are just pytrees + tree.map under the hood.

print("\nnext: 02_control_flow_and_scan.py -- loops, streaming state, and gradients through them")

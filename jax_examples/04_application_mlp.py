"""Application: train a small MLP in pure JAX (no Flax / no Optax).

Shows what the ecosystem libraries automate for you, by hand:
pytree parameters, He init, a jit'ed minibatch Adam step, tree-mapped updates.

Run:  .venv/Scripts/python.exe jax_examples/04_application_mlp.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
print("=== dataset: 3-class spiral ===")

def make_spiral(n_per_class=300, seed=1):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(3):
        r = np.linspace(0.05, 1.0, n_per_class)
        theta = 2.0 * np.pi * c / 3.0 + 3.5 * r + 0.08 * rng.standard_normal(n_per_class)
        xs.append(np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1))
        ys.append(np.full(n_per_class, c))
    return jnp.asarray(np.concatenate(xs)), jnp.asarray(np.concatenate(ys))

X, y = make_spiral()
Y = jax.nn.one_hot(y, 3)
print("points:", X.shape, " classes: 3")

# ---------------------------------------------------------------------------
print("\n=== model: 2 -> 64 -> 64 -> 3 MLP as a pytree of dicts ===")

def init_params(sizes, key):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, k_w = jax.random.split(key)
        params.append({
            "w": jax.random.normal(k_w, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in),
            "b": jnp.zeros(fan_out),
        })
    return params

def forward(params, X):
    h = X
    for layer in params[:-1]:
        h = jax.nn.relu(h @ layer["w"] + layer["b"])
    last = params[-1]
    return h @ last["w"] + last["b"]                 # logits

def cross_entropy(params, X, Y):
    logits = forward(params, X)
    log_probs = jax.nn.log_softmax(logits)             # log p, computed stably
    return -jnp.mean(jnp.sum(Y * log_probs, axis=1))

def accuracy(params, X, y):
    return jnp.mean(jnp.argmax(forward(params, X), axis=1) == y)

# ---------------------------------------------------------------------------
print("\n=== hand-rolled Adam over the whole pytree ===")

def adam_step(params, grads, state, lr=1e-2):
    m, v, t = state
    m = jax.tree.map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, grads)
    v = jax.tree.map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, grads)
    t += 1
    m_hat = jax.tree.map(lambda m_: m_ / (1.0 - 0.9**t), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1.0 - 0.999**t), v)
    params = jax.tree.map(
        lambda p, m_, v_: p - lr * m_ / (jnp.sqrt(v_) + 1e-8), params, m_hat, v_hat
    )
    return params, (m, v, t)

# the whole update -- forward, backward, optimizer -- is one compiled call
@jax.jit
def update(params, state, Xb, Yb):
    grads = jax.grad(cross_entropy)(params, Xb, Yb)
    return adam_step(params, grads, state)

params = init_params([2, 64, 64, 3], jax.random.key(0))
zeros = jax.tree.map(jnp.zeros_like, params)
state = (zeros, jax.tree.map(jnp.zeros_like, params), 0)

rng = np.random.default_rng(0)
for step in range(1, 3001):
    idx = jnp.asarray(rng.choice(len(X), size=128))
    params, state = update(params, state, X[idx], Y[idx])
    if step % 500 == 0:
        print(f"  step {step:4d}   loss {float(cross_entropy(params, X, Y)):.4f}   "
              f"train acc {float(accuracy(params, X, y)):.3f}")

X_new, y_new = make_spiral(seed=7)
print(f"\nfinal: train acc {float(accuracy(params, X, y)):.3f}, "
      f"fresh-spiral acc {float(accuracy(params, X_new, y_new)):.3f}")
print("\nin real projects: Flax gives you the modules, Optax the optimizer --")
print("this file is what they compile down to.")

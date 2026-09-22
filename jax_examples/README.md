# JAX examples

Four standalone scripts, each runnable on its own. All verified under
`jax==0.11.2` (CPU wheel) on Python 3.12 / Windows.

| script | what it shows |
|---|---|
| [`01_jax_basics.py`](01_jax_basics.py) | immutable arrays, `jit`, `grad`, `vmap`, explicit PRNG keys, pytrees |
| [`02_control_flow_and_scan.py`](02_control_flow_and_scan.py) | `lax.fori_loop` / `scan` / `cond` / `while_loop`; gradients *through* a 20,000-step IIR filter |
| [`03_application_echo_path_id.py`](03_application_echo_path_id.py) | echo-path identification three ways: NLMS-as-`scan`, full-batch Adam (the `neural_train` GradientTape idea in JAX), and a meta-gradient w.r.t. NLMS's own step size |
| [`04_application_mlp.py`](04_application_mlp.py) | pure-JAX MLP on a 3-class spiral: pytree params + hand-rolled minibatch Adam |

## Running them

```bash
# CPU JAX is installed in this repo's venv (uv pip install jax).
.venv/Scripts/python.exe jax_examples/01_jax_basics.py
# ... or all four:
for f in jax_examples/0*.py; do .venv/Scripts/python.exe "$f"; done
```

Note: native Windows supports **CPU-only** JAX. GPU/TPU acceleration requires
Linux or WSL2.

## JAX in one paragraph

JAX is NumPy plus a set of **composable function transformations** — `jit`
(compile via XLA), `grad` (autodiff), `vmap` (auto-vectorization), `pmap`
(multi-device), `lax.scan` (loops with state). You write plain numerical
functions and stack the transformations freely: `jax.jit(jax.vmap(jax.grad(f)))`
is one compiled kernel. The price is a functional style: no in-place mutation,
explicit PRNG state, static shapes, and loops expressed as `lax` primitives.

## Sharp bits checklist (the ones that bite first)

1. **Arrays are immutable** — use `x.at[i].set(v)`, which returns a new array.
2. **float32 by default** — `jax.config.update("jax_enable_x64", True)` before
   creating any array if you need float64.
3. **Randomness is explicit** — `jax.random.key(seed)` + `jax.random.split`;
   there is no global seed.
4. **Python `for` loops unroll under `jit`** — use `lax.scan` / `fori_loop`;
   `while_loop` has no reverse-mode gradients.
5. **Jitted functions must be pure** — side effects (print, mutation, global
   state) don't work as in Python.
6. **Shapes are static** — every new shape/dtype triggers a recompile.

## Online resources

### Official

- [JAX documentation](https://jax.readthedocs.io) — start here; also served at [docs.jax.dev](https://docs.jax.dev)
- [JAX 101](https://jax.readthedocs.io/en/latest/jax-101/index.html) — the official step-by-step tutorial (jit, autodiff, vmap, pytrees, PRNG…), runnable in Colab
- [The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) — official guide to the gotchas listed above; read this second
- [jax-ml/jax on GitHub](https://github.com/jax-ml/jax) — source, release notes, and tutorial notebooks under `docs/notebooks/`
- [JAX discussions forum](https://github.com/jax-ml/jax/discussions) — where "how do I…" questions get answered

### Ecosystem libraries (when you outgrow hand-rolling)

- [Flax](https://flax.readthedocs.io) — neural-network library (modules, training utilities)
- [Optax](https://optax.readthedocs.io) — gradient processing & optimizers (Adam etc.)
- [Equinox](https://docs.kidger.site/equinox/) — "everything is a pytree" NN library; by the same author:
- [Diffrax](https://docs.kidger.site/diffrax/) — numerical ODE/SDE/CDE solvers with autodiff
- [NumPyro](https://num.pyro.ai) — probabilistic programming; a good showcase of JAX's flexibility
- [awesome-jax](https://github.com/n2cholas/awesome-jax) — curated list of libraries, projects, papers, and talks

### Tutorials & community write-ups

- [UvA Deep Learning Course notebooks — Tutorial 2: Introduction to JAX+Flax](https://uvadlc-notebooks.readthedocs.io) — hands-on, from transforms to training loops
- [A guide to JAX for PyTorch developers (Google Cloud blog, 2025)](https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers) — mental-model mapping if you come from PyTorch (or TF)

### Differentiable DSP / audio (closest to this repo)

- [DDSP: Differentiable Digital Signal Processing (Engel et al., 2020)](https://arxiv.org/abs/2001.04643) — the foundational paper: classic DSP elements (filters, oscillators, reverb) inside a trainable network
- [A Review of Differentiable DSP for Music & Speech Synthesis (Hayes et al.)](https://arxiv.org/abs/2306.07877) — survey of the field (published in Frontiers in Signal Processing, 2024)
- [jaxdsp](https://github.com/khiner/jaxdsp) — differentiable audio processors with a browser client for real-time control

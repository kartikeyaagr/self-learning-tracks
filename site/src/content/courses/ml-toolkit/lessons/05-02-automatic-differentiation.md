---
id: "05.02"
course: "ml-toolkit"
module: "05"
slug: "automatic-differentiation"
title: "Automatic differentiation"
keyIdeas: ["`jax.grad(f)` is a new function that returns the gradient of a scalar-valued `f` with respect to its first argument (or the ones in `argnums`). There is no tape and no `.grad` attribute: the gradient is a return value with the same structure as the input.", "`jax.value_and_grad` returns the loss and its gradient from one forward and backward pass, and `has_aux=True` lets the function also return metrics that are not differentiated.", "`jvp` (forward mode) pushes a tangent through the function; `vjp` (reverse mode) pulls a cotangent back. `jacfwd` builds Jacobians from JVPs and suits tall Jacobians (few inputs); `jacrev` uses VJPs and suits wide ones (few outputs).", "Because transformations compose, higher-order derivatives are just nesting: `jax.hessian` is `jacfwd(jacrev(f))`, and a Hessian-vector product is a `jvp` of `grad` that never forms the Hessian.", "`lax.stop_gradient` blocks a path, and `jax.custom_vjp` replaces the backward rule of a function, for example a straight-through estimator for rounding."]
question: "You want the largest eigenvalue of the loss Hessian for a 50-million-parameter model, to check whether your learning rate is near the stability limit $2/\\lambda_{max}$. A colleague proposes `jnp.linalg.eigvalsh(jax.hessian(loss)(params))`. Explain why this fails and describe a JAX approach that works, including its cost per iteration."
answer: "The full Hessian has $(5\\times10^7)^2 = 2.5\\times10^{15}$ entries, about 10 petabytes in float32, so `jax.hessian` cannot even be materialised, let alone decomposed (and with a pytree of parameters it would be a nested block structure anyway). Power iteration only needs products $Hv$. Define `hvp = lambda v: jax.jvp(jax.grad(loss_of_params), (params,), (v,))[1]`, where `loss_of_params` closes over a fixed batch. Start with a random pytree `v` of the same structure as `params`, then repeat: `w = hvp(v)`, estimate $\\lambda = \\langle v, w\\rangle$ (summing over all leaves), and set `v = w / ||w||`. Each iteration costs roughly one forward and two backward-like passes (a JVP through the gradient computation, about 2 to 3 times a gradient), and memory is a small multiple of one gradient. Wrap the step in `jax.jit` so it compiles once, run some tens of iterations until $\\lambda$ stabilises, and use the same batch throughout because the Hessian depends on the data. The top eigenvalue is the one of largest magnitude, which near a minimum is usually the largest positive one."
sources: [{"title": "JAX docs: Transformations: grad and vmap", "url": "https://docs.jax.dev/en/latest/101/transformations.html", "note": "grad, value_and_grad, has_aux and differentiating with respect to pytrees."}, {"title": "JAX docs: The autodiff cookbook", "url": "https://docs.jax.dev/en/latest/301/cookbook.html", "note": "jvp and vjp, jacfwd versus jacrev, Hessians and Hessian-vector products."}, {"title": "JAX docs: Custom derivative rules", "url": "https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html", "note": "custom_jvp and custom_vjp with worked numerical-stability examples."}]
practice: ["value-and-grad-of-loss", "hessian-vector-product", "straight-through-estimator-with-custom-vjp", "spectral-norm-by-power-iteration"]
tags: ["autodiff", "grad", "Hessian-vector products"]
tools: ["JAX"]
---

## Gradients as functions

In PyTorch, autograd records operations as they run and `loss.backward()` writes results into `.grad` attributes (lesson 04.02). JAX takes a functional view. `jax.grad` takes a Python function and returns another Python function that computes the derivative:

```python
import jax, jax.numpy as jnp

def loss(w, x, y):
    pred = x @ w
    return jnp.mean((pred - y) ** 2)

dloss_dw = jax.grad(loss)                 # gradient w.r.t. the first argument
g = dloss_dw(w, x, y)                     # same shape and dtype as w
gw, gx = jax.grad(loss, argnums=(0, 1))(w, x, y)
```

Nothing accumulates, so there is nothing to zero. The function must return a real scalar, and the arguments being differentiated must be floating point: `grad` on an integer input raises a `TypeError` rather than silently returning zeros. Arguments can be any pytree (lesson 5), so if `w` is a dict of arrays, the gradient is a dict with the same keys.

## value_and_grad and auxiliary outputs

Training needs both the loss and its gradient. Calling `loss` and then `grad(loss)` runs the forward pass twice; `value_and_grad` does it once. Metrics such as accuracy go in an auxiliary output:

```python
def loss_fn(params, batch):
    logits = model_apply(params, batch["x"])
    l = cross_entropy(logits, batch["y"])
    acc = jnp.mean(jnp.argmax(logits, -1) == batch["y"])
    return l, {"acc": acc}                # (scalar, aux)

(l, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
```

With `has_aux=True` the function must return a pair, and only the first element is differentiated.

## Forward and reverse mode

For $f: \mathbb{R}^n \to \mathbb{R}^m$ with Jacobian $J$ at $x$, JAX exposes the two primitive products:

- `jax.jvp(f, (x,), (v,))` returns $(f(x), Jv)$: forward mode, one pass per input direction, memory about the same as the forward pass.
- `y, f_vjp = jax.vjp(f, x)` then `f_vjp(u)` returns $u^\top J$: reverse mode, one pass per output direction, but it must store intermediate values from the forward pass.

`grad` is `vjp` with $u = 1$ for a scalar output. `jacfwd` stacks $n$ JVPs and `jacrev` stacks $m$ VJPs (both use `vmap` internally, lesson 4), so prefer `jacfwd` when $n \ll m$ and `jacrev` when $m \ll n$. A loss with millions of parameters and one output is the extreme reverse-mode case, which is why training uses `grad`.

~~~dg-flow How the derivative tools are built from the two primitives.
vjp -(cotangent 1)-> grad -(jvp of grad)-> *HVP
vjp -(vmap over outputs)-> jacrev -(jacfwd over it)-> hessian
jvp -(vmap over inputs)-> jacfwd
~~~

## Higher-order derivatives

Derivatives are ordinary JAX functions, so you can transform them again. `jax.grad(jax.grad(f))` gives a second derivative of a scalar function, and `jax.hessian(f)` is `jacfwd(jacrev(f))`: reverse mode for the gradient, then forward mode over it, which is the efficient order for a square Hessian.

For a model with $n$ parameters the Hessian has $n^2$ entries, which is impossible to store for any real network. What optimisation and analysis methods usually need is a Hessian-vector product $Hv$, and forward-over-reverse computes it at the cost of a few gradient evaluations:

```python
def hvp(f, x, v):
    return jax.jvp(jax.grad(f), (x,), (v,))[1]
```

This is the building block for power iteration on the top Hessian eigenvalue, Newton-CG and influence functions. If `f` takes a pytree, `x` and `v` are pytrees of the same structure.

## Controlling the backward pass

`jax.lax.stop_gradient(z)` is the identity in the forward pass and has zero derivative, the JAX counterpart of `.detach()`. Use it for targets that must not receive gradients (a target network, an EMA teacher) or to hide a normalisation constant.

Sometimes you want a different backward rule. Rounding has zero gradient almost everywhere, so quantisation-aware training uses a straight-through estimator: round in the forward pass, pass gradients through unchanged in the backward pass. `jax.custom_vjp` expresses this directly:

```python
@jax.custom_vjp
def round_ste(x):
    return jnp.round(x)

def round_fwd(x):
    return jnp.round(x), None            # output, residuals saved for backward

def round_bwd(res, g):
    return (g,)                          # one cotangent per input, as a tuple

round_ste.defvjp(round_fwd, round_bwd)
```

The same tool gives numerically stable gradients for functions like `log1p(exp(x))`, or gradient clipping inside a layer. `jax.custom_jvp` is the forward-mode counterpart and is preferred when a JVP rule is enough, because it supports both modes.

<div class="ex">
<div class="lab">Worked example</div>

**grad of a loss that was secretly integer**

A contrastive loss takes `temperature` as an argument and you call `jax.grad(loss, argnums=2)(params, batch, 1)`. JAX raises a `TypeError`, because `1` is an integer and integer inputs have no meaningful derivative.

Passing `1.0` fixes it. The same check catches a subtler bug: if the loss casts its inputs with `.astype(jnp.int32)` along the way, the gradient through that path is zero. JAX does not warn about a cast inside the function, so test that gradients are non-zero for every parameter group on a small batch.

Finally, compare against finite differences with `check_grads(f, (x,), order=2)` (imported with `from jax.test_util import check_grads`) before trusting a custom rule.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

in JAX a derivative is just another function. Use `value_and_grad` with `has_aux` for training, pick `jacfwd` or `jacrev` by the shape of the Jacobian, compose `jvp` with `grad` for Hessian-vector products, and reach for `custom_vjp` only when the default rule is wrong for your purpose.

</div>

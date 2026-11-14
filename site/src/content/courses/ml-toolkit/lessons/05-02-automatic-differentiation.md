---
id: "05.02"
course: "ml-toolkit"
module: "05"
slug: "automatic-differentiation"
title: "Automatic differentiation"
keyIdeas: ["`jax.grad(f)` is a new Python function that returns the gradient of a scalar-valued `f` with respect to its first argument (or the ones in `argnums`). There is no tape and no `.grad` attribute: the gradient is a return value with the same structure (pytree) as the input.", "Autodiff applies the chain rule to each primitive operation, so it is exact up to rounding, unlike finite differences, and it handles loops and branches because it only sees the operations that ran.", "`jax.value_and_grad` returns the loss and its gradient from one forward and backward pass, and `has_aux=True` lets the function also return metrics that are not differentiated.", "`jvp` (forward mode) pushes an input direction through the function; `vjp` (reverse mode) pulls an output weighting back. `jacfwd` stacks JVPs and suits tall Jacobians (few inputs); `jacrev` stacks VJPs and suits wide ones (few outputs), which is why training uses reverse mode.", "Transformations compose, so higher derivatives are nesting: `jax.hessian` is `jacfwd(jacrev(f))`, and a Hessian-vector product is a `jvp` of `grad` that never forms the Hessian.", "`lax.stop_gradient` blocks a path, and `jax.custom_vjp` (or `custom_jvp`) replaces a function's derivative rule, for example a straight-through estimator for rounding or a numerically stable gradient."]
question: "You want the largest eigenvalue of the loss Hessian for a 50-million-parameter model, to check whether your learning rate is near the stability limit $2/\\lambda_{max}$. A colleague proposes `jnp.linalg.eigvalsh(jax.hessian(loss)(params))`. Explain why this fails and describe a JAX approach that works, including its cost and memory per iteration."
answer: "The full Hessian has $(5\\times10^7)^2 = 2.5\\times10^{15}$ entries, about 10 petabytes in float32, so `jax.hessian` cannot even be materialised, let alone decomposed (and with a pytree of parameters it would come back as a nested block structure anyway). Building it would also cost 50 million JVPs of the gradient, one per column. Power iteration needs only products $Hv$, and forward-over-reverse computes them without forming $H$. Define `hvp = lambda v: jax.jvp(jax.grad(loss_of_params), (params,), (v,))[1]`, where `loss_of_params` closes over one fixed batch, because the Hessian depends on the data. Start from a random pytree `v` with the same structure as `params` and normalise it. Then repeat: `w = hvp(v)`, estimate $\\lambda = \\langle v, w\\rangle$ by summing elementwise products over all leaves (`jax.tree.map` plus a sum), and set `v = w / ||w||`. Each iteration costs one JVP through the gradient computation, roughly two to three times the cost of one gradient, and memory is a small multiple of one gradient plus the residuals of the backward pass, which `jax.checkpoint` can reduce if needed. Wrap the step in `jax.jit` so it compiles once, and run some tens of iterations until $\\lambda$ stabilises. Power iteration converges to the eigenvalue of largest magnitude, which near a minimum is usually the largest positive one; if you suspect a large negative curvature, check the sign of $\\lambda$. Compare $2/\\lambda_{max}$ with your learning rate: if they are close, training is at the edge of stability. Before trusting the result, validate `hvp` on a tiny model against `jax.hessian(loss)(params) @ v`."
sources: [{"title": "JAX docs: Transformations: grad and vmap", "url": "https://docs.jax.dev/en/latest/101/transformations.html", "note": "grad, value_and_grad, has_aux and differentiating with respect to pytrees."}, {"title": "JAX docs: The autodiff cookbook", "url": "https://docs.jax.dev/en/latest/301/cookbook.html", "note": "jvp and vjp, jacfwd versus jacrev, Hessians and Hessian-vector products."}, {"title": "JAX docs: Custom derivative rules", "url": "https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html", "note": "custom_jvp and custom_vjp with worked numerical-stability examples."}]
practice: ["value-and-grad-of-loss", "hessian-vector-product", "straight-through-estimator-with-custom-vjp", "spectral-norm-by-power-iteration"]
tags: ["autodiff", "grad", "Hessian-vector products"]
tools: ["JAX"]
---

## What you will learn

- What automatic differentiation is, and why it beats both pencil-and-paper derivatives and finite differences.
- How `jax.grad` turns a Python function into its derivative function, and the rules that function must follow (scalar output, float inputs).
- How to get the loss, its gradient and extra metrics in one pass with `value_and_grad` and `has_aux`.
- The two engines underneath, forward mode (`jvp`) and reverse mode (`vjp`), and how to pick `jacfwd` or `jacrev` from the shape of the Jacobian.
- How to nest transformations for second derivatives and Hessian-vector products, and how to change the backward pass with `stop_gradient` and `custom_vjp`.

*Before you start:* lesson 05.01 (JAX as NumPy: `jax.numpy`, immutable arrays, float32 by default) and the idea of a derivative and the chain rule. It helps to have seen PyTorch autograd (lesson 04.02), because this lesson contrasts the two.

## From arrays to derivatives

Lesson 05.01 showed that JAX looks like NumPy. This lesson adds what NumPy cannot do: differentiate your code, which is what training needs. PyTorch records operations as they run and `loss.backward()` fills `.grad` attributes. JAX has no recording mode and no `.grad`: a derivative is just another Python function, obtained by transforming the original one.

## The idea in one picture

Take three data points, $x = (1, 2, 3)$ and $y = (2, 4, 7)$, and a model with a single weight: predict $w \cdot x$. The loss is the mean squared error:

```python
import jax, jax.numpy as jnp
x = jnp.array([1., 2., 3.]); y = jnp.array([2., 4., 7.])

def loss(w):
    return jnp.mean((w * x - y) ** 2)

dloss = jax.grad(loss)        # a new Python function
print(loss(1.0), dloss(1.0))  # 7.0 -11.333334
print(dloss(31 / 14))         # -1.46e-06  (about 0: the minimum)
```

~~~dg-fig 05.02-idea.svg
Left: the loss you wrote, with its tangent line at w = 1. Right: the function jax.grad gave back. Its value at w = 1 is the tangent's slope (−11.33), and it crosses zero exactly where the loss is smallest.
~~~

`jax.grad(loss)` is "the derivative of `loss`"; calling it at `1.0` gives the slope there. The negative sign says increasing $w$ lowers the loss. Check it by hand: the derivative of $\frac13 \sum_i (w x_i - y_i)^2$ is $\frac13 \sum_i 2(w x_i - y_i)x_i$. At $w = 1$ the residuals are $(-1, -2, -4)$, so the terms are $2 \cdot (-1) \cdot 1 = -2$, $2 \cdot (-2) \cdot 2 = -8$ and $2 \cdot (-4) \cdot 3 = -24$. Their mean is $-34/3 = -11.33$, matching JAX.

Terms to keep:

- **Automatic differentiation (autodiff)** computes exact derivatives of a program by applying the chain rule to each primitive operation it executes. It is neither symbolic algebra nor a numerical approximation.
- A **function transformation** takes a function and returns a new function. `jax.grad` is the first one you meet; `jit` (05.03) and `vmap` (05.04) are others, and they all combine freely.
- The **gradient** is the vector of partial derivatives of a scalar output with respect to each input. It always has the same shape as the input it belongs to.

## Step 1: grad is a function that returns a function

`jax.grad(f)` differentiates `f` with respect to its first positional argument. Use `argnums` to choose others:

```python
def f(a, b):
    return a * b + jnp.sin(a)

jax.grad(f)(2., 3.)                    # 2.5838532   (df/da)
jax.grad(f, argnums=(0, 1))(2., 3.)    # (2.5838532, 2.0)  one gradient per argument
```

The arguments you differentiate can be any **pytree**: a nested structure of dicts, lists and tuples whose leaves are arrays (lesson 05.05 covers them in depth). The gradient comes back with the same structure, so parameter dictionaries just work:

```python
params = {"w": jnp.array([0.5, -1.0]), "b": 0.0}
X = jnp.array([[1., 2.], [3., 4.], [5., 6.]]); Y = jnp.array([1., 2., 3.])

def mse(p, X, Y):
    return jnp.mean((X @ p["w"] + p["b"] - Y) ** 2)

print(jax.grad(mse)(params, X, Y))
# {'b': -9.0, 'w': [-32.333336, -41.333336]}
```

Two rules are enforced, and both are deliberate:

- **The output must be a real scalar.** For a vector output JAX raises `TypeError: Gradient only defined for scalar-output functions`. Reduce it first (sum or mean), or use a Jacobian tool from Step 4.
- **Differentiated inputs must be floating point.** `jax.grad(f)(2, 3.)` raises a `TypeError`: integers have no infinitesimal change. Pass `2.0`.

Nothing accumulates, so there is nothing to zero between steps (no forgotten `zero_grad()`); you simply hold the gradients and pass them on.

## Step 2: how the derivative is actually computed

**Symbolic differentiation** rewrites the formula like a computer-algebra system; expressions blow up and loops or branches are out of reach. **Finite differences** nudge the input and measure the change: $f'(x) \approx \frac{f(x+h) - f(x)}{h}$. That costs one extra evaluation per input (hopeless for a million parameters) and is inaccurate. If $h$ is large the slope changes within the step; if $h$ is small the two function values agree in almost every digit and subtracting them leaves mostly rounding noise.

~~~dg-fig 05.02-fd.svg
Error of the derivative of sin(x)·exp(x) at x = 1. Finite differences trace a V: no step size is good in float32, and even the best float64 step is only borrowed accuracy. jax.grad needs no step size and lands at float32 rounding level (about 2e-8).
~~~

**Automatic differentiation** runs your program and, for every primitive operation (`*`, `sin`, `exp`, `dot`), applies that operation's known derivative rule. The chain rule then glues the pieces together. The result is exact up to rounding (the flat line in the figure), and Python control flow is no problem, because autodiff only sees the operations that ran.

The chain rule can be applied in two orders, the key idea of this lesson:

- **Forward mode** carries a derivative alongside each value from inputs to output. One pass gives the derivative of every output with respect to *one* input direction.
- **Reverse mode** first runs the function forward and stores intermediate values, then sweeps backwards from the output. One pass gives the derivative of *one* output with respect to every input.

<div class="ex">
<div class="lab">Worked example</div>

**one function, both modes, by hand**

Take $f(a, b) = a \cdot b + \sin a$ at $a = 2$, $b = 3$. Break it into primitives: $v_1 = a \cdot b = 6$, $v_2 = \sin a = 0.9093$, $f = v_1 + v_2 = 6.9093$.

Forward mode, seeding the direction $(\dot a, \dot b) = (1, 0)$ ("wiggle $a$ only"). Each value carries a dot: $\dot v_1 = \dot a \cdot b + a \cdot \dot b = 1 \cdot 3 + 2 \cdot 0 = 3$. Then $\dot v_2 = \cos(a) \cdot \dot a = -0.4161 \cdot 1 = -0.4161$. So $\dot f = 3 - 0.4161 = 2.5839 = \partial f/\partial a$. Getting $\partial f/\partial b$ needs a second pass with seed $(0, 1)$, which gives $\dot v_1 = 2$, $\dot v_2 = 0$, so $\dot f = 2$.

Reverse mode, seeding the output with $\bar f = 1$. Since $f = v_1 + v_2$, both get $\bar v_1 = \bar v_2 = 1$. From $v_1 = a b$: $a$ receives $\bar v_1 \cdot b = 3$ and $b$ receives $\bar v_1 \cdot a = 2$. From $v_2 = \sin a$: $a$ receives another $\bar v_2 \cos a = -0.4161$. Adding up, $\bar a = 3 - 0.4161 = 2.5839$ and $\bar b = 2$.

Reverse mode delivered both partials in one sweep; forward mode needed one pass per input (with a million inputs: one pass versus a million). JAX agrees: `jax.grad(f, argnums=(0, 1))(2., 3.)` returns `(2.5838532, 2.0)`.

</div>

## Step 3: value_and_grad and auxiliary outputs

A training step needs the loss (to log) and the gradient (to update). Calling both `loss` and `jax.grad(loss)` runs the forward pass twice; `jax.value_and_grad` does it once. Metrics that should not be differentiated go in a second return value with `has_aux=True`:

```python
def loss_fn(p, X, Y):
    err = X @ p["w"] + p["b"] - Y
    return jnp.mean(err ** 2), {"mae": jnp.mean(jnp.abs(err))}   # (scalar, aux)

(l, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, X, Y)
print(l, aux)    # 22.916668 {'mae': 4.5}
```

With `has_aux=True` the function must return a pair, and only the first element is differentiated. Forget the flag and JAX complains that the output is a tuple rather than a scalar. The same `has_aux` flag exists on `grad`, `jacfwd`, `jacrev` and `vjp`.

## Step 4: forward and reverse mode, by name

Everything in JAX's autodiff is built on two primitives. Let $f$ map $n$ inputs to $m$ outputs, and let $J$ be its **Jacobian**: the $m \times n$ table of partial derivatives, with one row per output and one column per input.

- `jax.jvp(f, (x,), (v,))` returns $f(x)$ and $Jv$, the **Jacobian-vector product**. Here $v$ is a direction in input space (the "tangent"). This is forward mode: one pass, memory about the same as running $f$.
- `y, f_vjp = jax.vjp(f, x)` runs $f$ and returns a function; `f_vjp(u)` returns $u^\top J$, the **vector-Jacobian product**. Here $u$ is a weighting of the outputs (the "cotangent"). This is reverse mode: one pass per call, but the forward values must be kept in memory until the backward sweep.

```python
print(jax.jvp(f, (2., 3.), (1., 0.)))   # (6.9092975, 2.5838532)  value, J·v
y, f_vjp = jax.vjp(f, 2., 3.)
print(f_vjp(1.0))                       # (2.5838532, 2.0)        u·J, one entry per input
```

`grad` is simply `vjp` with $u = 1$ for a scalar output. To get a full Jacobian, JAX stacks many products, using `vmap` internally (lesson 05.04): `jacfwd` computes $n$ JVPs, one per column, and `jacrev` computes $m$ VJPs, one per row.

~~~dg-fig 05.02-modes.svg
A Jacobian is filled one column per forward pass or one row per reverse pass. For a tall Jacobian (few inputs) jacfwd needs only two passes; for a wide one (few outputs) jacrev does. A loss has one output, the widest case of all.
~~~

Rule: `jacfwd` when inputs are fewer ($n \ll m$), `jacrev` when outputs are fewer ($m \ll n$).

~~~dg-fig 05.02-timing.svg
Measured on this machine's CPU (timings are machine-dependent; compare the shapes, not the milliseconds). Left: the right mode is tens to a hundred times faster on each shape. Right: for a scalar loss, jacfwd's cost grows with the number of inputs while jax.grad stays nearly flat; at 3000 inputs it is hundreds of times slower.
~~~

A loss with millions of parameters and one output is the extreme reverse-mode case: "backpropagation" is just reverse-mode autodiff on a network.

## Step 5: derivatives of derivatives

Since `jax.grad(f)` is an ordinary JAX function, you can transform it again. Nesting gives higher derivatives with no new API:

```python
d1 = jax.grad(jnp.tanh)
d2 = jax.grad(d1)
print(d1(0.5), d2(0.5))      # 0.7864477 -0.72686195
```

~~~dg-fig 05.02-nested.svg
tanh and its first three derivatives, each made by wrapping the previous function in jax.grad once more (and plotted with vmap over a grid of x values).
~~~

The matrix of second derivatives is the **Hessian**, $H_{ij} = \partial^2 f / \partial x_i \partial x_j$. `jax.hessian(f)` is defined as `jacfwd(jacrev(f))`: reverse mode for the gradient (one output, many inputs), then forward mode over it (the gradient has as many outputs as inputs, so the square shape favours the cheaper-memory forward mode).

```python
def q(v):
    return v[0] ** 2 * v[1] + jnp.exp(v[1])

v0 = jnp.array([2., 3.])
print(jax.hessian(q)(v0))    # [[ 6.  4.]
                             #  [ 4. 20.085537]]
```

Check by hand: $\partial^2 q/\partial v_0^2 = 2 v_1 = 6$, the mixed term is $2 v_0 = 4$, and $\partial^2 q/\partial v_1^2 = e^{v_1} = e^3 = 20.09$.

A model with $n$ parameters has a Hessian with $n^2$ entries, which no real network can store. Most methods only need the **Hessian-vector product** $Hv$, and forward-over-reverse computes it without ever forming $H$:

```python
def hvp(f, x, v):
    return jax.jvp(jax.grad(f), (x,), (v,))[1]

print(hvp(q, v0, jnp.array([1., 0.])))   # [6. 4.]  = first column of H
```

The cost is a small constant multiple of one gradient (roughly two to three), and memory stays near that of a gradient. It drives power iteration for the top Hessian eigenvalue, Newton-CG and influence functions; for pytree `x`, `v` has the same structure.

## Step 6: controlling the backward pass

`jax.lax.stop_gradient(z)` returns `z` unchanged in the forward pass but has zero derivative, the JAX counterpart of PyTorch's `.detach()`:

```python
def g(w):
    return w ** 2 + jax.lax.stop_gradient(w ** 2)

print(g(3.), jax.grad(g)(3.))    # 18.0 6.0   (the second term is invisible to grad)
```

Use it for targets that must not receive gradients (a Q-learning target network, an EMA teacher).

Sometimes the correct gradient is useless for learning. Rounding has derivative zero almost everywhere, so quantisation-aware training would never learn. A **straight-through estimator** rounds going forward and acts as the identity going backward. `jax.custom_vjp` lets you write exactly that:

```python
@jax.custom_vjp
def round_ste(x):
    return jnp.round(x)

def round_fwd(x):
    return jnp.round(x), None        # (output, residuals saved for backward)

def round_bwd(res, g):
    return (g,)                      # one cotangent per input, as a tuple

round_ste.defvjp(round_fwd, round_bwd)
print(jax.grad(lambda x: 2 * jnp.round(x))(1.3),
      jax.grad(lambda x: 2 * round_ste(x))(1.3))   # 0.0 2.0
```

~~~dg-fig 05.02-custom.svg
Left: rounding's true gradient is zero, while the straight-through rule passes gradient 1. Right: written naively, log(1 + exp(x)) has a gradient that collapses to 0 near x = 88 and becomes NaN from 89 in float32; jax.nn.softplus uses a stable rule and stays at 1.
~~~

Custom rules also fix unstable gradients (right panel); prefer library functions such as `jax.nn.softplus` and `jax.nn.logsumexp`, which already carry stable rules. `jax.custom_jvp` is the forward-mode counterpart, and it is preferred whenever a JVP rule is enough, because JAX can derive reverse mode from it automatically (see Going deeper), so the function supports both.

## Autodiff in a training loop

Putting the pieces together, here is a complete linear-regression fit. `jax.jit` (lesson 05.03) compiles the step; `jax.tree.map` applies the update leaf by leaf to the parameter dict.

```python
import numpy as np
rng = np.random.default_rng(0)
Xt = rng.normal(size=(200, 3)).astype(np.float32)
Yt = Xt @ np.array([2., -1., 0.5], np.float32) + 1.0 + 0.1 * rng.normal(size=200).astype(np.float32)

p = {"w": jnp.zeros(3), "b": jnp.array(0.)}
step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
for i in range(101):
    (l, aux), g = step(p, Xt, Yt)
    p = jax.tree.map(lambda a, b: a - 0.1 * b, p, g)   # gradient descent, lr 0.1
    if i % 25 == 0:
        print(i, float(l), float(aux["mae"]))
# 0 5.883 1.933
# 25 0.01027 0.0810
# 100 0.01000 0.0803
print(p)   # w ≈ [1.991, -1.010, 0.505], b ≈ 0.996
```

The loss settles at 0.0100, the variance of the added noise ($0.1^2$), and the weights recover the true $(2, -1, 0.5)$ and bias $1$.

## Checking your gradients

A gradient that runs can still be wrong. Three habits catch almost everything:

- **Compare with finite differences:** `from jax.test_util import check_grads`, then `check_grads(f, (x,), order=2)` compares first and second derivatives in both modes with numerical estimates and raises on disagreement. Run it on every custom rule.
- **Check every parameter group is non-zero** on a small batch. An all-zero layer gradient means a disconnected path: a stray `stop_gradient`, an integer cast, an `argmax`.
- **Check the structure** matches the parameters; a tuple where you expected a dict usually means the wrong `argnums`.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| `TypeError: grad requires real- or complex-valued inputs` | an integer argument such as `1` instead of `1.0` | pass floats, or mark the argument non-differentiable via `argnums`
| `TypeError: Gradient only defined for scalar-output functions` | the function returns a vector, or a (loss, metrics) tuple | reduce with `mean`/`sum`, or add `has_aux=True`
| Gradient silently exactly 0 | `.astype(jnp.int32)`, `round`, `argmax` or `stop_gradient` on the path | remove the cast, or use a straight-through rule
| NaN gradient although the forward value is fine | `jnp.where(x > 0, jnp.sqrt(x), 0.)`: the unused branch still has an infinite derivative at 0, and 0 × inf = NaN | apply `where` to the input too ("double where"), so `sqrt` never sees 0
| NaN or 0 gradient for large inputs | overflow inside a hand-written formula such as `log(1 + exp(x))` | use `jax.nn.softplus`, `logsumexp`, or a `custom_jvp`
| `TracerBoolConversionError` | Python `if x > 0` on a traced value inside `jit` | plain `grad` handles it, but under `jit` use `jnp.where` or `lax.cond`
| Very slow Jacobian | `jacfwd` on a wide function or `jacrev` on a tall one | switch to the mode matching the shape

For the `where` trap, computing `xs = jnp.where(x > 0, x, 1.)` and taking `jnp.sqrt(xs)` turns the `nan` at 0 into the expected `0.0`. At kinks JAX returns a convention, not an error: `grad(jnp.abs)(0.)` is `1.0`, `grad(jax.nn.relu)(0.)` is `0.0`, and `jnp.max` splits its gradient among ties.

## Going deeper

**The cheap gradient principle.** Reverse mode gives the full gradient of a scalar function for a small constant multiple (typically two to four) of the cost of evaluating it, whatever the number of inputs. This is why training billion-parameter models is feasible.

**Memory is the price of reverse mode.** The backward sweep needs the forward pass's intermediate values (the **residuals**), so memory grows with depth. `jax.checkpoint` (alias `jax.remat`) discards a block's intermediates and recomputes them during the backward pass, trading compute for memory. Forward mode has no such cost.

**JAX derives reverse mode from forward mode.** Internally, `vjp` is built as `jax.linearize` (which evaluates the function and returns its JVP as a linear function) followed by *transposition* of that linear function. Every primitive therefore only needs a JVP rule and a transpose rule. This is why a `custom_jvp` gives you both modes for free, while a `custom_vjp` only supports reverse mode: a function defined with `custom_vjp` cannot be used inside `jvp` or `jacfwd`.

**Why forward-over-reverse for HVPs.** Reverse-over-reverse (`grad` of $\langle \nabla f, v\rangle$) also works, but its outer reverse pass must store the inner backward pass; an outer forward pass adds little memory.

**Control flow.** Without `jit`, `grad` follows whatever Python branches actually ran, so a function with `if` and `for` on concrete values differentiates fine (`grad` of `x**2 if x > 0 else -x` gives `6.0` at 3 and `-1.0` at −2). Under `jit`, values are abstract and you need `lax.cond`, `lax.scan` or `lax.fori_loop`, all of which have derivative rules (`fori_loop` supports reverse mode only when the trip count is fixed).

~~~dg-flow How the derivative tools are built from the two primitives.
jvp -(vmap over inputs)-> jacfwd
vjp -(cotangent 1)-> grad -(jvp of grad)-> *HVP
vjp -(vmap over outputs)-> jacrev -(jacfwd over it)-> hessian
~~~

## Key terms

| Term | Meaning
| --- | --- |
| Automatic differentiation | Exact derivatives of a program, from the chain rule applied to each primitive it runs
| Function transformation | An operation that takes a function and returns a new one, like `grad`, `jit` or `vmap`
| Jacobian | The table of partial derivatives of every output with respect to every input
| JVP (forward mode) | Jacobian times an input direction; one pass per input direction
| VJP (reverse mode) | Output weighting times the Jacobian; one pass per output, needs stored residuals
| `has_aux` | Flag that lets the function return extra values that are not differentiated
| Hessian-vector product | $Hv$ computed as the JVP of the gradient, without forming $H$
| `stop_gradient` | Identity in the forward pass, zero derivative in the backward pass
| Straight-through estimator | A custom backward rule that treats a non-differentiable step as the identity

<div class="take">
<div class="lab">Key takeaway</div>

in JAX a derivative is just another function. Use `value_and_grad` with `has_aux` for training, pick `jacfwd` or `jacrev` from the shape of the Jacobian, compose `jvp` with `grad` for Hessian-vector products, and reach for `custom_vjp` or `custom_jvp` only when the default rule is wrong or unstable for your purpose.

</div>

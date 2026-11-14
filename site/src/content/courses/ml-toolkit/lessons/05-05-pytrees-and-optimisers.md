---
id: "05.05"
course: "ml-toolkit"
module: "05"
slug: "pytrees-and-optimisers"
title: "Pytrees and optimisers"
keyIdeas: ["A **pytree** is any nesting of lists, tuples, dicts, `None` and registered classes, with arrays (or other non-container objects) as leaves. Every JAX transformation flattens it into leaves plus a treedef, works on the leaves and rebuilds the same structure, which is how a model with hundreds of parameter arrays passes through `grad`, `jit` and `vmap`.", "`jax.tree.map(f, tree, *rest)` applies `f` leaf by leaf and rebuilds the same structure. With several trees, they must have the same structure, and `f` receives the matching leaves together; dict leaves are visited in sorted key order.", "A model's parameters, its gradients and each slot of optimiser state are pytrees of the same structure, so SGD with momentum or Adam is a handful of `tree.map` calls, and the whole training step can sit inside one `jit`.", "Adam is a pure function `(grads, state) -> (updates, new_state)` with `m`, `v` and a step count in the state; bias correction $1-\\beta^t$ stops the early steps from being several times too large. Optax packages this interface as `GradientTransformation(init, update)` and lets you `chain` clipping, masked weight decay and schedules.", "`None` is an empty subtree, not a leaf; custom classes must be registered (for example with `jax.tree_util.register_dataclass`) before transformations can see inside them; and non-trainable settings such as a dropout rate do not belong in the parameter tree."]
question: "Your training state is `(params, opt_state)` and you checkpoint it by flattening with `jax.tree.leaves` and saving the list of arrays. After a refactor that renamed `'mlp'` to `'ffn'` in the parameter dict, restoring the checkpoint with `jax.tree.unflatten(treedef, leaves)` succeeds, but the loss starts much higher than when you saved. What happened, and how should the checkpoint be stored instead?"
answer: "Dict leaves are flattened in sorted key order, so renaming `'mlp'` to `'ffn'` moves that subtree to a different position in the flattened list (for instance from after `'ln'` to before it, because 'f' sorts before 'l' while 'm' sorts after it). The saved list is still in the old order, and `unflatten` only checks that the number of leaves matches, not their names or shapes. If several subtrees have the same structure and leaf shapes (for example the MLP weights and another block with matching shapes), unflattening the old list into the new treedef assigns arrays to the wrong parameters without any error, and the Adam moments in `opt_state` get scrambled in the same way, so even the optimiser's per-parameter scaling is wrong. The model then starts from a partly permuted set of weights, so the loss jumps; if shapes had differed you would at best get an error later in the forward pass. A positional list of leaves is not a safe checkpoint format. Store leaves keyed by their path, for example from `jax.tree_util.tree_flatten_with_path` with `keystr` names, or use a checkpointing library such as Orbax that saves the tree structure with names, then restore by name and fail loudly on any key that is missing or unexpected. A rename should be handled by an explicit key mapping during restore (old name to new name), applied to both `params` and every tree inside `opt_state`, followed by a check that `jax.tree.structure` of the restored state equals that of a freshly initialised one."
sources: [{"title": "JAX docs: Pytrees", "url": "https://docs.jax.dev/en/latest/101/pytrees.html", "note": "What counts as a node or leaf, tree.map over several trees, key paths and custom registration."}, {"title": "Optax documentation", "url": "https://optax.readthedocs.io/en/latest/", "note": "GradientTransformation, chain, schedules, and reference implementations of Adam and AdamW."}, {"title": "JAX API: jax.tree module", "url": "https://docs.jax.dev/en/latest/jax.tree.html", "note": "The jax.tree functions (map, leaves, structure, flatten, unflatten)."}]
practice: ["sgd-with-momentum-over-pytree", "adamw", "global-norm-gradient-clipping", "gradient-accumulation-with-scan"]
tags: ["pytrees", "optimisers", "Optax"]
tools: ["JAX", "Optax"]
---

## What you will learn

- What a **pytree** is, and why `grad`, `jit` and `vmap` can take a whole model's parameters as one argument.
- The tree utilities you will use daily: `leaves`, `structure`, `flatten`/`unflatten`, `map`, and key paths.
- How an optimiser becomes a pure function over trees: SGD with momentum and Adam by hand, bias correction included.
- How Optax packages the same idea as `GradientTransformation(init, update)` and chains clipping, weight decay and schedules.
- The usual traps: mismatched structures, unregistered classes, stray floats in the parameter tree, positional checkpoints.

*Before you start:* you should be comfortable with `jax.grad` (05.02), `jax.jit` (05.03) and the PyTorch optimiser loop (04.03, 04.04), which this lesson re-creates without objects.

## From one array to a whole model

Lesson 05.04 mapped functions of one weight matrix over a batch. Real models have hundreds of arrays. PyTorch hides them inside `nn.Module` objects; JAX has no module system in its core, so parameters are plain data, usually nested dicts and lists of arrays, passed as an ordinary argument. How can `jax.grad` differentiate with respect to "a dict of lists of dicts"? The answer is the pytree, and once you see it, optimisers become a few lines of code.

## The idea in one picture

Here is the parameter set of a tiny two-layer network:

```python
import jax, jax.numpy as jnp

params = {"layer1": {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)},
          "layer2": {"w": jnp.ones((3, 1)), "b": jnp.zeros(1)}}

leaves, treedef = jax.tree.flatten(params)
print([x.shape for x in leaves])   # [(3,), (2, 3), (1,), (3, 1)]
print(treedef)    # PyTreeDef({'layer1': {'b': *, 'w': *}, 'layer2': {'b': *, 'w': *}})
print(sum(x.size for x in leaves)) # 13 parameters in total
```

~~~dg-fig 05.05-tree.svg
The dicts are nodes, the arrays are leaves. Flattening walks the tree and returns the leaves as a list (dict keys in sorted order, so b comes before w) plus a treedef that remembers the nesting.
~~~

A **pytree** is any nesting of containers (dicts, lists, tuples, `None`, registered classes) whose ends are other objects, usually arrays. Containers are **nodes**, the ends are **leaves**, and the **treedef** (`PyTreeDef`) is the nesting with leaves replaced by `*`.

Every JAX transformation flattens a pytree argument into leaves plus a treedef, works on the flat list of arrays, and unflattens the results with the same treedef. That is why `jax.grad(loss)(params, X, y)` returns a gradient with exactly the keys and nesting of `params`.

~~~dg-flow What every transformation does with a pytree argument.
params tree -(flatten)-> leaves + treedef -(transform each leaf)-> *new leaves -(unflatten)-> tree of same shape
~~~

## Step 1: what counts as a node and what counts as a leaf

The rules are short, but they decide what JAX can see inside.

| object | node or leaf | note
| --- | --- | --- |
| `dict`, `list`, `tuple`, `namedtuple` | node | dict children are ordered by sorted key
| `None` | node with no children | contributes zero leaves
| `jax.Array`, NumPy array, Python scalar | leaf | the usual case
| your own class | leaf, unless registered | register with `register_dataclass` or `register_pytree_node`

Two rows deserve a second look. `None` is not a leaf: `jax.tree.leaves({"a": 1, "b": None, "c": (2, 3)})` returns `[1, 2, 3]`, which lets optimiser state say "nothing here" for a frozen parameter. And an unregistered class is one opaque leaf: `tree.map` hands your function the whole object and `jit` refuses it ("Error interpreting argument ... as an abstract array"). Flax and Equinox register their module classes so they behave like dicts of arrays.

Sorted keys make the leaf order deterministic, but the order changes when you rename a key, which matters for checkpoints (see the question at the end).

## Step 2: the tree utilities

Everything you will do with parameter trees is built from a few functions in the `jax.tree` namespace:

```python
jax.tree.leaves(params)                    # list of arrays, in a fixed order
jax.tree.structure(params)                 # PyTreeDef, comparable with ==
jax.tree.unflatten(treedef, new_leaves)    # rebuild a tree from a list
jax.tree.map(jnp.zeros_like, params)       # same structure, zero leaves
jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)   # two trees, leaf by leaf
print(jax.tree.map(jnp.shape, params))
# {'layer1': {'b': (3,), 'w': (2, 3)}, 'layer2': {'b': (1,), 'w': (3, 1)}}
```

`jax.tree.map(f, tree)` calls `f` once per leaf and rebuilds the same structure. With several trees, `f` receives the matching leaves of all of them together, so the trees must have the same structure (strictly, the first must be a *prefix* of the others).

~~~dg-fig 05.05-treemap.svg
One step of gradient descent as a tree.map over two trees with the same structure. The b leaves meet each other, the w leaves meet each other, and the result is a new tree of the same shape.
~~~

If the structures disagree you get an error, not a silent mismatch: mapping over `{"a": 1, "b": None}` and `{"a": 1, "b": 2}` raises `ValueError: pytree structure error: different types at key path tree_map tree['b']`. Treat it as a useful check that parameters and optimiser state no longer line up.

The `jax.tree` functions are the modern spelling of `jax.tree_util.tree_map`, `tree_leaves` and friends, which still work. `jax.tree_util` also has **key-path** versions that pass each leaf's location along with it:

```python
for path, x in jax.tree_util.tree_flatten_with_path(params)[0]:
    print(jax.tree_util.keystr(path), x.shape)   # ['layer1']['b'] (3,)  ... 4 lines
```

Paths are how you say "this rule applies to biases only" without relying on leaf positions.

## Step 3: gradients and optimiser state are trees of the same shape

The observation that makes everything easy: if the parameters are a pytree, then

- the gradient from `jax.grad` has the same structure;
- momentum's velocity has one slot per parameter, so it has the same structure too;
- Adam's two moment estimates are two more trees of that structure.

So an optimiser update is a few `tree.map` calls. SGD with **momentum** keeps a running velocity $v$:

$$v_t = \beta\, v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t.$$

$\theta$ is a parameter, $g_t$ its gradient at step $t$, $\eta$ the learning rate, and $\beta$ (typically 0.9) how much old velocity survives. In code, every symbol is a whole tree:

```python
def sgd_momentum(params, grads, vel, lr=0.1, beta=0.9):
    vel = jax.tree.map(lambda v, g: beta * v + g, vel, grads)
    params = jax.tree.map(lambda p, v: p - lr * v, params, vel)
    return params, vel

vel = jax.tree.map(jnp.zeros_like, params)   # initial state: zeros, same shape
```

Nothing here knows how many layers there are; the loop over arrays is hidden inside `tree.map`, so the same code serves 13 parameters or a billion.

## Step 4: Adam as a pure function

**Adam** keeps two running averages per parameter: $m$, of recent gradients (momentum), and $v$, of recent *squared* gradients (how large that parameter's gradients usually are). Dividing one by the square root of the other makes every parameter move by roughly the learning rate, whatever its raw gradient scale:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,$$

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$ to prevent division by zero. The hats are the **bias correction**. Both averages start at zero, so early on they are pulled towards zero; dividing by $1-\beta^t$ undoes exactly that pull. Since $\beta_2$ is much closer to 1, $v$ is pulled down far more than $m$, so without correction early steps are too large, not too small.

~~~dg-fig 05.05-bias.svg
Adam with a constant gradient. Without bias correction the step starts at 3.2 times the learning rate and peaks at 6.5 times around step 10, settling only after a few thousand steps. With correction every step is exactly the learning rate.
~~~

In the update-returning style Optax also uses, Adam is an `init` that builds the state and an `update` from (gradients, state) to (updates, new state):

```python
def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.zeros((), jnp.int32)}

def adam_update(grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mhat_c, vhat_c = 1 - b1 ** t, 1 - b2 ** t          # bias corrections
    updates = jax.tree.map(lambda m, v: -lr * (m / mhat_c) / (jnp.sqrt(v / vhat_c) + eps), m, v)
    return updates, {"m": m, "v": v, "t": t}

updates, opt_state = adam_update(grads, opt_state)
params = jax.tree.map(lambda p, u: p + u, params, updates)
```

There is no in-place mutation and no hidden counter: state goes in, new state comes out, so the whole step, optimiser included, can sit inside one `jit`. The figure shows what the state buys on a narrow valley, a stand-in for the badly scaled directions of real losses.

~~~dg-fig 05.05-paths.svg
Forty steps on the valley 0.5(x² + 30y²). Plain SGD must use a learning rate small enough for the steep direction, so it zigzags across it and crawls along the gentle one. Momentum averages the zigzag away; Adam rescales each coordinate so both directions move at a similar pace.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two Adam steps by hand**

One parameter $\theta = 1.0$, learning rate $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and gradients $g_1 = 0.5$ then $g_2 = 0.3$. Ignore $\epsilon$, which is far too small to matter here.

Step 1. $m_1 = 0.9 \times 0 + 0.1 \times 0.5 = 0.05$. $v_1 = 0.999 \times 0 + 0.001 \times 0.25 = 0.00025$.

Bias corrections: $1 - 0.9^1 = 0.1$ and $1 - 0.999^1 = 0.001$. So $\hat m_1 = 0.05 / 0.1 = 0.5$ and $\hat v_1 = 0.00025 / 0.001 = 0.25$, with $\sqrt{0.25} = 0.5$.

Update $= -0.1 \times 0.5 / 0.5 = -0.1$, so $\theta_1 = 0.9$. The first Adam step always has size $\eta$.

Without correction the step would have been $-0.1 \times 0.05 / \sqrt{0.00025} = -0.1 \times 0.05 / 0.01581 = -0.316$, more than three times too big.

Step 2. $m_2 = 0.9 \times 0.05 + 0.1 \times 0.3 = 0.045 + 0.03 = 0.075$. $v_2 = 0.999 \times 0.00025 + 0.001 \times 0.09 = 0.00024975 + 0.00009 = 0.00033975$.

Corrections: $1 - 0.9^2 = 0.19$ and $1 - 0.999^2 = 0.001999$. So $\hat m_2 = 0.075 / 0.19 = 0.3947$ and $\hat v_2 = 0.00033975 / 0.001999 = 0.16996$, with square root $0.4123$.

Update $= -0.1 \times 0.3947 / 0.4123 = -0.0957$, so $\theta_2 = 0.9 - 0.0957 = 0.8043$. Running `adam_update` above and `optax.adam(0.1)` both give 0.80425.

</div>

## Step 5: clipping, weight decay and schedules

Production training adds three things to bare Adam, each a small function over trees.

**Global-norm clipping** protects against the occasional huge gradient. Treat all leaves together as one long vector, measure its length $\|g\| = \sqrt{\sum_\ell \|g_\ell\|^2}$ (the sum runs over leaves $\ell$), and if it exceeds a threshold $c$, shrink every leaf by the same factor:

$$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|}\right).$$

Every leaf gets the same factor, so the direction is unchanged and only the length is capped:

```python
def clip_by_global_norm(grads, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads), norm

g = {"w": jnp.array([[3., 0.], [0., 0.]]), "b": jnp.array([4., 0.])}
clipped, norm = clip_by_global_norm(g, 1.0)
# norm = 5.0;  clipped["w"][0, 0] = 0.6,  clipped["b"][0] = 0.8
```

The norm is $\sqrt{3^2 + 4^2} = 5$, so every entry is multiplied by $1/5$. Clipping each leaf separately would bend the update's direction.

**Decoupled weight decay** (AdamW) adds $-\eta \lambda \theta$ to each update, with decay strength $\lambda$ (often 0.01 to 0.1). It usually skips biases and normalisation scales, expressed as a **mask**: a tree of booleans mirroring the parameters.

**Learning-rate schedules** change $\eta$ with the step: a short linear **warmup** (early second-moment estimates are noisy), then a slow decay, often a cosine.

~~~dg-fig 05.05-schedule.svg
Left: optax.warmup_cosine_decay_schedule(0, 3e-4, 1000, 100000) rises linearly for the first 1,000 steps, then follows a cosine down to zero. Right: global-norm clipping shrinks long gradients onto the circle of radius 1 without changing their direction; a short gradient passes untouched.
~~~

## Optimisers in code with Optax

**Optax** is the standard optimiser library for JAX and follows exactly the design above. An optimiser is a `GradientTransformation`, a pair of pure functions: `init(params) -> state` and `update(grads, state, params) -> (updates, state)`. `optax.apply_updates(params, updates)` adds the updates to the parameters. A complete training loop for a small MLP fitting $y = \sin(2x)$ plus noise:

```python
import jax, jax.numpy as jnp, optax

kx, kn, kp = jax.random.split(jax.random.key(0), 3)
X = jax.random.uniform(kx, (256, 1), minval=-2, maxval=2)
y = jnp.sin(2 * X[:, 0]) + 0.1 * jax.random.normal(kn, (256,))

def init(key, sizes=(1, 32, 32, 1)):
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"w": jax.random.normal(k, (m, n)) / jnp.sqrt(m), "b": jnp.zeros(n)}
            for k, m, n in zip(keys, sizes[:-1], sizes[1:])]

def loss(params, X, y):
    h = X
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    out = h @ params[-1]["w"] + params[-1]["b"]
    return jnp.mean((out[:, 0] - y) ** 2)

params = init(kp)
print(sum(x.size for x in jax.tree.leaves(params)))   # 1153

tx = optax.adam(learning_rate=0.01)
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, X, y):
    l, grads = jax.value_and_grad(loss)(params, X, y)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, l

for step in range(300):
    params, opt_state, l = train_step(params, opt_state, X, y)
    if step in (0, 100, 299):
        print(step, float(l))   # 0 0.3549   100 0.0148   299 0.0104
print(jax.tree.map(jnp.shape, opt_state[0].mu)[1])
# {'b': (32,), 'w': (32, 32)}   Adam's first moment mirrors the params tree
```

The parameters are a *list* of dicts, and Optax does not care. `opt_state[0]` is a `ScaleByAdamState` named tuple whose `mu` and `nu` mirror `params`, plus a step `count`: our hand-written `{"m", "v", "t"}` under other names.

Transformations compose with `optax.chain`, which runs them in order and keeps a tuple of their states:

```python
def decay_mask(params):          # True for weight matrices, False for biases
    return jax.tree_util.tree_map_with_path(lambda path, x: path[-1].key == "w", params)

tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.0, 3e-4, 1_000, 100_000),
                weight_decay=0.01, mask=decay_mask),
)
opt_state = tx.init(params)
updates, opt_state = tx.update(grads, opt_state, params)   # params needed for weight decay
params = optax.apply_updates(params, updates)
```

Write Adam yourself once to understand the state; in real code use Optax, which handles dtypes, masks and schedules consistently behind one interface.

## Checking that it works

**Compare against a reference.** Run your optimiser and the Optax one from the same start. In the run behind the next figure, `adam_update` and `optax.adam` agreed to a relative $3 \times 10^{-7}$ over the first 100 steps (float32 rounding). Later the curves may drift slightly, because training amplifies tiny rounding differences, so early agreement is the meaningful test.

~~~dg-fig 05.05-train.svg
Left: the same 1,153-parameter MLP trained with SGD, SGD with momentum and Adam; the dashed hand-written Adam lies on top of optax.adam. Plain SGD at this learning rate keeps spiking. Right: measured time per training step on this CPU, run eagerly versus with gradient and optimiser inside one jit (timings are machine-dependent).
~~~

**Check structures.** `jax.tree.structure(grads) == jax.tree.structure(params)` should be `True`, and `jax.tree.map(jnp.shape, opt_state)` shows what is stored (for Adam, about twice the parameter count).

**Log the global gradient norm.** `optax.global_norm(grads)` gives one number per step; sudden spikes often precede loss spikes and show whether clipping is active.

**Watch the step time.** Eagerly, each `tree.map` dispatches small operations from Python; inside one `jit`, XLA fuses everything into one program, roughly 50 times faster in the run shown (50 to 80 times across runs here).

## What goes wrong

| symptom | cause | fix
| --- | --- | --- |
| `ValueError: pytree structure error` in `tree.map` | params and state trees differ (extra key, `None` versus array, list versus tuple) | build state with `jax.tree.map(..., params)` so it mirrors params; print both structures
| `jit` rejects an argument of your class | unregistered class is an opaque leaf | `jax.tree_util.register_dataclass` or `register_pytree_node`
| `AttributeError: 'float' object has no attribute 'ndim'` | a Python float stored inside the params tree | keep configuration out of params (see example below)
| early hand-written Adam steps far too large | bias correction missing, or `t` = 0 used | increment `t` before computing $1-\beta^t$
| biases and norm scales shrink towards zero | weight decay applied to every leaf | pass a `mask` built from key paths
| restored model has high loss, no error | checkpoint saved as a positional list of leaves | save with key paths (Orbax), restore by name
| step is slow even though `grad` is jitted | optimiser update runs eagerly outside `jit` | put the whole step (grad, update, apply) in one `jit`

<div class="ex">
<div class="lab">Worked example</div>

**a float in the parameter tree**

A weight-decay mask is built as `jax.tree.map(lambda x: x.ndim > 1, params)`, so matrices are decayed and vectors (biases, norm scales) are not.

Later someone stores a dropout rate as a Python float inside `params`. It becomes a leaf, `x.ndim` fails on a float, and worse, `grad` differentiates with respect to it: in a quick test `jax.grad` returned 4.0 for a `"drop": 0.1` entry, so the optimiser would "train" the dropout rate.

Keep non-trainable configuration out of the parameter tree (pass it as a static argument or close over it), and build masks from key paths with `tree_map_with_path` so the intent ("decay everything except names ending in `b` or `scale`") is explicit.

</div>

## Going deeper

**Registering your own classes.** `data_fields` become children (traced, differentiated); `meta_fields` are static parts of the treedef, so changing them recompiles under `jit`:

```python
import dataclasses, functools

@functools.partial(jax.tree_util.register_dataclass, data_fields=["w", "b"], meta_fields=["act"])
@dataclasses.dataclass
class Dense:
    w: jax.Array
    b: jax.Array
    act: str = "tanh"

layer = Dense(jnp.ones((2, 3)), jnp.zeros(3))
print(jax.tree.structure(layer))   # PyTreeDef(CustomNode(Dense[('tanh',)], [*, *]))
```

The general form, `register_pytree_node(cls, flatten, unflatten)`, takes functions that split an object into children plus static data and rebuild it; Flax and Equinox modules plug into `grad` and `jit` this way.

**Optax state is a pytree too.** `optax.chain(clip, adamw(schedule))` produces a nested tuple: `EmptyState` for clipping, `ScaleByAdamState` for the moments, `ScaleByScheduleState` for the step count. So it passes through `jit`, shards with the parameters (05.07), and with `donate_argnums=(0, 1)` XLA can reuse the old buffers for the new state.

**Gradient accumulation.** To simulate a batch larger than memory allows, average gradients over micro-batches; with `lax.scan` the accumulator is a carried tree:

```python
def accum_grads(params, micro_batches):           # micro_batches: (k, B, ...)
    def body(acc, mb):
        g = jax.grad(loss_fn)(params, mb)
        return jax.tree.map(jnp.add, acc, g), None
    acc, _ = jax.lax.scan(body, jax.tree.map(jnp.zeros_like, params), micro_batches)
    return jax.tree.map(lambda g: g / micro_batches.shape[0], acc)
```

For a mean loss with equal micro-batches this matches the full-batch gradient (a four-micro-batch test differed by 0.0). `optax.MultiSteps(tx, every_k_schedule=k)` does the same inside its state, emitting zero updates in between.

**Why Adam behaves as it does.** For a consistent gradient, $\hat m / \sqrt{\hat v}$ is close to its sign, so each coordinate moves about $\eta$ per step whatever its gradient scale. That flattens the valley in the paths figure, and is why Adam's learning rate is set in parameter units (1e-3, 3e-4). It also explains why an L2 penalty in the loss differs from AdamW's decay: the L2 term enters the gradient and is then divided by $\sqrt{\hat v}$, so weights with large gradients get less decay. Decoupling restores a uniform pull.

## Key terms

| Term | Meaning
| --- | --- |
| Pytree | Any nesting of containers (dict, list, tuple, None, registered classes) with leaves at the ends
| Node / leaf | A container JAX looks inside / an object (usually an array) it does not
| Treedef (`PyTreeDef`) | The structure of a pytree with leaves replaced by placeholders; comparable with ==
| `jax.tree.map` | Applies a function leaf by leaf across one or more trees of the same structure
| Key path | The sequence of keys and indices leading to a leaf, used for masks and named checkpoints
| Optimiser state | Per-parameter running quantities (velocity, Adam moments) stored as trees mirroring the params
| Bias correction | Dividing Adam's moments by $1-\beta^t$ to undo their zero initialisation
| `GradientTransformation` | Optax's pair of pure functions `init(params)` and `update(grads, state, params)`

<div class="take">
<div class="lab">Key takeaway</div>

in JAX a model is a pytree of arrays and an optimiser is a pure function from gradient and state trees to update and state trees. Learn `jax.tree.map`, keep the structures aligned, build masks from key paths, put the whole step inside one `jit`, and let Optax supply the production versions.

</div>

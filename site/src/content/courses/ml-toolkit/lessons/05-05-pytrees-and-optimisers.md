---
id: "05.05"
course: "ml-toolkit"
module: "05"
slug: "pytrees-and-optimisers"
title: "Pytrees and optimisers"
keyIdeas: ["A **pytree** is any nesting of lists, tuples, dicts, `None` and registered classes, with arrays (or other non-container objects) as leaves. Every JAX transformation accepts and returns pytrees, which is how a model with hundreds of parameter arrays passes through `grad`, `jit` and `vmap`.", "`jax.tree.map(f, tree, *rest)` applies `f` leaf by leaf and rebuilds the same structure. With several trees, they must have the same structure, and `f` receives the matching leaves together.", "A model's parameters, its gradients and each slot of optimiser state are pytrees of the same structure, so an optimiser update is a handful of `tree.map` calls.", "Adam is a pure function `(grads, state) -> (updates, new_state)` with `m`, `v` and a step count in the state. Optax packages exactly this interface as `GradientTransformation(init, update)` and lets you `chain` clipping, weight decay and scheduling.", "`None` is an empty subtree, not a leaf, and custom classes must be registered (for example with `jax.tree_util.register_dataclass`) before transformations can see inside them."]
question: "Your training state is `(params, opt_state)` and you checkpoint it by flattening with `jax.tree.leaves` and saving the list of arrays. After a refactor that renamed `'mlp'` to `'ffn'` in the parameter dict, restoring the checkpoint with `jax.tree.unflatten(treedef, leaves)` succeeds, but the loss starts much higher than when you saved. What happened, and how should the checkpoint be stored instead?"
answer: "Dict leaves are flattened in sorted key order, so renaming `'mlp'` to `'ffn'` moves that subtree to a different position in the flattened list (for instance from after `'ln'` to before it). The saved list is still in the old order. If several subtrees have the same structure and leaf shapes (for example the MLP weights and another block with matching shapes), unflattening the old list into the new treedef assigns arrays to the wrong parameters without any shape error, and the Adam moments get scrambled in the same way. The model then starts from a partly permuted set of weights, so the loss jumps. A positional list of leaves is not a safe checkpoint format. Store leaves keyed by their path, for example from `jax.tree_util.tree_flatten_with_path`, or use a checkpointing library such as Orbax that saves the tree structure with names, then restore by name and fail loudly on any key that is missing or unexpected. A rename should be handled by an explicit key mapping during restore."
sources: [{"title": "JAX docs: Pytrees", "url": "https://docs.jax.dev/en/latest/101/pytrees.html", "note": "What counts as a node or leaf, tree.map over several trees, key paths and custom registration."}, {"title": "Optax documentation", "url": "https://optax.readthedocs.io/en/latest/", "note": "GradientTransformation, chain, schedules, and reference implementations of Adam and AdamW."}, {"title": "JAX API: jax.tree module", "url": "https://docs.jax.dev/en/latest/jax.tree.html", "note": "The jax.tree functions (map, leaves, structure, flatten, unflatten)."}]
practice: ["sgd-with-momentum-over-pytree", "adamw", "global-norm-gradient-clipping", "gradient-accumulation-with-scan"]
tags: ["pytrees", "optimisers", "Optax"]
tools: ["JAX", "Optax"]
---

## Containers all the way down

In PyTorch, parameters live inside `nn.Module` objects and the optimiser holds references to them (lessons 04.03 and 04.04). JAX has no module system in the core library. Parameters are plain data, usually nested dicts of arrays, and functions take them as arguments:

```python
params = {
    "embed": {"table": jnp.zeros((V, D))},
    "layers": [{"w": jnp.zeros((D, D)), "b": jnp.zeros(D)} for _ in range(4)],
    "head": {"w": jnp.zeros((D, V))},
}
```

JAX calls such a structure a pytree. Lists, tuples, dicts, `namedtuple`s and `None` are built-in **nodes**; everything else, typically arrays and scalars, is a **leaf**. Transformations flatten a pytree into a list of leaves plus a description of the structure (a `PyTreeDef`), work on the leaves, and unflatten the result. That is why `jax.grad(loss)(params, batch)` returns a gradient with exactly the keys and nesting of `params`.

~~~dg-flow What every transformation does with a pytree argument.
params tree -(flatten)-> leaves + treedef -(transform each leaf)-> *new leaves -(unflatten)-> tree of same shape
~~~

## The tree utilities

```python
jax.tree.leaves(params)            # list of arrays, in a fixed order
jax.tree.structure(params)         # PyTreeDef, comparable with ==
jax.tree.map(jnp.zeros_like, params)                 # same structure, zero leaves
jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)  # two trees, leaf by leaf
n_params = sum(x.size for x in jax.tree.leaves(params))
```

The `jax.tree` namespace is the modern spelling of the older `jax.tree_util.tree_map`, `tree_leaves` and so on, which still work. `jax.tree_util.tree_flatten_with_path` and `tree_map_with_path` also hand you each leaf's key path, which is how you build masks such as "no weight decay on biases or norm scales".

Dict leaves are ordered by sorted key, not insertion order, so the leaf order is deterministic. Multi-tree `map` requires identical structures: passing a `params` tree and an optimiser-state tree that has an extra `None` somewhere fails with a structure-mismatch error, which is a useful check rather than a nuisance.

| object | node or leaf | note
| --- | --- | --- |
| `dict`, `list`, `tuple`, `namedtuple` | node | dict children ordered by sorted key
| `None` | node with no children | contributes zero leaves
| `jax.Array`, NumPy array, Python scalar | leaf | the usual case
| your own class | leaf, unless registered | register with `register_dataclass` or `register_pytree_node`

A custom class that is not registered is treated as a single opaque leaf, so `tree.map` passes the whole object to your function and `jit` refuses it as an argument. Libraries such as Flax and Equinox register their module classes so they behave as pytrees.

## An optimiser is a pure function over trees

SGD with momentum keeps a velocity tree of the same structure as the parameters:

```python
def sgd_momentum(params, grads, vel, lr=0.1, beta=0.9):
    vel = jax.tree.map(lambda v, g: beta * v + g, vel, grads)
    params = jax.tree.map(lambda p, v: p - lr * v, params, vel)
    return params, vel
```

Adam needs two moment trees and a step counter. Written in the update-returning style that Optax uses:

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

AdamW adds decoupled weight decay, `-lr * wd * p`, to each update, normally masked so that biases and normalisation parameters are excluded. Global-norm clipping, $g \leftarrow g \cdot \min(1, c / \|g\|)$ with $\|g\| = \sqrt{\sum_\ell \|g_\ell\|^2}$ over all leaves, is one `tree.leaves` reduction and one `tree.map`. Because everything is a pure function of arrays, the whole training step, including the optimiser, can sit inside one `jit`.

## Optax

Optax is the standard optimiser library for JAX and follows exactly this design. An optimiser is a `GradientTransformation` with `init(params) -> state` and `update(grads, state, params) -> (updates, state)`, and `optax.apply_updates(params, updates)` adds them. Transformations compose:

```python
import optax
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=optax.warmup_cosine_decay_schedule(0.0, 3e-4, 1_000, 100_000)),
)
opt_state = tx.init(params)
updates, opt_state = tx.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

Writing Adam yourself once is worth it to understand the state, but in real code use Optax: its implementations handle dtype details, masks and schedules consistently.

<div class="ex">
<div class="lab">Worked example</div>

**a float in the parameter tree**

A weight-decay mask is built as `jax.tree.map(lambda x: x.ndim > 1, params)`, so matrices are decayed and vectors (biases, norm scales) are not.

Later someone stores a layer's dropout rate as a Python float inside `params`. It becomes a leaf, `x.ndim` fails on a float, and worse, `grad` now tries to differentiate with respect to it.

Keep non-trainable configuration out of the parameter tree (pass it as a static argument or close over it), and build masks from key paths with `tree_map_with_path` so the intent ("decay everything except names ending in `b` or `scale`") is explicit.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

in JAX a model is a pytree of arrays and an optimiser is a pure function from gradient and state trees to update and state trees. Learn `jax.tree.map`, keep the structures aligned, and let Optax supply the production versions.

</div>

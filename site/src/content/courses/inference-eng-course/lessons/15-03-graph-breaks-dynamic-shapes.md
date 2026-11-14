---
id: "15.03"
course: "inference-eng-course"
module: "15"
slug: "graph-breaks-dynamic-shapes"
title: "Graph Breaks & Dynamic Shapes"
keyIdeas: ["Graph breaks split the graph and stop fusion across the break.", "Common causes: tensor-value control flow, .item(), unsupported Python, printing.", "Find them with TORCH_LOGS=graph_breaks or torch._dynamo.explain.", "Inference has varying batch sizes and lengths; static compilation recompiles per shape.", "Options: mark dims dynamic, pad to buckets, or compile only shape-stable parts."]
question: "What is a graph break, give two causes, and how do dynamic shapes cause recompilation?"
answer: "A graph break is where TorchDynamo cannot trace the Python code, so it ends the current graph, runs that code eagerly, and starts a new graph; fusion across the break is lost. Causes: data-dependent control flow (if on a tensor value), calls to unsupported functions or .item(). With static shapes, a new input shape (different batch size or sequence length) triggers a new compilation; marking dimensions dynamic avoids that at some optimization cost."
sources: [{"title": "PyTorch: torch.compile troubleshooting", "url": "https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html", "note": "Graph breaks and recompiles."}, {"title": "PyTorch: Dynamic shapes", "url": "https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html", "note": "Symbolic shapes in compile."}]
tags: ["compile"]
---

## Graph breaks

TorchDynamo captures tensor operations by watching Python run. When it reaches code it cannot express as a graph, it stops the current graph, runs that code in normal Python, and starts a new graph afterwards. This is a **graph break**.

```
def f(x):
    y = x * 2
    if y.sum() > 0:        # control flow on a tensor's value: graph break
        y = y + 1
    return y.relu()
```

To decide the `if`, Python needs the actual value of `y.sum()`, which lives on the GPU. Dynamo compiles `x * 2` and `y.sum()` as one graph, synchronizes to read the value, runs the branch in Python, then compiles the rest as another graph.

Each break costs:

- a synchronization or a Python round trip,
- lost fusion opportunities across the break,
- an extra point where CUDA graph capture is harder.

## Common causes in inference code

- Control flow on tensor values, as above.
- `.item()`, `.tolist()` or printing tensors.
- Calls into unsupported libraries or C extensions.
- Data-dependent output shapes, such as `torch.nonzero` or boolean masking that produces a tensor of unknown length.
- Some Python constructs that Dynamo does not support.

## Finding them

```
TORCH_LOGS="graph_breaks" python run.py
```

Each break is reported with the line of code and the reason. You can also use `torch._dynamo.explain(fn)(*inputs)` for a summary of graphs, breaks and their causes. For the core of the model, aim for zero breaks. Setting `fullgraph=True` in torch.compile makes any break an error, which is a good way to enforce it.

## Dynamic shapes

Inference inputs change shape all the time:

- The number of tokens in each step changes as sequences join and finish.
- Prefill chunk lengths vary.
- Context lengths grow by one every step.

By default, Dynamo specializes on input shapes. When it sees a new shape, it either recompiles for that exact shape or, after a couple of different values, marks the dimension as dynamic and compiles a version that works for any size.

| Strategy | Pros | Cons |
| --- | --- | --- |
| Static: compile per shape | the most specialized, fastest kernels | many compiles, long startup, stalls on new shapes |
| Dynamic dimensions | one compile covers all sizes | less specialization, sometimes slower kernels |
| Bucketing: pad to a few sizes | few compiles, specialized kernels | padding waste |

## What engines do

A common pattern:

- Compile the model body once with the token dimension marked dynamic, so any number of tokens per step works without recompiling.
- Capture CUDA graphs for a fixed list of batch sizes, such as 1, 2, 4, 8, 16, 24, 32 and so on up to a maximum, and pad each decode step up to the nearest captured size.
- Keep attention outside the compiled region as a custom operator, so its variable per-sequence metadata does not cause recompiles.

<div class="ex">
<div class="lab">Worked example</div>

**a recompile storm**

A team compiled their model with default settings and started a load test. TPOT p99 was 4 s for the first ten minutes, then settled to 30 ms.

The recompile log showed 47 recompilations: one for each new batch size seen during the ramp-up, each taking 3 to 8 seconds, during which the engine served nothing.

Marking the token dimension as dynamic reduced this to one compile at startup. A warm-up script that sent every expected shape before taking traffic removed the rest.

</div>

## Practical rules

- Keep data-dependent control flow, logging and sampling logic outside the compiled region.
- Avoid host synchronizations inside the model.
- Mark the batch or token dimension as dynamic, or bucket it.
- During a benchmark, check the recompile counter or log. After warm-up, it should stop increasing. Any recompile in steady state is a latency spike in production.

## Deep dive: padding waste from shape buckets

To avoid recompiling for every batch size, engines pad to a fixed set of sizes. If the true batch is ⟦m812⟧ and the bucket is ⟦m813⟧, the wasted share of the step is

⟦M811⟧

~~~dg-raw
%%chart Worst-case padding waste just above a bucket boundary.
x = 5, 9, 17, 33, 65, 129
logx = 1
xlab = True batch size
ylab = Wasted share of the step (%)
Power-of-two buckets: 37.5, 44, 47, 48, 49, 50
Buckets every 8 above 8: 37.5, 44, 29, 17.5, 9.7, 5.1
%%
~~~

For memory-bound decode, padding rows cost little (the weights are read anyway), so the waste above overstates the time lost. It matters more for compute-bound steps and for attention, where padded sequences still launch work.

Recompiles are the other side of the trade. Each distinct shape compiled costs seconds to minutes. With ⟦m814⟧ buckets and ⟦m815⟧ per compile, startup grows by ⟦m816⟧, which is why engines capture graphs for a few dozen sizes, not hundreds.

<div class="take">
<div class="lab">Key takeaway</div>

graph breaks and shape changes are the main ways compilation goes wrong. Keep the model trace-friendly with no breaks, handle variable shapes with dynamic dimensions or buckets, and confirm recompiles stop after warm-up.

</div>

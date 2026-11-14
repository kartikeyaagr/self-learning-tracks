---
id: "04.03"
course: "ml-toolkit"
module: "04"
slug: "building-models-with-nn-module"
title: "Building models with nn.Module"
keyIdeas: ["An `nn.Module` registers anything assigned as an attribute: `nn.Parameter`s become trainable weights, other modules become submodules, and `register_buffer` adds state that moves and saves with the model but is not trained. Plain tensors are invisible.", "Registration is what makes `parameters()`, `.to(device)`, `state_dict()` and `train()/eval()` work. Layers kept in a plain Python list are never trained or moved; use `nn.ModuleList`, `nn.ModuleDict` or `nn.Sequential`.", "Write the computation in `forward` but call the module as `model(x)`, so hooks and tools such as `torch.compile` run. `eval()` changes dropout and batch norm behaviour; it does not turn off gradients.", "`state_dict()` is an ordered map from dotted names (`fc1.weight`) to tensors. Save and load it, and when loading with `strict=False` always inspect the missing and unexpected keys.", "`nn.Linear` starts from $U(-1/\\sqrt{\\text{fan\\_in}}, 1/\\sqrt{\\text{fan\\_in}})$. Custom schemes go in one function applied with `model.apply(fn)`; in deep networks without normalisation the choice decides whether the signal survives."]
question: "A colleague's Transformer stores its blocks as `self.blocks = [Block(d) for _ in range(12)]` and its causal mask as `self.mask = torch.tril(torch.ones(T, T))`. Training runs on CPU without errors but the loss barely moves. On GPU it crashes. Explain both symptoms, say what a saved `state_dict()` would contain, and describe how you would fix and verify the model."
answer: "A plain Python list does not go through `nn.Module`'s registration, so the 12 blocks are not submodules. `model.parameters()` returns only the embedding, final norm and head, and the optimiser built from it updates only those. The blocks still run in `forward` (and even receive gradients), but their weights never change, so they act as fixed random feature extractors: the loss moves a little because the head adapts, then stalls, exactly like the plain-list curve in this lesson. On GPU, `model.to('cuda')` moves only registered parameters and buffers. The blocks' weights and the plain-attribute mask stay on the CPU, so the first block's matrix multiply with CUDA activations fails with an `Expected all tensors to be on the same device` error. A saved `state_dict()` would contain only the embedding, norm and head tensors (and no mask); loading it into a corrected model with `strict=True` raises missing keys for every `blocks.i....` entry. Fix: `self.blocks = nn.ModuleList([Block(d) for _ in range(12)])` and `self.register_buffer('mask', torch.tril(torch.ones(T, T)), persistent=False)` (non-persistent because it can be recomputed and need not be checkpointed). Verify by printing the model (all 12 blocks appear), comparing `sum(p.numel() for p in model.parameters())` with a hand count, checking that every parameter and buffer reports the same device after `.to('cuda')`, and confirming that the training loss now falls well below its earlier plateau. Old checkpoints are not worth recovering: the blocks in them were never trained."
sources: [{"title": "PyTorch docs: torch.nn.Module", "url": "https://pytorch.org/docs/stable/generated/torch.nn.Module.html", "note": "Registration, apply, state_dict, load_state_dict and hooks: the whole API on one page."}, {"title": "PyTorch docs: torch.nn.init", "url": "https://pytorch.org/docs/stable/nn.init.html", "note": "Every initialiser with its formula, including the gain values per nonlinearity."}, {"title": "PyTorch tutorial: Saving and loading models", "url": "https://pytorch.org/tutorials/beginner/saving_loading_models.html", "note": "What a state_dict contains and the recommended save/load patterns."}, {"title": "He et al.: Delving deep into rectifiers (2015)", "url": "https://arxiv.org/abs/1502.01852", "note": "The derivation behind Kaiming initialisation for ReLU networks."}]
practice: ["simple-neural-network", "custom-linear-layer", "residual-block", "collect-module-parameters-recursively", "weight-initialization"]
tags: ["nn.Module", "state_dict", "initialisation"]
tools: ["PyTorch"]
---

## What you will learn

- What an `nn.Module` is and why PyTorch models are written as classes with `__init__` and `forward`.
- How **registration** works: which attributes become trainable parameters, submodules or buffers, and which stay invisible.
- How to compose layers with `nn.Sequential`, `nn.ModuleList` and `nn.ModuleDict`, and why a plain Python list silently breaks training.
- What `state_dict()` contains, how to save and load it, and how to read the missing and unexpected keys when loading fails.
- How layers are initialised by default, how to override that with `model.apply`, and why the choice matters more as networks get deeper.

*Before you start:* you need tensors, devices and dtypes (04.01) and how autograd builds gradients for tensors that have `requires_grad=True` (04.02). Basic Python classes (`__init__`, `self`, inheritance) are assumed.

## From loose tensors to a model

In lesson 04.02 you created a weight tensor with `requires_grad=True`, called `backward()` and updated it yourself. That works for one matrix. A real network has dozens of weight tensors, some untrained state (running averages, masks), a training/evaluation switch, and must be saved, loaded and moved to a GPU as one unit.

`nn.Module` does that bookkeeping. You declare the pieces as attributes, describe the computation in `forward`, and the base class gives you `parameters()`, `.to(device)`, `state_dict()` and `train()`/`eval()` for free. The losses and optimisers of lesson 04.04 consume exactly what a module hands them.

## The idea in one picture

Here is a complete model: three inputs, a hidden layer of four units, two outputs.

```python
import torch
from torch import nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()               # set up the registry first
        self.hidden = nn.Linear(3, 4)    # registered as a submodule
        self.act = nn.ReLU()
        self.out = nn.Linear(4, 2)

    def forward(self, x):                # the computation, nothing else
        return self.out(self.act(self.hidden(x)))

net = TinyNet()
for name, p in net.named_parameters():
    print(name, tuple(p.shape))
# hidden.weight (4, 3)
# hidden.bias (4,)
# out.weight (2, 4)
# out.bias (2,)
print(sum(p.numel() for p in net.parameters()))   # 26
print(net(torch.tensor([[1.0, 2.0, 3.0]])).shape)  # torch.Size([1, 2])
```

You never told PyTorch "these are my weights": assigning `nn.Linear` objects to `self` was enough, and `named_parameters()` walked into each one. The count is $3 \times 4 + 4 = 16$ for the hidden layer plus $4 \times 2 + 2 = 10$ for the output, 26 in total.

A few words used throughout:

- A **module** is any object that inherits from `nn.Module`. A single layer is a module; a whole network is a module made of modules.
- A **parameter** (`nn.Parameter`) is a tensor the optimiser should train. It always has `requires_grad=True` by default.
- A **submodule** (or child) is a module stored as an attribute of another module. Modules therefore form a tree.
- A **buffer** is a tensor that belongs to the model's state, so it is moved and saved with it, but is not trained.

The figure shows this tree for a slightly bigger model, the MNIST-sized MLP we will use for the rest of the lesson.

~~~dg-fig 04.03-tree.svg
The module tree of the MLP used in this lesson. Each child contributes its parameters under a dotted name (fc1.weight); the Dropout child has no parameters, and the mean buffer is saved but never trained.
~~~

## Step 1: what registration actually does

`nn.Module` overrides `__setattr__`, the method Python calls on every `self.something = value`. Before storing the value it checks its type:

1. If the value is an `nn.Parameter`, it goes into the module's private `_parameters` dictionary.
2. If the value is an `nn.Module`, it goes into `_modules`.
3. Tensors passed to `register_buffer` go into `_buffers`.
4. Anything else, including an ordinary `torch.Tensor` or a Python list, is stored as a normal attribute and nothing else knows about it.

Every feature of the class is just a recursive walk over those three dictionaries. `parameters()` walks `_parameters` of this module and of every module in `_modules`, and so on down the tree. `.to()` walks parameters and buffers. `state_dict()` walks parameters and persistent buffers. That is why `super().__init__()` must run before your first assignment: it creates the dictionaries, and without it you get `cannot assign module before Module.__init__() call`.

To see it with nothing hidden, write a linear layer by hand:

```python
class MyLinear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) / d_in**0.5)
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.scale = torch.ones(d_out)          # plain tensor: NOT registered

    def forward(self, x):
        return x @ self.weight.T + self.bias    # (batch, d_in) -> (batch, d_out)

lin = MyLinear(3, 2)
print([n for n, _ in lin.named_parameters()])   # ['weight', 'bias']
print(list(lin._parameters))                    # ['weight', 'bias']
print(lin(torch.ones(5, 3)).shape)              # torch.Size([5, 2])
```

`scale` is missing: it will not be trained, moved or saved. Usually that is a bug.

Note the layout: a linear weight is stored as `(out_features, in_features)` and the layer computes $y = x W^\top + b$, so `nn.Linear(784, 256).weight` has shape `(256, 784)`.

## Step 2: forward, and why you call the module

`forward` describes the computation for one batch and may contain any Python (loops, `if`s); autograd records whatever actually runs.

Always call the module itself, `model(x)`, rather than `model.forward(x)`. The module's `__call__` method runs registered **hooks** before and after `forward` (used by profilers, debuggers and wrappers such as FSDP); calling `forward` directly skips them silently.

<div class="ex">
<div class="lab">Worked example</div>

**a forward pass by hand**

Take `nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))` and set the first layer to $W_1 = \begin{pmatrix}1 & -1\\ 0.5 & 2\end{pmatrix}$, $b_1 = (0, -1)$, the second to $W_2 = (2, -1)$, $b_2 = 0.5$.

Input $x = (3, 1)$. Hidden pre-activation: row 1 is $1 \cdot 3 + (-1) \cdot 1 + 0 = 2$; row 2 is $0.5 \cdot 3 + 2 \cdot 1 - 1 = 2.5$.

ReLU keeps both because they are positive: $h = (2, 2.5)$. Output: $2 \cdot 2 + (-1) \cdot 2.5 + 0.5 = 4 - 2.5 + 0.5 = 2$.

Second input $x = (1, 2)$. Row 1: $1 - 2 + 0 = -1$; row 2: $0.5 + 4 - 1 = 3.5$. ReLU zeroes the first: $h = (0, 3.5)$.

Output: $2 \cdot 0 - 3.5 + 0.5 = -3$. PyTorch prints `tensor([[2.]])` and `tensor([[-3.]])` for the two inputs, and the parameter names are `0.weight`, `0.bias`, `2.weight`, `2.bias`: `nn.Sequential` names children by position, and the ReLU at position 1 has no parameters.

</div>

## Step 3: containers, and the plain-list trap

Three containers cover almost every architecture:

- `nn.Sequential(a, b, c)` runs its children in order, feeding each output into the next. Good for simple stacks.
- `nn.ModuleList([...])` is a registered list. It has no `forward`; you loop over it yourself, which lets you add residual connections or pass extra arguments.
- `nn.ModuleDict({...})` is a registered dictionary, useful for named branches such as one head per task.

The classic mistake is to build a stack of blocks in an ordinary list:

```python
class ListNet(nn.Module):
    def __init__(self, d=8, use_modulelist=True):
        super().__init__()
        blocks = [nn.Linear(d, d) for _ in range(4)]
        self.blocks = nn.ModuleList(blocks) if use_modulelist else blocks
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        for b in self.blocks:
            x = torch.relu(b(x))
        return self.head(x)

for flag in [False, True]:
    m = ListNet(use_modulelist=flag)
    print(flag, sum(p.numel() for p in m.parameters()), len(m.state_dict()))
# False 9 2       <- only the head: 8 weights + 1 bias
# True 297 10     <- 4 blocks x (64 + 8) + 9
```

Nothing crashes. The forward pass uses all four blocks, but the optimiser was built from `m.parameters()` and never received their weights, so they keep their random starting values forever. The figure shows the effect on a small regression task.

~~~dg-fig 04.03-listtrap.svg
The same network trained with Adam for 400 steps (measured on CPU here). With a plain list the optimiser only updates the 9 head parameters and the loss stalls near 0.68; with nn.ModuleList all 297 parameters train and the loss falls below 0.011.
~~~

The plain-list version is a linear head on fixed random features, and that is as well as it can do. On a GPU the bug is louder: `.to("cuda")` also skips the blocks, so the first matrix multiply fails with a device-mismatch error.

## Step 4: buffers and train/eval mode

Normalisation statistics, a fixed mean to subtract, a causal mask and position tables must travel with the model but never be trained. Register them as buffers:

```python
class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d))                    # saved and moved
        self.register_buffer("cache", torch.ones(d), persistent=False)  # moved only
        self.plain = torch.ones(d)                                       # invisible
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x - self.mean)

m = Norm(3).to(torch.float64)
print(m.mean.dtype, m.cache.dtype, m.plain.dtype)  # float64 float64 float32
print(list(m.state_dict()))                        # ['mean', 'lin.weight', 'lin.bias']
```

`.to(dtype)` walks the same registry as `.to(device)`, so this CPU example shows the effect: both buffers were converted and the plain attribute was left behind (on a GPU it would stay on the CPU). The non-persistent buffer stays out of `state_dict()`, keeping checkpoints small for anything recomputable.

The grid summarises who sees what.

~~~dg-fig 04.03-registry.svg
Which module features see each kind of state. Only registered parameters reach the optimiser; buffers are moved and (if persistent) saved; plain tensors and layers kept in a Python list are invisible to everything.
~~~

Every module also carries a boolean `training` flag. `model.train()` sets it to `True` on the whole tree and `model.eval()` sets it to `False`. Layers whose behaviour differs between the two read the flag: dropout zeroes random entries and scales the rest by $1/(1-p)$ in training, and does nothing in evaluation; batch norm uses batch statistics in training and its running-average buffers in evaluation.

```python
torch.manual_seed(0)
drop = nn.Dropout(p=0.5)
print(drop(torch.ones(8)))   # tensor([0., 0., 2., 0., 0., 0., 2., 2.])  training mode
drop.eval()
print(drop(torch.ones(8)))   # tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

`eval()` does not turn off gradient tracking; wrap inference in `torch.no_grad()` or `torch.inference_mode()` for that. They are two separate switches.

## Step 5: state_dict, the model as a dictionary

`model.state_dict()` returns an ordered dictionary from dotted names to tensors: every parameter and every persistent buffer, named by its path in the tree. For the MLP in the figure:

```python
class MLP(nn.Module):
    def __init__(self, d_in=784, d_hid=256, n_cls=10, p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(d_hid, n_cls)
        self.register_buffer("mean", torch.zeros(d_in))

    def forward(self, x):
        x = x.flatten(1) - self.mean
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

model = MLP()
print({k: tuple(v.shape) for k, v in model.state_dict().items()})
# {'mean': (784,), 'fc1.weight': (256, 784), 'fc1.bias': (256,),
#  'fc2.weight': (10, 256), 'fc2.bias': (10,)}
print(sum(p.numel() for p in model.parameters()))   # 203530
```

<div class="ex">
<div class="lab">Worked example</div>

**counting the MLP's parameters**

`fc1` maps 784 inputs to 256 units: $784 \times 256 = 200{,}704$ weights plus 256 biases, so 200,960 parameters.

`fc2` maps 256 to 10: $256 \times 10 = 2{,}560$ weights plus 10 biases, so 2,570 parameters. Dropout has none.

Total trainable: $200{,}960 + 2{,}570 = 203{,}530$. The buffer `mean` (784 values) is in `state_dict()` but not in `parameters()`.

In float32 that is $203{,}530 \times 4 = 814{,}120$ bytes, about 0.8 MB. Training with Adam adds the gradients (another 0.8 MB) and two moment tensors per parameter (1.6 MB), so roughly 3.3 MB before activations.

</div>

Save the `state_dict`, not the pickled model object, which ties the file to your exact class and file layout.

```python
torch.save(model.state_dict(), "mlp.pt")
sd = torch.load("mlp.pt", map_location="cpu")   # weights_only=True is the default since 2.6
model.load_state_dict(sd)                       # <All keys matched successfully>
```

When names do not line up, `load_state_dict` with the default `strict=True` raises an error listing them. A very common cause is a checkpoint saved from a wrapper (`DataParallel` adds `module.`, `torch.compile` adds `_orig_mod.`):

```python
bad = {"module." + k: v for k, v in sd.items()}
res = model.load_state_dict(bad, strict=False)
print(len(res.missing_keys), len(res.unexpected_keys))   # 5 5
fixed = {k.removeprefix("module."): v for k, v in bad.items()}
print(model.load_state_dict(fixed))                      # <All keys matched successfully>
```

`strict=False` is the right tool when you deliberately load a pretrained backbone into a model with a new head, but always print the two lists: a load that "succeeds" while every key is missing has loaded nothing. Shapes must match exactly even with `strict=False`.

## Step 6: initialisation

Every built-in layer fills its own parameters in a `reset_parameters()` method that runs at construction. `nn.Linear` and `nn.Conv2d` use Kaiming-uniform with $a = \sqrt5$. Written out, the weights are drawn from

$$W_{ij} \sim U\!\left(-\frac{1}{\sqrt{\text{fan\_in}}},\ \frac{1}{\sqrt{\text{fan\_in}}}\right),$$

where **fan_in** is the number of inputs to each output unit (784 for `fc1`), and the bias uses the same bound. So $1/\sqrt{784} = 1/28 \approx 0.0357$, and indeed the largest absolute weight in a fresh `nn.Linear(784, 256)` is 0.0357. Embeddings start from $\mathcal N(0, 1)$, and normalisation layers start at weight 1 and bias 0.

~~~dg-fig 04.03-init-hist.svg
Default nn.Linear weights are spread evenly between the two dashed bounds. Four times more inputs (16 to 256) means a bound four times smaller, so each output starts as a sum of more but smaller terms.
~~~

To use a different scheme, write one function that handles each layer type and let `apply` call it on every module in the tree (children first, then the parent):

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
print(round(model.fc1.weight.std().item(), 4))   # 0.02
```

The functions in `nn.init` end in an underscore because they write in place, and they run under `no_grad` internally, so they are safe on parameters that require grad. If you write your own init by hand (`m.weight.copy_(...)`), wrap it in `with torch.no_grad():`.

Why bother? Each layer multiplies the scale of its input by a factor set by its weights, and over 30 layers a factor slightly off compounds geometrically. The figure measures this in a plain ReLU network without normalisation or residual connections.

~~~dg-fig 04.03-depth.svg
Spread of activations through 30 plain ReLU layers (measured here). Kaiming-normal keeps it near 0.7; the PyTorch default shrinks it about 15-fold in five layers before the random biases hold it up; std 0.02 shrinks it to about 3e-20, so the signal effectively vanishes.
~~~

This is why GPT-style models that use std 0.02 depend on normalisation layers and residual connections, and why the default works fine for the shallow MLPs and CNNs you will meet in this module.

## Freezing, parameter groups and inspection

Because every parameter has a dotted name, model surgery is a few lines of filtering:

```python
for p in model.fc1.parameters():               # freeze the first layer
    p.requires_grad_(False)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))   # 2570

decay    = [p for n, p in model.named_parameters() if p.ndim >= 2]   # weight matrices
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]    # biases, norm weights
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
```

Keeping biases and normalisation weights out of weight decay appears in nearly every Transformer script; lesson 04.04 explains parameter groups. `named_modules()` gives the same access to whole layers.

## Checking a model before you train it

A few seconds of inspection catches most structural bugs:

- `print(model)` shows the registered tree. A layer you created but cannot see there is not registered.
- Count parameters per child, `{n: sum(p.numel() for p in c.parameters()) for n, c in model.named_children()}`, and compare with a hand count. For the MLP: `{'fc1': 200960, 'drop': 0, 'fc2': 2570}`.
- Run one dummy batch: `model(torch.randn(2, 1, 28, 28)).shape` should be `(2, 10)`.
- `{p.device for p in model.parameters()} | {b.device for b in model.buffers()}` should contain one device.
- After loading weights, read what `load_state_dict` returned; before evaluating, confirm `model.training` is `False`.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Loss barely moves; parameter count too small | layers in a Python list or dict | `nn.ModuleList` / `nn.ModuleDict`
| `Expected all tensors to be on the same device` | tensor stored as a plain attribute, or created in `forward` without `device=x.device` | `register_buffer`; create tensors on the input's device
| `cannot assign module before Module.__init__() call` | forgot `super().__init__()` or called it late | call it on the first line of `__init__`
| Missing keys `fc1.weight`, unexpected `module.fc1.weight` | checkpoint saved from a wrapper | strip the prefix before loading
| Evaluation scores vary run to run | model left in training mode, dropout active | `model.eval()` before evaluating
| Parameter never changes although it is in `parameters()` | created inside `forward`, so a new leaf each call | create every `nn.Parameter` in `__init__`
| Checkpoint far bigger than expected | large recomputable tables saved as persistent buffers | `register_buffer(..., persistent=False)`

## Going deeper

**Weight tying.** Assigning one module's parameter to another (`self.head.weight = self.embed.weight`) makes both refer to the same `Parameter` object. `parameters()` deduplicates by identity (a 100-token, 16-dimension tied model reports 1,600 parameters, not 3,200), and the optimiser applies one update with the gradient summed over both uses. `state_dict()` still lists both names, pointing at one storage.

**Where the default bound comes from.** `kaiming_uniform_` draws from $U(-\text{bound}, \text{bound})$ with $\text{bound} = \text{gain}\sqrt{3/\text{fan\_in}}$ and $\text{gain} = \sqrt{2/(1+a^2)}$. With $a = \sqrt5$ the gain is $\sqrt{1/3}$, and the bound becomes $\sqrt{1/3}\cdot\sqrt{3/\text{fan\_in}} = 1/\sqrt{\text{fan\_in}}$. A uniform distribution on that interval has variance $\text{bound}^2/3 = 1/(3\,\text{fan\_in})$, which is one sixth of the $2/\text{fan\_in}$ that He et al. derived for keeping ReLU activations at a constant scale. That gap is the shrinking you saw in the depth figure.

**The variance argument in one line.** For a unit $y = \sum_{j=1}^{n} w_j x_j$ with independent zero-mean weights, $\operatorname{Var}(y) = n \operatorname{Var}(w)\, \mathbb E[x^2]$. ReLU passes half of a symmetric input's second moment, so keeping the scale constant needs $n \operatorname{Var}(w) \cdot \tfrac12 = 1$, that is $\operatorname{Var}(w) = 2/n$. GPT-2-style models additionally scale the output projection of each residual branch by $1/\sqrt{2L}$ for $L$ layers, because each of the $2L$ branches adds variance to the residual stream.

**state_dict ordering.** Keys come out as a module's own parameters, its own buffers, then each child in registration order (so `mean` comes first for the MLP). Optimisers save state by parameter position, not name, so reordering registration breaks resuming an optimiser even when the model still loads.

## Key terms

| Term | Meaning
| --- | --- |
| `nn.Module` | Base class that registers parameters, buffers and submodules and walks them recursively
| `nn.Parameter` | A tensor marked as trainable; returned by `parameters()` and updated by the optimiser
| Submodule | A module stored as an attribute of another module; modules form a tree
| Buffer | Registered tensor that moves and (if persistent) saves with the model but is not trained
| `nn.ModuleList` / `nn.ModuleDict` | Registered list and dict of modules; plain Python containers are invisible
| `state_dict` | Ordered map from dotted names to parameter and persistent buffer tensors; the unit you save and load
| `train()` / `eval()` | Set the `training` flag that dropout and batch norm read; does not affect gradient tracking
| fan_in | Number of inputs feeding each output unit; sets the scale of default and Kaiming initialisation
| `model.apply(fn)` | Calls `fn` on every module in the tree, children first, typically for custom initialisation

<div class="take">
<div class="lab">Key takeaway</div>

create every weight in `__init__` as a registered parameter, submodule or buffer, call the model rather than `forward`, and treat the `state_dict()` names as the model's real interface: what you save, load, freeze, group and initialise by.

</div>

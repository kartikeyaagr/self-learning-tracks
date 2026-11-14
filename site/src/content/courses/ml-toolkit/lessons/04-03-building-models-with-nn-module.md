---
id: "04.03"
course: "ml-toolkit"
module: "04"
slug: "building-models-with-nn-module"
title: "Building models with nn.Module"
keyIdeas: ["An `nn.Module` is a container that registers anything assigned as an attribute: `nn.Parameter`s become trainable weights, other modules become submodules, and `register_buffer` adds state that moves with the model but is not trained.", "Registration is what makes `parameters()`, `.to(device)`, `state_dict()`, `train()/eval()` and hooks work. A plain Python list of layers is invisible to all of them; use `nn.ModuleList` or `nn.ModuleDict`.", "`state_dict()` is an ordered map from dotted names (`blocks.3.attn.qkv.weight`) to tensors. It is the unit you save, load and inspect, and `load_state_dict` reports missing and unexpected keys.", "Default initialisations are layer-specific (`nn.Linear` draws from $U(-1/\\sqrt{\\text{fan\\_in}}, 1/\\sqrt{\\text{fan\\_in}})$); custom schemes go in one function applied with `model.apply(fn)` under `torch.no_grad()`."]
question: "A colleague's Transformer stores its blocks as `self.blocks = [Block(d) for _ in range(12)]` and its causal mask as `self.mask = torch.tril(torch.ones(T, T))`. Training runs on CPU without errors but the loss barely moves. On GPU it crashes. Explain both symptoms and what saving a checkpoint would contain."
answer: "A plain list does not register the blocks as submodules, so `model.parameters()` returns only the embedding, final norm and head. The optimiser updates those, which is why loss moves a little, while the 12 blocks stay at their random initialisation, acting as fixed random features. On GPU, `model.to('cuda')` moves only registered state: the blocks' weights and the plain-attribute mask stay on the CPU, so the first block's matmul with CUDA activations fails with a device-mismatch error. A saved `state_dict()` would contain only the embedding, norm and head tensors; loading it into a corrected model with `strict=True` raises missing keys for every block. Fix: `nn.ModuleList` for the blocks and `register_buffer('mask', ..., persistent=False)` for the mask."
sources: [{"title": "PyTorch docs: torch.nn.Module", "url": "https://pytorch.org/docs/stable/generated/torch.nn.Module.html", "note": "Registration, apply, state_dict, load_state_dict and hooks: the whole API on one page."}, {"title": "PyTorch docs: torch.nn.init", "url": "https://pytorch.org/docs/stable/nn.init.html", "note": "Every initialiser with its formula, including the gain values per nonlinearity."}, {"title": "PyTorch tutorial: Saving and loading models", "url": "https://pytorch.org/tutorials/beginner/saving_loading_models.html", "note": "What a state_dict contains and the recommended save/load patterns."}, {"title": "He et al.: Delving deep into rectifiers (2015)", "url": "https://arxiv.org/abs/1502.01852", "note": "The derivation behind Kaiming initialisation for ReLU networks."}]
practice: ["simple-neural-network", "custom-linear-layer", "residual-block", "collect-module-parameters-recursively", "weight-initialization"]
tags: ["nn.Module", "state_dict", "initialisation"]
tools: ["PyTorch"]
---

## What a module is

`nn.Module` is a base class with an overridden `__setattr__`. When you write `self.fc = nn.Linear(784, 256)` in `__init__`, the module records `fc` as a child. When you assign an `nn.Parameter`, it records a trainable tensor. Everything else the class offers walks that registry recursively.

```python
class MLP(nn.Module):
    def __init__(self, d_in=784, d_hid=256, n_cls=10, p=0.1):
        super().__init__()                      # required before any assignment
        self.fc1 = nn.Linear(d_in, d_hid)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(d_hid, n_cls)
        self.register_buffer("mean", torch.zeros(d_in))   # moved and saved, not trained

    def forward(self, x):
        x = (x - self.mean).flatten(1)
        return self.fc2(self.drop(torch.relu(self.fc1(x))))

model = MLP().to("cuda")
sum(p.numel() for p in model.parameters())     # 203,530
```

Call the module, `model(x)`, not `model.forward(x)`: `__call__` runs registered hooks around `forward`, and some tooling (hooks, `torch.compile`, FSDP) relies on that.

~~~dg-layers What a module registers, and which features see each kind of state.
nn.Parameter | trained by the optimiser; in parameters() and state_dict()
*Submodule | recursed into by parameters(), to(), train()/eval(), state_dict()
Buffer | moved by to() and saved in state_dict(), never trained
Plain attribute | invisible: not moved, not saved, not trained
~~~

## The registration traps

- `self.layers = [nn.Linear(d, d) for _ in range(4)]` creates four layers that the optimiser never sees and `model.to("cuda")` never moves. Use `nn.ModuleList(...)`; for named branches use `nn.ModuleDict`. `nn.Sequential` chains modules whose outputs feed the next input.
- A tensor stored as `self.mask = torch.tril(...)` stays on the CPU when the model moves. Use `register_buffer("mask", ...)`. Pass `persistent=False` for caches you can recompute (RoPE tables, causal masks) so they stay out of checkpoints.
- Wrapping an existing tensor in `nn.Parameter` inside `forward` creates a fresh, untracked leaf every call; parameters belong in `__init__`.
- **Weight tying** (`self.head.weight = self.embed.weight`) makes two modules share one `Parameter`. `parameters()` deduplicates it, so the optimiser updates it once with the summed gradient from both uses.

## state_dict: the model as a dictionary

`model.state_dict()` returns an ordered dictionary of parameter and persistent buffer tensors keyed by dotted path, such as `fc1.weight`, `fc1.bias`, `mean`. It is the canonical way to save weights (lesson 7 covers full checkpoints):

```python
torch.save(model.state_dict(), "mlp.pt")
sd = torch.load("mlp.pt", map_location="cpu")    # weights_only=True by default in 2.6+
missing, unexpected = model.load_state_dict(sd, strict=False)
```

With `strict=True` (the default), any name mismatch raises. Typical causes: a checkpoint saved from a `DataParallel` or `torch.compile` wrapper (keys prefixed with `module.` or `_orig_mod.`), a renamed attribute, or loading a backbone into a model with a new head. Use `strict=False` deliberately and inspect both returned lists rather than assuming success. Shapes must still match exactly.

`named_parameters()` and `named_modules()` are the tools for surgery: freezing everything except the head, building optimiser parameter groups that exclude biases and norm weights from weight decay, or counting parameters per block.

## Initialisation

Each built-in layer initialises itself in `reset_parameters()`. `nn.Linear` and `nn.Conv2d` use Kaiming-uniform with $a=\sqrt5$, which works out to $W \sim U(-1/\sqrt{\text{fan\_in}},\ 1/\sqrt{\text{fan\_in}})$; embeddings use $\mathcal N(0, 1)$; norm layers start at weight 1 and bias 0. Those defaults are reasonable for small MLPs and CNNs, but deep residual stacks and Transformers usually override them. The Deep Learning course derives why variance preservation matters; mechanically, you apply one function to every submodule:

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)          # recursive, children first
```

The `nn.init` functions run under `no_grad` internally, so they are safe on parameters that require grad. GPT-2-style models also scale the output projection of each residual branch by $1/\sqrt{2L}$ for $L$ layers so that the residual stream's variance does not grow with depth.

<div class="ex">
<div class="lab">Worked example</div>

**counting the MLP above**

`fc1`: $784 \times 256$ weights plus 256 biases = 200,960 parameters.

`fc2`: $256 \times 10$ weights plus 10 biases = 2,570 parameters. Dropout has none; the buffer `mean` (784 values) is in `state_dict()` but not in `parameters()`.

Total trainable: 203,530, about 0.8 MB in float32. Adam adds two state tensors per parameter, so training state is about 2.4 MB plus gradients.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

put every weight in `__init__` as a registered parameter, submodule or buffer, and treat `state_dict()` names as the model's real interface: what you save, load, freeze and group by.

</div>

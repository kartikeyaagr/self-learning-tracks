import numpy as np

META = dict(title='LoRA: fine-tune low-rank adapters on a frozen layer, then merge', tags=['Language Models', 'Training'],
            diff='Hard', runtime='numpy', lessons=['air-fanout-v1:10.02', 'ieng:26.02'],
            related=['custom-linear-layer', 'low-rank-approximation', 'svd-components'])

STATEMENT = r"""
LoRA freezes a pretrained weight $W_0$ and learns a low-rank update $\Delta W = \tfrac{\alpha}{r}AB$. Train the adapter on a regression task, then fold it into the weight so inference costs nothing extra.

Write `lora_finetune(X, Y, W0, A, B, alpha, lr, n_steps)` returning `(losses, A, B, W_merged)`.

## Layer
$\hat Y = XW_0 + \tfrac{\alpha}{r}(XA)B$ with `W0` `(Din, Dout)`, `A` `(Din, r)`, `B` `(r, Dout)`; $r$ is `A.shape[1]`.

## Training
Full-batch gradient descent for `n_steps` steps on $L = \mathrm{mean}(\hat Y - Y)^2$ over all $N \cdot D_{out}$ entries, updating only `A` and `B` (both from the same forward pass). Record $L$ before each step.

## Merge
$W_{merged} = W_0 + \tfrac{\alpha}{r}AB$ with the trained adapters, so $XW_{merged}$ equals the adapted layer's output.

Return `losses` `(n_steps,)` and new arrays. `W0` must not change.
"""

STARTER = '''import numpy as np

def lora_finetune(X, Y, W0, A, B, alpha, lr, n_steps):
    """Return (losses, A, B, W_merged)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lora_finetune(X, Y, W0, A, B, alpha, lr, n_steps):
    X, Y, W0 = (np.asarray(a, dtype=float) for a in (X, Y, W0))
    A, B = np.array(A, dtype=float), np.array(B, dtype=float)
    s = alpha / A.shape[1]
    losses = []
    for _ in range(n_steps):
        XA = X @ A
        R = X @ W0 + s * XA @ B - Y
        losses.append(float((R ** 2).mean()))
        G = 2 * R / R.size
        gB = s * XA.T @ G
        gA = s * X.T @ G @ B.T
        A, B = A - lr * gA, B - lr * gB
    return np.array(losses), A, B, W0 + s * A @ B
'''

HINTS = [
    'Let $G = 2(\\hat Y - Y)/(N D_{out})$. Then $\\partial L/\\partial B = \\tfrac{\\alpha}{r}(XA)^\\top G$ and $\\partial L/\\partial A = \\tfrac{\\alpha}{r}X^\\top G B^\\top$.',
    'Compute both gradients from the same forward pass before updating either. With the usual initialisation ($B = 0$) the first step changes only $B$.',
    'Never form a $D_{in}\\times D_{out}$ update during training; merge once at the end.',
]

_r = np.random.default_rng(2021)
def _case(N, Din, Dout, r, true_rank=None, zeroB=True):
    W0 = _r.normal(size=(Din, Dout)) / np.sqrt(Din)
    k = true_rank or r
    dW = _r.normal(size=(Din, k)) @ _r.normal(size=(k, Dout)) / Din
    X = _r.normal(size=(N, Din))
    Y = X @ (W0 + dW) + 0.01 * _r.normal(size=(N, Dout))
    A = _r.normal(size=(Din, r)) / np.sqrt(Din)
    B = np.zeros((r, Dout)) if zeroB else 0.1 * _r.normal(size=(r, Dout))
    return X, Y, W0, A, B
TESTS = [
    (*_case(6, 4, 3, 1), 1.0, 0.1, 1),
    (*_case(6, 4, 3, 1), 1.0, 0.1, 2),
    (*_case(32, 8, 6, 2), 4.0, 0.05, 50),
    (*_case(32, 8, 6, 2, true_rank=4), 2.0, 0.05, 80),
    (*_case(20, 6, 5, 3, zeroB=False), 6.0, 0.02, 30),
    (*_case(10, 5, 4, 2), 2.0, 0.0, 3),
]

EXPLANATION = r"""
## The idea
Fine-tuning updates of large pretrained models tend to have low "intrinsic rank", so LoRA learns $\Delta W$ as a product of two thin matrices. For a $4096\times4096$ projection and $r = 8$ that is 65,536 trainable values instead of 16.8 million, and optimiser state shrinks by the same factor.

## Initialisation and scaling
$A$ is random and $B$ is zero, so the adapted model starts identical to the pretrained one and the first gradient step moves only $B$ (tests 1 and 2). The $\alpha/r$ factor means that changing $r$ doesn't require retuning the learning rate as much; rsLoRA argues for $\alpha/\sqrt r$ at large ranks.

## Merging
Because the update is linear, $W_0 + \tfrac\alpha r AB$ can be computed once and the adapter disappears at inference. Keeping adapters separate instead lets one server host many fine-tunes over a shared base model (multi-LoRA serving), batching requests that use different adapters.

## What rank can and can't do
Test 4's target update has rank 4 but the adapter has rank 2, so the loss plateaus above the noise floor: the best it can do is the rank-2 truncation of the needed change.

## Common bugs
- Updating $W_0$ as well.
- Updating $A$ with the already-updated $B$.
- Forgetting the $\alpha/r$ factor in either the forward pass or the merge.

## Interview follow-ups
- Which projections in a transformer would you adapt first, and why?
- How does QLoRA combine this with a 4-bit base model?
"""

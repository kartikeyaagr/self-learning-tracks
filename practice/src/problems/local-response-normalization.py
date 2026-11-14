import numpy as np

META = dict(title='Local response normalisation across channels', tags=['Convolutional Networks'], diff='Easy',
            runtime='numpy', lessons=['dlcore:03.03'],
            related=['batch-normalization', 'layer-norm', 'pooling-layers'])

STATEMENT = r"""
AlexNet normalises each activation by the energy of the same pixel in **neighbouring channels**, a form of lateral inhibition borrowed from neuroscience:

$$b_{c,x,y} = a_{c,x,y} \Big/ \Big(k + \alpha \sum_{c'=\max(0,\,c - \lfloor n/2\rfloor)}^{\min(C-1,\,c + \lfloor n/2\rfloor)} a_{c',x,y}^2\Big)^{\beta}$$

Write `lrn(a, n=5, k=2.0, alpha=1e-4, beta=0.75)` for `a` of shape `(B, C, H, W)`, returning the same shape. The defaults are the paper's values. The window is clipped at the first and last channel. `n` is odd.

Note: this is the paper's formula. PyTorch's `nn.LocalResponseNorm` divides `alpha` by `n`, so the same arguments give a different result there.
"""

STARTER = '''import numpy as np

def lrn(a, n=5, k=2.0, alpha=1e-4, beta=0.75):
    """Local response normalisation over channels."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def lrn(a, n=5, k=2.0, alpha=1e-4, beta=0.75):
    a = np.asarray(a, dtype=float)
    C, h = a.shape[1], n // 2
    sq = np.pad(a * a, ((0, 0), (h, h), (0, 0), (0, 0)))         # zero channels outside the range
    csum = np.concatenate([np.zeros_like(sq[:, :1]), np.cumsum(sq, axis=1)], axis=1)
    window = csum[:, n:n + C] - csum[:, 0:C]                      # sum over channels c-h..c+h
    return a / (k + alpha * window) ** beta
'''

HINTS = [
    'Square the input once, then for each channel sum the squares of channels $c - n/2 .. c + n/2$. Zero-padding the channel axis by $\\lfloor n/2 \\rfloor$ on both sides handles the clipping.',
    'A cumulative sum along channels turns each window sum into one subtraction, so the whole thing is vectorised.',
]

_r = np.random.default_rng(2012)
TESTS = [
    (np.arange(1.0, 7.0).reshape(1, 6, 1, 1),),
    (np.arange(1.0, 7.0).reshape(1, 6, 1, 1), 3, 1.0, 0.1, 0.5),
    (_r.normal(size=(2, 8, 3, 3)) * 20,),
    (_r.normal(size=(1, 3, 2, 2)) * 50, 5, 2.0, 1e-3, 0.75),
    (_r.normal(size=(2, 10, 2, 4)), 7, 1.0, 0.5, 1.0),
    (np.zeros((1, 4, 2, 2)),),
]

EXPLANATION = r"""
## What it does
A strong activation in one channel suppresses the same location in nearby channels, so channels compete and the response is sharpened. With the paper's constants the effect is small unless activations are large (tens to hundreds), which they are after ReLU in an unnormalised network.

## History
AlexNet reported that LRN reduced top-1 error by about 1.4%. VGG found it gave no benefit, and batch normalisation (which normalises each channel over the batch, with learned scale and shift) replaced it entirely.

## Complexity
$O(BCHW)$ with the cumulative-sum trick; a naive loop over the window is $O(nBCHW)$.

## Common bugs
- Wrapping the window around the channel axis instead of clipping it.
- Summing over spatial neighbours instead of channels (that is a different, "within-channel" variant).
- Using PyTorch's $\alpha/n$ convention while claiming the paper's formula.

## Interview follow-ups
- Why does batch norm make LRN unnecessary?
- How would you implement LRN as a 1-D convolution over channels?
"""

import numpy as np

META = dict(title='Original U-Net: valid convolutions, crop-and-concat, weighted loss', tags=['U-Net', 'Computer Vision'],
            diff='Hard', runtime='numpy', lessons=['dlcore:03.02'],
            related=['tiny-u-net-forward-pass', 'u-net-skip-connection', 'transposed-convolution', 'zero-pad-and-center-crop'])

STATEMENT = r"""
The original U-Net uses **unpadded** ("valid") 3×3 convolutions, so every convolution shrinks the map by 2 pixels. The decoder's upsampled map is therefore smaller than the encoder map it is joined with, and the encoder map is **centre-cropped** before concatenation. Training uses a per-pixel **weight map** that emphasises thin borders between touching objects. Implement a two-level version and its loss.

Write `unet_valid(x, target, wmap, p)` returning `(logits, loss)`.

## Architecture (inference of the logits)
- `conv(z, W, b)`: valid 3×3 convolution with bias. `double(z, a, b)`: `relu(conv(relu(conv(z, *a)), *b))`.
- Encoder: $e = $ `double(x, p['e1'], p['e2'])`; pool: 2×2 max, stride 2.
- Bottom: $m = $ `double(pool(e), p['m1'], p['m2'])`.
- Up: 2×2 transposed convolution with stride 2, `p['up'] = (W, b)` with `W` `(C_in, C_out, 2, 2)`: output pixel $(2i + a, 2j + c)$ of channel $o$ is $\sum_{ch} m_{ch,i,j}W_{ch,o,a,c} + b_o$.
- Crop $e$ to the up-sampled size, centred: offset $\lfloor(H_e - H_u)/2\rfloor$ (same for width). Concatenate **[cropped $e$; up]** along channels.
- Decoder: `double(cat, p['d1'], p['d2'])`, then a 1×1 convolution `p['out'] = (W (K, C, 1, 1), b)` to $K$ class logits.

## Loss
`target` `(B, H_o, W_o)` int classes and `wmap` `(B, H_o, W_o)` weights, at the output resolution. Loss $= \frac{1}{B H_o W_o}\sum w_{b,i,j}\,\big(-\log\mathrm{softmax}(\text{logits}_{b,:,i,j})[\text{target}_{b,i,j}]\big)$.

Inputs are sized so that every pooled map has even height and width.
"""

STARTER = '''import numpy as np

def unet_valid(x, target, wmap, p):
    """Return (logits (B, K, Ho, Wo), loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv(z, W, b):
    B, C, H, Wd = z.shape
    O, _, k, _ = W.shape
    Ho, Wo = H - k + 1, Wd - k + 1
    out = np.zeros((B, O, Ho, Wo)) + np.asarray(b)[None, :, None, None]
    for i in range(k):
        for j in range(k):
            out += np.einsum('bchw,oc->bohw', z[:, :, i:i + Ho, j:j + Wo], W[:, :, i, j])
    return out

def _double(z, a, b):
    return np.maximum(_conv(np.maximum(_conv(z, *a), 0.0), *b), 0.0)

def _up(z, W, b):
    B, C, H, Wd = z.shape
    out = np.einsum('bchw,coxy->bohxwy', z, W).reshape(B, W.shape[1], 2 * H, 2 * Wd)
    return out + np.asarray(b)[None, :, None, None]

def unet_valid(x, target, wmap, p):
    x = np.asarray(x, dtype=float)
    e = _double(x, p['e1'], p['e2'])
    B, C, H, W = e.shape
    pooled = e.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
    u = _up(_double(pooled, p['m1'], p['m2']), *p['up'])
    oh, ow = (H - u.shape[2]) // 2, (W - u.shape[3]) // 2
    skip = e[:, :, oh:oh + u.shape[2], ow:ow + u.shape[3]]
    d = _double(np.concatenate([skip, u], axis=1), p['d1'], p['d2'])
    logits = _conv(d, *p['out'])
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    nll = -np.take_along_axis(logp, np.asarray(target)[:, None], axis=1)[:, 0]
    return logits, float((np.asarray(wmap) * nll).mean())
'''

HINTS = [
    'Track sizes: each valid 3×3 conv subtracts 2, the pool halves, the up-convolution doubles. For a 20×20 input: 16, 8, 4, 8, then 4 after the decoder.',
    'The 2×2 stride-2 transposed convolution has no overlap, so it is one einsum to `(B, O, H, 2, W, 2)` followed by a reshape to `(B, O, 2H, 2W)`.',
    'Crop the encoder map, not the decoder map, and put it **first** in the concatenation. The weighted loss is a plain mean over all pixels of `w * nll`.',
]

_r = np.random.default_rng(1505)
def _c(O, C, k=3):
    return (_r.normal(size=(O, C, k, k)) * np.sqrt(2 / (C * k * k)), 0.1 * _r.normal(size=O))
def _model(Cin, c1, c2, K):
    return dict(e1=_c(c1, Cin), e2=_c(c1, c1), m1=_c(c2, c1), m2=_c(c2, c2),
                up=(_r.normal(size=(c2, c1, 2, 2)) * np.sqrt(2 / (4 * c2)), 0.1 * _r.normal(size=c1)),
                d1=_c(c1, 2 * c1), d2=_c(c1, c1), out=_c(K, c1, 1))
_m1, _m2 = _model(1, 2, 3, 2), _model(2, 3, 4, 3)
TESTS = [
    (_r.normal(size=(1, 1, 20, 20)), _r.integers(0, 2, size=(1, 4, 4)), np.ones((1, 4, 4)), _m1),
    (_r.normal(size=(1, 1, 20, 20)), _r.integers(0, 2, size=(1, 4, 4)), 1 + 4 * _r.random((1, 4, 4)), _m1),
    (_r.normal(size=(2, 2, 28, 20)), _r.integers(0, 3, size=(2, 12, 4)), _r.random((2, 12, 4)), _m2),
    (_r.normal(size=(1, 2, 24, 24)), _r.integers(0, 3, size=(1, 8, 8)), np.zeros((1, 8, 8)), _m2),
    (_r.normal(size=(2, 1, 20, 28)), _r.integers(0, 2, size=(2, 4, 12)), 2 * np.ones((2, 4, 12)), _m1),
]

EXPLANATION = r"""
## Why valid convolutions and cropping
With no padding, every output pixel is computed only from real image pixels, never from invented zeros. The price is that the output is smaller than the input, so the paper predicts a centre tile from a larger input tile and segments big images with an **overlap-tile** strategy, mirroring the image at its borders to provide context. Because encoder maps are larger than the decoder maps at the same level, they are cropped before concatenation.

## The weight map
Cells in microscopy images often touch. The paper precomputes, for every training mask, a weight map that balances class frequencies and adds a large weight on background pixels squeezed between two cells (a function of the distances to the nearest and second-nearest cell). The network then learns to keep a thin separating border.

## Shapes in this version
For a 20×20 input: encoder 16×16, pooled 8×8, bottom 4×4, up-sampled 8×8, the encoder map is cropped from 16×16 to its centre 8×8, decoder 4×4. The real network repeats the pattern four times and turns a 572×572 tile into a 388×388 output.

## Complexity
Dominated by the 3×3 convolutions at full resolution: $O(B\,C^2\,9\,HW)$ for the first level.

## Common bugs
- Cropping the up-sampled decoder map instead of the encoder map.
- Concatenating in the wrong order, which scrambles the channels the decoder weights expect.
- Averaging the weighted loss by $\sum w$ instead of the pixel count (also valid, but a different loss).

## Interview follow-ups
- Modern U-Nets use padded convolutions and no cropping. What do they gain and lose?
- Derive the input size needed for a 4-level valid U-Net to produce a 388×388 output.
"""

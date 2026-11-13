import numpy as np

META = dict(title='U-Net decoder step: upsample, align, concatenate', tags=['U-Net', 'Convolutional Networks'], diff='Medium',
            runtime='numpy', lessons=['air-fanout-v1:03.06'],
            related=['transposed-convolution', 'resnet-residual-block', '2d-convolution'])

STATEMENT = r"""
Each decoder stage of a U-Net upsamples the coarse features with a 2x2 transposed convolution of stride 2, then concatenates the matching encoder features (the skip connection) along the channel axis. The two maps often differ by a few pixels: the original U-Net used unpadded convolutions, so the encoder map is **larger** and gets centre-cropped; modern variants instead **pad** the upsampled map to the encoder's size.

Write `unet_merge(x, skip, W, b, mode='crop')`.

## Upsampling
- `x`: `(N, C_in, h, w)`; `W`: `(C_in, C_up, 2, 2)` (PyTorch transposed-conv layout); `b`: `(C_up,)`.
- With kernel 2 and stride 2 the patches don't overlap: $u_{n,o,\,2i+a,\,2j+c} = b_o + \sum_{k} x_{n,k,i,j}\,W_{k,o,a,c}$ for $a, c \in \{0, 1\}$. So $u$ has shape `(N, C_up, 2h, 2w)`.

## Alignment (each spatial axis separately)
Let $\Delta = $ (skip size) $-$ (upsampled size) on that axis.
- `mode='crop'`: requires $\Delta \ge 0$. Crop the **skip** to the upsampled size, removing $\lfloor \Delta/2 \rfloor$ from the start and the rest from the end.
- `mode='pad'`: requires $\Delta \ge 0$. Zero-pad the **upsampled** map to the skip size, adding $\lfloor \Delta/2 \rfloor$ at the start and the rest at the end.

## Output
`np.concatenate([skip_aligned, up_aligned], axis=1)`: skip channels first. Shape `(N, C_skip + C_up, H', W')` where $H' \times W'$ is the upsampled size for `'crop'` and the skip size for `'pad'`.
"""

STARTER = '''import numpy as np

def unet_merge(x, skip, W, b, mode='crop'):
    """x: (N, Cin, h, w); skip: (N, Cs, Hs, Ws); W: (Cin, Cup, 2, 2). Return the concatenated map."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def unet_merge(x, skip, W, b, mode='crop'):
    x, skip, W = (np.asarray(a, dtype=float) for a in (x, skip, W))
    N, _, h, w = x.shape
    Cup = W.shape[1]
    # non-overlapping 2x2 stamps: (N, Cup, h, 2, w, 2) -> (N, Cup, 2h, 2w)
    u = np.einsum('nkij,koac->noiajc', x, W).reshape(N, Cup, 2 * h, 2 * w) + np.asarray(b, dtype=float)[None, :, None, None]
    dh, dw = skip.shape[2] - 2 * h, skip.shape[3] - 2 * w
    if mode == 'crop':
        skip = skip[:, :, dh // 2: dh // 2 + 2 * h, dw // 2: dw // 2 + 2 * w]
    else:
        u = np.pad(u, ((0, 0), (0, 0), (dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)))
    return np.concatenate([skip, u], axis=1)
'''

HINTS = [
    'A 2x2/stride-2 transposed conv writes each input pixel into its own 2x2 block. `einsum("nkij,koac->noiajc", x, W)` gives `(N, C_up, h, 2, w, 2)`, and reshaping to `(N, C_up, 2h, 2w)` interleaves the blocks correctly.',
    'Compute $\\Delta$ per axis from the skip shape and the upsampled shape. Floor half of it goes at the start, the remainder at the end.',
    'Crop with a slice `[dh//2 : dh//2 + 2h]` on the skip, or pad with `np.pad(u, ((0,0), (0,0), (dh//2, dh - dh//2), (dw//2, dw - dw//2)))`. Concatenate skip first.',
]

_r = np.random.default_rng(137)
TESTS = [
    (np.array([[[[1.0]]]]), np.arange(16.0).reshape(1, 1, 4, 4), np.ones((1, 1, 2, 2)), np.zeros(1)),
    (np.array([[[[1.0]]]]), np.zeros((1, 1, 3, 3)), np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), np.zeros(1), 'pad'),
    (_r.normal(size=(2, 3, 2, 2)), _r.normal(size=(2, 2, 4, 4)), _r.normal(size=(3, 2, 2, 2)), _r.normal(size=2)),
    (_r.normal(size=(1, 4, 3, 2)), _r.normal(size=(1, 3, 10, 7)), _r.normal(size=(4, 2, 2, 2)), _r.normal(size=2)),
    (_r.normal(size=(1, 4, 3, 2)), _r.normal(size=(1, 3, 7, 5)), _r.normal(size=(4, 2, 2, 2)), _r.normal(size=2), 'pad'),
    (_r.normal(size=(2, 2, 2, 3)), _r.normal(size=(2, 1, 4, 6)), _r.normal(size=(2, 3, 2, 2)), np.zeros(3), 'pad'),
    (_r.normal(size=(1, 8, 4, 4)), _r.normal(size=(1, 4, 13, 9)), _r.normal(size=(8, 4, 2, 2)), _r.normal(size=4)),
]

EXPLANATION = r"""
## Why skip connections
The encoder throws away resolution to gain context; the decoder must put back precise boundaries. Concatenating the encoder's high-resolution features gives the decoder both: *what* is there (from the upsampled path) and *where* exactly (from the skip). In diffusion models the same U-Net shape carries fine detail of the noisy image straight to the output.

## Crop vs pad
With unpadded 3x3 convs, each conv shaves 2 pixels, so encoder maps are larger than their decoder counterparts and must be centre-cropped (the original 572 -> 388 output sizes). With "same" convs the only mismatch comes from odd sizes being floored by pooling: a 7-pixel map pools to 3 and upsamples to 6, so the decoder pads by 1. Putting the odd pixel at the end matches `F.pad(u, [dw//2, dw - dw//2, ...])` used in common implementations.

## 2x2 / stride 2 is a reshape
Because the stamps don't overlap, the transposed conv is a per-pixel matrix product followed by a reshape; no accumulation and no checkerboard artefacts.

## Complexity
$O(N C_\text{in} C_\text{up} \cdot 4hw)$ for the upsampling; cropping and padding are copies.

## Common bugs
- Reshaping `(N, C, 2, 2, h, w)` instead of `(N, C, h, 2, w, 2)`: the right shape, scrambled pixels.
- Cropping from one side only, which shifts the skip features by a pixel relative to the decoder.
- Concatenating in the opposite channel order from the one the next conv's weights expect.

## Interview follow-ups
- Why concatenate (U-Net) instead of add (ResNet, FPN) at the skip? What does each cost?
- Where would you inject a diffusion timestep embedding in this block?
"""

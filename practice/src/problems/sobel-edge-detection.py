import numpy as np

META = dict(title='Sobel gradients and edge magnitude', tags=['Computer Vision', 'Convolutional Networks'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:04.02'], related=['gaussian-blur-2d', 'multichannel-2d-convolution'])

STATEMENT = r"""
The Sobel operator estimates image gradients with two small kernels: a central difference in one direction, smoothed with weights `1, 2, 1` in the other.

$$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad K_y = K_x^\top$$

Write `sobel(img)` and return `(gx, gy, mag)`.

## Definition
Pad the image by one pixel on every side by **replicating the border** (`np.pad(..., mode="edge")`), call the result `P` (so `P[i+1, j+1] = img[i, j]`). Then apply each kernel as a **cross-correlation** (no kernel flip):

$$g_x[i, j] = \sum_{a=0}^{2} \sum_{b=0}^{2} K_x[a, b]\; P[i + a,\, j + b]$$

and the same with $K_y$ for $g_y$. So $g_x > 0$ where brightness increases to the **right**, and $g_y > 0$ where it increases **downward**. The magnitude is $\sqrt{g_x^2 + g_y^2}$.

## Input / output
- `img`: 2-D array `(H, W)` of any numeric dtype, `H, W >= 1`.
- Return three `float64` arrays, each of shape `(H, W)`.
"""

STARTER = '''import numpy as np

def sobel(img):
    """img: (H, W). Return (gx, gy, mag), each float64 (H, W), with edge-replicated borders."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

def _correlate3(P, K, H, W):
    # Sum of 9 shifted views of the padded image: no loops over pixels.
    return sum(K[a, b] * P[a:a + H, b:b + W] for a in range(3) for b in range(3))

def sobel(img):
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    P = np.pad(img, 1, mode='edge')
    gx = _correlate3(P, KX, H, W)
    gy = _correlate3(P, KX.T, H, W)
    return gx, gy, np.hypot(gx, gy)
'''

HINTS = [
    'After padding by one pixel, the output at `(i, j)` only needs the 3x3 block `P[i:i+3, j:j+3]`.',
    'Instead of looping over pixels, loop over the 9 kernel taps: tap `(a, b)` contributes `K[a, b] * P[a:a+H, b:b+W]` to every output pixel at once.',
    'Pad with `np.pad(img, 1, mode="edge")`, sum the 9 weighted slices for `KX` and for `KX.T`, and use `np.hypot(gx, gy)` for the magnitude.',
]

_r = np.random.default_rng(16)
TESTS = [
    (np.array([[0, 0, 1, 1]] * 3),),
    (np.array([[0.0], [1.0], [2.0]]),),
    (np.full((4, 5), 3.0),),
    (np.array([[5.0]]),),
    (np.tril(np.ones((5, 5))),),
    (_r.integers(0, 256, size=(6, 7), dtype=np.uint8),),
    (_r.normal(size=(8, 4)),),
]

EXPLANATION = r"""
## Smoothing plus differencing
$K_x$ factorises as the outer product $[1, 2, 1]^\top [-1, 0, 1]$: a vertical blur times a horizontal central difference. The blur makes the derivative less sensitive to pixel noise, which is why Sobel beats a plain $[-1, 0, 1]$ filter on real images. The factorisation also means you can run it as two 1-D passes: $6$ multiply-adds per pixel instead of $9$.

## Borders
Replicating the edge means the derivative across the border is zero, so a constant image gives zero gradient everywhere (test 3) instead of fake edges at the frame. Zero padding would report a strong edge all around the image.

## Complexity
$O(HW)$ with a constant of 9 taps per kernel; memory $O(HW)$.

## Common bugs
- Flipping the kernel (true convolution) silently negates $g_x$ and $g_y$. The magnitude is unchanged, which hides the bug until someone uses the direction.
- Integer arithmetic on `uint8` input: negative gradients wrap around.
- Mixing up which kernel goes with which axis: $K_x$ varies along columns (index $b$).

## Interview follow-ups
- How would you turn the magnitude into thin edges? (Non-maximum suppression along the gradient direction, then hysteresis thresholding: the Canny detector.)
- How is this related to a CNN's first layer? (Learned first-layer filters often look like oriented derivative filters; Sobel is a hand-designed one.)
"""

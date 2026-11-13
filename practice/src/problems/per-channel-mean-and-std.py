import numpy as np

META = dict(title='Dataset per-channel mean and std', tags=['Computer Vision', 'Probability and Statistics'], diff='Easy', runtime='numpy',
            lessons=[], related=['image-normalize', 'rgb-to-grayscale'])

STATEMENT = r"""
Before training you measure the per-channel mean and standard deviation of the training images, so you can feed them to the normalisation step. Datasets often contain images of **different sizes**, and every pixel should count equally.

Write `channel_stats(images)` and return `(mean, std)`.

## Input
- `images`: a Python list of `uint8` arrays, each of shape `(H_i, W_i, C)`. Sizes may differ between images; `C` is the same for all. The list is non-empty.

## Output
Two `float64` arrays of shape `(C,)`, measured on the **0-1 scale** (divide pixel values by 255):
- `mean[c]`: average of channel `c` over **all pixels of all images** pooled together.
- `std[c]`: the population standard deviation (divide by the pixel count, not count minus 1) of the same pooled values.

## Example
One `1 x 2` image with pixels `[255, 0]` and `[0, 0]` (`C = 2`) has `mean = [0.5, 0.0]` and `std = [0.5, 0.0]`.

Averaging each image's own mean would be wrong here: a big image must weigh more than a small one.
"""

STARTER = '''import numpy as np

def channel_stats(images):
    """images: list of uint8 (H_i, W_i, C). Return (mean (C,), std (C,)) over all pixels, 0-1 scale."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def channel_stats(images):
    C = images[0].shape[-1]
    # Flatten every image to (pixels, C) and pool them, so each pixel has equal weight.
    pixels = np.concatenate([np.asarray(im, dtype=np.float64).reshape(-1, C) for im in images]) / 255.0
    return pixels.mean(axis=0), pixels.std(axis=0)    # std defaults to ddof=0 (population)
'''

HINTS = [
    'Image boundaries do not matter for these statistics. Only the pool of pixel values per channel does.',
    'Reshape each image to `(H_i * W_i, C)` and stack them all with `np.concatenate` before reducing over axis 0.',
    'Divide by 255 in float64, then `pixels.mean(axis=0)` and `pixels.std(axis=0)` (ddof 0 is NumPy\'s default).',
]

_r = np.random.default_rng(13)
TESTS = [
    ([np.array([[[255, 0], [0, 0]]], dtype=np.uint8)],),
    ([np.full((1, 1, 1), 255, dtype=np.uint8), np.zeros((1, 3, 1), dtype=np.uint8)],),
    ([np.full((2, 2, 3), 7, dtype=np.uint8)],),
    ([_r.integers(0, 256, size=(4, 5, 3), dtype=np.uint8), _r.integers(0, 256, size=(2, 7, 3), dtype=np.uint8),
      _r.integers(0, 256, size=(1, 1, 3), dtype=np.uint8)],),
    ([_r.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)],),
    ([np.array([[[0, 255, 10, 20]]], dtype=np.uint8), np.array([[[255, 0, 30, 40]], [[255, 0, 50, 60]]], dtype=np.uint8)],),
]

EXPLANATION = r"""
## Pixel-weighted pooling
The mean you normalise with should be the expected value of a pixel drawn from the training distribution. Pooling all pixels gives exactly that. Averaging per-image means gives a different number whenever image sizes differ (test 2: the pooled mean is $0.25$, the mean of means is $0.5$).

## Scaling to large datasets
You can't concatenate ImageNet in memory. Instead accumulate three running quantities per channel: the pixel count $n$, the sum $S$ and the sum of squares $Q$; then $\mu = S/n$ and $\sigma^2 = Q/n - \mu^2$. In float32 that subtraction cancels badly, so use float64 accumulators or Welford/Chan's parallel update, which merges `(n, mean, M2)` triples from batches exactly.

## Complexity
$O(\text{total pixels} \cdot C)$ time. This version uses memory for every pixel; the streaming version uses $O(C)$.

## Common bugs
- Averaging per-image means or stds.
- Using `ddof=1`. With millions of pixels it barely matters, but it isn't what the statement asks.
- Computing in `uint8`: summing overflows immediately.
- Averaging per-image **stds**: that ignores the spread between images' means.

## Interview follow-ups
- Implement the streaming version over a dataloader in batches. How do you combine two batches' `(n, mean, M2)`?
- Should you compute statistics before or after augmentation? (Usually on the raw training images, resized the way they are at test time.)
"""

import numpy as np

META = dict(title='CutMix with area-corrected labels', tags=['Computer Vision', 'Training'], diff='Medium', runtime='numpy',
            lessons=['air-fanout-v1:03.05'], related=['mixup-augmentation', 'zero-pad-and-center-crop'])

STATEMENT = r"""
**CutMix** pastes a rectangle from a partner image into each image and mixes the labels in proportion to the area each image keeps. The rectangle is clipped at the image border, so the label weight must be computed from the **actual** pasted area, not from the requested one.

Write `cutmix(images, labels, num_classes, lam, perm, cy, cx)` and return `(mixed, targets, lam_adj)`.

## Box
With image size `H x W`:
1. `r = sqrt(1 - lam)`, `cut_h = int(H * r)`, `cut_w = int(W * r)` (truncation toward zero).
2. `y1 = clip(cy - cut_h // 2, 0, H)`, `y2 = clip(cy + cut_h // 2, 0, H)`, and likewise `x1, x2` from `cx`, `cut_w`, `W`.
3. Rows `y1 .. y2 - 1` and columns `x1 .. x2 - 1` of image `i` are replaced by the same pixels of image `perm[i]` (all channels). The box may be empty.

## Labels
`lam_adj = 1 - (y2 - y1) * (x2 - x1) / (H * W)` and `targets[i] = lam_adj * onehot(labels[i]) + (1 - lam_adj) * onehot(labels[perm[i]])`.

## Input / output
- `images`: array `(B, H, W, C)`, any dtype. Do not modify it in place.
- `labels`: int `(B,)`; `lam`: float in `[0, 1]`; `perm`: a permutation of `0..B-1`; `cy, cx`: ints with `0 <= cy < H`, `0 <= cx < W` (the box centre).
- `mixed`: same shape and dtype as `images`. `targets`: `float64` `(B, num_classes)`. `lam_adj`: Python float.
"""

STARTER = '''import numpy as np

def cutmix(images, labels, num_classes, lam, perm, cy, cx):
    """Return (mixed images, targets (B, num_classes), lam_adj)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def cutmix(images, labels, num_classes, lam, perm, cy, cx):
    images = np.asarray(images)
    B, H, W = images.shape[:3]
    r = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * r), int(W * r)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, H)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, W)

    mixed = images.copy()                                   # never mutate the caller's batch
    mixed[:, y1:y2, x1:x2] = images[perm, y1:y2, x1:x2]
    lam_adj = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)         # the area actually kept

    onehot = np.eye(num_classes)[np.asarray(labels)]
    targets = lam_adj * onehot + (1 - lam_adj) * onehot[perm]
    return mixed, targets, float(lam_adj)
'''

HINTS = [
    'Compute the box once for the whole batch; every image gets the same rectangle from its own partner.',
    'Paste with one slice assignment on a copy: `mixed[:, y1:y2, x1:x2] = images[perm, y1:y2, x1:x2]`.',
    'Recompute the label weight from the clipped box: `lam_adj = 1 - (y2 - y1) * (x2 - x1) / (H * W)`, then mix one-hot labels with it.',
]

_r = np.random.default_rng(33)
_img = np.stack([np.full((4, 4, 1), 1), np.full((4, 4, 1), 2)]).astype(np.uint8)
TESTS = [
    (_img, np.array([0, 1]), 2, 0.75, np.array([1, 0]), 2, 2),
    (_img, np.array([0, 1]), 2, 0.75, np.array([1, 0]), 0, 3),
    (_img, np.array([0, 1]), 2, 1.0, np.array([1, 0]), 1, 1),
    (_r.integers(0, 256, size=(3, 6, 8, 3), dtype=np.uint8), np.array([0, 1, 2]), 3, 0.4, np.array([2, 0, 1]), 3, 4),
    (_r.normal(size=(4, 10, 10, 2)), _r.integers(0, 5, size=4), 5, 0.0, np.array([1, 2, 3, 0]), 5, 5),
    (_r.normal(size=(4, 9, 7, 1)), np.array([1, 1, 0, 2]), 3, 0.37, np.array([3, 2, 1, 0]), 8, 0),
]

EXPLANATION = r"""
## Why correct the label weight
The requested box covers a fraction $1 - \lambda$ of the image, but near the border it gets clipped, so the partner contributes less than $1 - \lambda$ of the pixels. Using the original $\lambda$ would tell the model the pasted patch matters more than it visibly does. Test 2 clips the box at the top edge, so only 2 of the 4 requested pixels are pasted. Note also that `cut_h // 2` is used on both sides, so an odd `cut_h` produces a box one row shorter than requested even away from the border: another reason to compute the area from the final coordinates.

## CutMix vs MixUp and Cutout
- MixUp blends whole images, which creates unnatural ghosted pictures.
- Cutout erases a patch, wasting those pixels.
- CutMix fills the patch with real content from another image, so every pixel stays informative and the model learns to recognise objects from partial views.

## Complexity
$O(B \cdot \text{patch size})$ for the paste plus one copy of the batch.

## Common bugs
- Writing into `images` directly while reading from `images[perm]`: earlier pastes then leak into later partners.
- Using `lam` instead of `lam_adj` for the targets.
- Sampling the box with its corner instead of its centre at `(cy, cx)`, which biases boxes toward the bottom-right.

## Interview follow-ups
- How would you apply CutMix to a detection or segmentation batch? (The pasted region carries its masks and the boxes inside it; boxes cut by the border must be clipped or dropped.)
- In practice `cy, cx` are uniform over the image. What's the expected label weight given $\lambda$?
"""

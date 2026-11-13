import numpy as np

META = dict(title='Compose an image transform pipeline', tags=['Computer Vision', 'Data Processing'], diff='Easy', runtime='torch',
            lessons=['air-fanout-v1:04.08', 'air-fanout-v1:11.12'],
            related=['image-normalize', 'custom-dataset-class', 'per-channel-mean-and-std'])

STATEMENT = r"""
Image datasets apply a chain of small transforms to every sample. Write the transforms as callable classes and a `Compose` that runs them in order, the way `torchvision.transforms` is built.

## The steps
Each step is a tuple: a name followed by its arguments.

| Step | Effect |
|---|---|
| `("to_tensor",)` | uint8 image `(H, W, C)` to float64 `(C, H, W)` in `[0, 1]` (divide by 255) |
| `("normalize", mean, std)` | per channel: `(x - mean[c]) / std[c]`; `mean`, `std` are lists of length `C` |
| `("hflip",)` | mirror left-right (reverse the width axis) |
| `("center_crop", h, w)` | keep rows `top : top + h` and columns `left : left + w`, with `top = (H - h) // 2`, `left = (W - w) // 2` |
| `("pad", p, value)` | add `p` pixels of constant `value` on all four sides |

Every pipeline starts with `("to_tensor",)`; the rest come in any order and may repeat. All spatial steps act on the last two axes of a `(C, H, W)` tensor. Crops are never larger than the current image.

## The function that's graded
Write `apply_pipeline(img, steps)`:

- `img`: uint8 tensor `(H, W, C)`; `steps`: a list of step tuples.
- Build one transform object per step, `Compose` them, and return the result for `img` as a float64 `(C, H', W')` tensor.

This problem runs in **Colab** (PyTorch isn't available in the browser). The notebook includes the same tests.
"""

STARTER = '''import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        raise NotImplementedError

def apply_pipeline(img, steps):
    """img: uint8 (H, W, C). Return float64 (C, H', W') after every step."""
    raise NotImplementedError
'''

SOLUTION = '''import torch
import torch.nn.functional as F

class ToTensor:
    def __call__(self, img):
        return img.permute(2, 0, 1).to(torch.float64) / 255.0       # HWC uint8 -> CHW float in [0, 1]

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float64).view(-1, 1, 1)   # (C, 1, 1) broadcasts over H, W
        self.std = torch.tensor(std, dtype=torch.float64).view(-1, 1, 1)

    def __call__(self, x):
        return (x - self.mean) / self.std

class HFlip:
    def __call__(self, x):
        return x.flip(-1)

class CenterCrop:
    def __init__(self, h, w):
        self.h, self.w = h, w

    def __call__(self, x):
        H, W = x.shape[-2:]
        top, left = (H - self.h) // 2, (W - self.w) // 2
        return x[..., top:top + self.h, left:left + self.w]

class Pad:
    def __init__(self, p, value):
        self.p, self.value = p, value

    def __call__(self, x):
        return F.pad(x, (self.p,) * 4, value=self.value)            # (left, right, top, bottom)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

REGISTRY = {'to_tensor': ToTensor, 'normalize': Normalize, 'hflip': HFlip, 'center_crop': CenterCrop, 'pad': Pad}

def apply_pipeline(img, steps):
    pipeline = Compose([REGISTRY[name](*args) for name, *args in steps])
    return pipeline(img)
'''

HINTS = [
    'Each transform is a class whose `__init__` stores its arguments and whose `__call__(x)` returns the new tensor. `Compose.__call__` just feeds the output of one into the next.',
    '`to_tensor` is `permute(2, 0, 1)`, a cast to float64 and a division by 255. Do the cast before dividing: integer division of uint8 truncates.',
    'Reshape `mean`/`std` to `(C, 1, 1)` so they broadcast over height and width. `F.pad(x, (p, p, p, p), value=v)` pads the last two axes; `x.flip(-1)` mirrors.',
]

_r = np.random.default_rng(8)
TESTS = [
    (np.array([[[0, 255]], [[51, 102]]], dtype=np.uint8), [("to_tensor",), ("hflip",)]),
    (np.array([[[255]]], dtype=np.uint8), [("to_tensor",), ("normalize", [0.5], [0.25]), ("pad", 1, 0.0)]),
    (_r.integers(0, 256, size=(5, 6, 3), dtype=np.uint8),
     [("to_tensor",), ("center_crop", 3, 3), ("normalize", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    (_r.integers(0, 256, size=(4, 4, 1), dtype=np.uint8), [("to_tensor",)]),
    (_r.integers(0, 256, size=(3, 5, 2), dtype=np.uint8),
     [("to_tensor",), ("pad", 2, -1.0), ("center_crop", 4, 6), ("hflip",)]),
    (_r.integers(0, 256, size=(6, 7, 3), dtype=np.uint8),
     [("to_tensor",), ("hflip",), ("hflip",), ("center_crop", 6, 7)]),
    (_r.integers(0, 256, size=(8, 8, 3), dtype=np.uint8),
     [("to_tensor",), ("center_crop", 5, 4), ("pad", 1, 0.5), ("normalize", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ("hflip",)]),
]

EXPLANATION = r"""
## Why it works
Making each transform an object with `__call__` separates configuration (crop size, statistics) from application, so a pipeline can be built once from a config and applied to millions of images. `Compose` is just function composition; the registry turns a declarative list of steps into objects. That's the same design as `torchvision.transforms.v2.Compose`.

## Order matters
- Normalising before padding pads with raw `value`; padding first and then normalising turns that value into `(value - mean) / std`. Test 2 pads after normalising, test 7 before.
- `center_crop` floors the offset, so an odd leftover puts the extra row or column at the bottom/right.
- Two horizontal flips are the identity (test 6), a quick sanity check for any augmentation code.

## Complexity
Each step is one pass, $O(CHW)$. In a real loader these run in `DataLoader` workers on the CPU, so a heavy pipeline can starve the GPU; moving augmentation to the GPU on whole batches is the usual fix.

## Common bugs
- Dividing the uint8 tensor by 255 before casting.
- Normalising with ImageNet statistics on images that weren't scaled to `[0, 1]` first.
- Applying a random flip to an image but not to its segmentation mask or boxes.

## Interview follow-ups
- How would you make random transforms (random crop, flip with probability 0.5) reproducible across workers?
- Which transforms belong in the validation pipeline, and which must be left out?
"""

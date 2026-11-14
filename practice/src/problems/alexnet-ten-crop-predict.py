import numpy as np

META = dict(title='A mini AlexNet with ten-crop test-time averaging', tags=['Convolutional Networks', 'Computer Vision'],
            diff='Hard', runtime='numpy', lessons=['dlcore:03.03'],
            related=['local-response-normalization', 'pca-color-augmentation', 'max-pool-2d', 'dropout'])

STATEMENT = r"""
AlexNet's recipe: strided first convolution, ReLU, local response normalisation, **overlapping** max pooling (3×3 windows with stride 2), fully connected layers with dropout, and at test time an average of the softmax over **ten crops** (four corners and the centre, plus their horizontal mirrors). Build a miniature version of the whole pipeline.

Write `alexnet_predict(img, p, crop)` returning `(probs, top)`: the averaged class probabilities `(K,)` and their argmax.

## Network on one crop `z` `(C, crop, crop)`
1. `conv(z, *p['c1'], stride=p['s1'], pad=0)` → ReLU → LRN → maxpool(3, 2).
2. `conv(·, *p['c2'], stride=1, pad=1)` → ReLU → LRN → maxpool(3, 2).
3. `conv(·, *p['c3'], stride=1, pad=1)` → ReLU.
4. Flatten in `(C, H, W)` order → `@ W + b` with `p['f1']` → ReLU → **dropout at test time** → `p['f2']` → logits.

- `conv` is cross-correlation with bias; weights `(O, C, k, k)`. Max pooling has no padding; output size $\lfloor (H - 3)/2\rfloor + 1$.
- LRN uses the paper's constants: $b_c = a_c/(2 + 10^{-4}\sum_{|c'-c|\le 2} a_{c'}^2)^{0.75}$, window clipped at the channel ends.
- AlexNet trained with dropout $p = 0.5$ **without** rescaling during training, so at test time the outputs of the dropout layer are multiplied by $0.5$.

## Ten crops
`img` is `(C, S, S)` with $S \ge$ `crop`. Crops: top-left, top-right, bottom-left, bottom-right, centre (offset $\lfloor (S - \text{crop})/2 \rfloor$), and the same five flipped left-right. `probs` is the mean of the ten softmax vectors.
"""

STARTER = '''import numpy as np

def alexnet_predict(img, p, crop):
    """Return (probs (K,), top int)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _conv(z, W, b, stride, pad):
    C, H, Wd = z.shape
    O, _, k, _ = W.shape
    zp = np.pad(z, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1
    out = np.zeros((O, Ho, Wo)) + np.asarray(b)[:, None, None]
    for i in range(k):
        for j in range(k):
            out += np.einsum('chw,oc->ohw', zp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride], W[:, :, i, j])
    return out

def _lrn(a, n=5, k=2.0, alpha=1e-4, beta=0.75):
    C, h = a.shape[0], n // 2
    sq = np.pad(a * a, ((h, h), (0, 0), (0, 0)))
    win = sum(sq[j:j + C] for j in range(n))
    return a / (k + alpha * win) ** beta

def _pool(a, k=3, s=2):
    C, H, W = a.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    out = np.full((C, Ho, Wo), -np.inf)
    for i in range(k):
        for j in range(k):
            out = np.maximum(out, a[:, i:i + s * Ho:s, j:j + s * Wo:s])
    return out

def _net(z, p):
    relu = lambda u: np.maximum(u, 0.0)
    z = _pool(_lrn(relu(_conv(z, *p['c1'], p['s1'], 0))))
    z = _pool(_lrn(relu(_conv(z, *p['c2'], 1, 1))))
    z = relu(_conv(z, *p['c3'], 1, 1))
    h = 0.5 * relu(z.reshape(-1) @ p['f1'][0] + p['f1'][1])    # test-time dropout scaling
    return h @ p['f2'][0] + p['f2'][1]

def alexnet_predict(img, p, crop):
    img = np.asarray(img, dtype=float)
    S = img.shape[1]
    e, c = S - crop, (S - crop) // 2
    crops = [img[:, a:a + crop, b:b + crop] for a, b in ((0, 0), (0, e), (e, 0), (e, e), (c, c))]
    crops += [x[:, :, ::-1] for x in crops]
    probs = np.zeros(len(p['f2'][1]))
    for z in crops:
        lg = _net(z, p)
        w = np.exp(lg - lg.max())
        probs += w / w.sum()
    probs /= len(crops)
    return probs, int(np.argmax(probs))
'''

HINTS = [
    'Get one crop through the network first and check the shapes after every stage; then the ten-crop loop is a list of slices plus their `[:, :, ::-1]` mirrors.',
    'Overlapping pooling is ordinary max pooling with window 3 and stride 2, so neighbouring windows share a row or column.',
    'The dropout layer sits after the first fully connected layer\'s ReLU. Because training did not rescale kept units, inference multiplies that layer\'s output by $1 - p = 0.5$.',
]

_r = np.random.default_rng(2012)
def _cv(O, C, k):
    return (_r.normal(size=(O, C, k, k)) * np.sqrt(2 / (C * k * k)), 0.1 * _r.normal(size=O))
def _fc(a, b):
    return (_r.normal(size=(a, b)) * np.sqrt(2 / a), 0.1 * _r.normal(size=b))
# crop 15, s1 2, k1 3: 15 -> 7 -> pool 3 -> conv 3 -> pool 1 -> conv 1
_p1 = dict(c1=_cv(4, 3, 3), s1=2, c2=_cv(6, 4, 3), c3=_cv(5, 6, 3), f1=_fc(5, 8), f2=_fc(8, 4))
# crop 19, s1 2, k1 3: 19 -> 9 -> 4 -> 4 -> 1 -> 1
_p2 = dict(c1=_cv(6, 3, 3), s1=2, c2=_cv(8, 6, 3), c3=_cv(8, 8, 3), f1=_fc(8, 10), f2=_fc(10, 5))
_p3 = dict(_p2, c1=(_p2['c1'][0] * 20, _p2['c1'][1]))
TESTS = [
    (_r.normal(size=(3, 15, 15)), _p1, 15),
    (_r.normal(size=(3, 18, 18)), _p1, 15),
    (_r.normal(size=(3, 22, 22)), _p2, 19),
    (_r.normal(size=(3, 24, 24)), _p2, 19),
    (_r.normal(size=(3, 24, 24)), _p3, 19),
    (np.tile(np.linspace(-1, 1, 20), (3, 20, 1)), _p2, 19),
]

EXPLANATION = r"""
## What AlexNet combined
None of the pieces was new on its own; the 2012 result came from putting them together at scale on two GPUs: ReLU instead of tanh (several times faster training), heavy augmentation (random 224 crops of 256 images, mirroring, PCA colour shifts), dropout in the large fully connected layers, LRN, and overlapping pooling (about 0.4% better top-1 than non-overlapping).

## Test-time augmentation
Averaging predictions over ten crops reduces the error a single centre crop makes on off-centre objects. It costs ten forward passes per image; the paper reports it as part of its headline numbers.

## Two dropout conventions
- AlexNet: drop units at random in training, multiply by $(1-p)$ at test time.
- Modern "inverted" dropout: divide the kept units by $(1-p)$ during training, do nothing at test time.
Both give the same expected activation; the inverted form keeps inference code unchanged.

## Complexity
Dominated by the convolutions; the real network's 60M parameters are mostly in the fully connected layers.

## Common bugs
- Forgetting the 0.5 test-time factor (or applying it to the logits instead of the dropout layer's output).
- Mirroring along the wrong axis (`::-1` on height flips the image upside down).
- Averaging logits instead of probabilities (a different ensemble).

## Interview follow-ups
- What would you change to train this network today (normalisation, pooling, head)?
- How would you batch the ten crops to make test-time augmentation cheap on a GPU?
"""

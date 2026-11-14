import numpy as np

META = dict(title='PCA colour augmentation ("fancy PCA")', tags=['Computer Vision', 'Linear Algebra'], diff='Medium',
            runtime='numpy', lessons=['dlcore:03.03'],
            related=['eigendecomposition', 'per-channel-mean-and-std', 'pca', 'mixup-augmentation'])

STATEMENT = r"""
AlexNet augments its training images by shifting the colour of every pixel along the principal components of the RGB values seen across the training set. This mimics changes in lighting intensity and colour while keeping object identity.

Write `pca_color_augment(images, alphas)`.

## Steps
1. Treat every pixel of every image in `images` `(N, H, W, 3)` as one RGB sample and compute their $3\times3$ covariance matrix (subtract the mean, divide by the number of pixels).
2. Eigendecompose it: eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ with unit eigenvectors $p_1, p_2, p_3$. Fix each eigenvector's sign so that its entry with the largest absolute value is positive.
3. For image $i$ with draws `alphas[i]` $= (\alpha_1, \alpha_2, \alpha_3)$ (in training these are $\mathcal N(0, 0.1^2)$), add the same vector $\sum_j \alpha_j\lambda_j\,p_j$ to every pixel.

Return `(augmented, eigvals)`: the augmented images `(N, H, W, 3)` and the eigenvalues in descending order. No clipping.
"""

STARTER = '''import numpy as np

def pca_color_augment(images, alphas):
    """Return (augmented (N, H, W, 3), eigvals (3,) descending)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def pca_color_augment(images, alphas):
    x = np.asarray(images, dtype=float)
    pix = x.reshape(-1, 3)
    cov = np.cov(pix, rowvar=False, bias=True)
    lam, P = np.linalg.eigh(cov)                       # ascending
    lam, P = lam[::-1], P[:, ::-1]
    big = np.argmax(np.abs(P), axis=0)
    P = P * np.sign(P[big, np.arange(3)])              # largest-magnitude entry positive
    shift = (np.asarray(alphas, dtype=float) * lam) @ P.T  # (N, 3): sum_j alpha_j lambda_j p_j
    return x + shift[:, None, None, :], lam
'''

HINTS = [
    'Reshape to `(N·H·W, 3)` and use `np.cov(..., rowvar=False, bias=True)` for the population covariance.',
    '`np.linalg.eigh` returns eigenvalues in ascending order with eigenvectors as **columns**. Reverse both, then flip each column\'s sign so its largest-magnitude entry is positive; the shift depends on that sign.',
    'The shift for image $i$ is `(alphas[i] * lam) @ P.T`; broadcast it over the height and width axes.',
]

_r = np.random.default_rng(2012)
_base = _r.random((3, 4, 4, 3))
_corr = _r.random((2, 3, 3, 1)) * np.array([1.0, 0.8, 0.3]) + 0.05 * _r.normal(size=(2, 3, 3, 3))
TESTS = [
    (_base, np.zeros((3, 3))),
    (_base, np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.2]])),
    (_base * 255, _r.normal(size=(3, 3)) * 0.1),
    (_corr, np.array([[1.0, 0.0, 0.0], [-0.5, 0.5, 0.5]])),
    (_r.random((1, 6, 5, 3)), _r.normal(size=(1, 3)) * 0.1),
]

EXPLANATION = r"""
## Why the principal components
Natural changes in illumination move pixel colours mostly along the directions in which colours already vary the most across the dataset (roughly brightness first, then warm-cool shifts). Scaling the shift by the eigenvalues keeps the perturbation proportional to that natural variation. AlexNet reported over 1% lower top-1 error from this augmentation.

## Why fix the sign
An eigenvector is only defined up to sign, and different libraries (or the same library on different hardware) can return $-p$ instead of $p$. Since $\alpha$ is drawn from a symmetric distribution this doesn't matter in training, but it makes a deterministic test (and reproducible augmentation) impossible unless a convention is fixed.

## Complexity
One pass over all pixels for the covariance, a 3×3 eigendecomposition, and one addition per pixel. In practice the eigendecomposition is done once for the whole training set.

## Common bugs
- Computing the covariance per image instead of over the dataset.
- Using rows of the eigenvector matrix instead of columns.
- Drawing a fresh $\alpha$ per pixel instead of per image.

## Interview follow-ups
- How does this compare with simple brightness/contrast jitter?
- Why might this augmentation hurt on tasks where colour is the label (e.g. ripeness detection)?
"""

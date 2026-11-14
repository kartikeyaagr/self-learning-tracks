import numpy as np

META = dict(title='Detect GAN mode collapse with mode coverage', tags=['Generative Models', 'Metrics & Evaluation'], diff='Medium', runtime='numpy')

STATEMENT = r"""
A standard check on a GAN trained on a mixture of Gaussians is how many modes its samples reach. Write `mode_coverage(samples, centers, radius, min_frac)`.

- Assign each sample to its nearest centre (Euclidean). A sample counts as **high quality** if that distance is at most `radius`.
- A mode is **covered** if it receives at least `min_frac` of the high-quality samples (strictly, count `>= min_frac * n_hq`).
- Also compute the KL divergence $\sum_k q_k \log(q_k / u_k)$ between the distribution $q$ of high-quality samples over modes and the uniform distribution $u_k = 1/K$, skipping terms with $q_k = 0$.

Return a dict `{"covered": int, "hq_rate": float, "kl": float}`. With no high-quality samples, return `covered=0`, `hq_rate=0.0`, `kl=float(log K)`.
"""

STARTER = '''import numpy as np

def mode_coverage(samples, centers, radius, min_frac):
    """Return {"covered", "hq_rate", "kl"}."""
    return {"covered": len(centers), "hq_rate": 1.0, "kl": 0.0}
'''

SOLUTION = '''import numpy as np

def mode_coverage(samples, centers, radius, min_frac):
    X, C = np.asarray(samples, float), np.asarray(centers, float)
    K = len(C)
    D = np.sqrt(((X[:, None] - C[None]) ** 2).sum(-1))
    nearest, dmin = D.argmin(1), D.min(1)
    hq = dmin <= radius
    n_hq = int(hq.sum())
    if n_hq == 0:
        return {"covered": 0, "hq_rate": 0.0, "kl": float(np.log(K))}
    counts = np.bincount(nearest[hq], minlength=K)
    q = counts / n_hq
    nz = q > 0
    kl = float((q[nz] * np.log(q[nz] * K)).sum())
    return {"covered": int((counts >= min_frac * n_hq).sum()), "hq_rate": n_hq / len(X), "kl": kl}
'''

HINTS = [
    'One pairwise distance matrix gives both the nearest mode and the distance to it.',
    'np.bincount with minlength=K counts samples per mode, including empty modes.',
    'log(q_k / (1/K)) equals log(q_k * K).',
]

_r = np.random.default_rng(9)
_ang = np.arange(8) * np.pi / 4
_C = np.c_[2 * np.cos(_ang), 2 * np.sin(_ang)]
_full = _C[_r.integers(0, 8, 400)] + 0.05 * _r.normal(size=(400, 2))
_coll = _C[_r.choice([0, 1, 2], 400)] + 0.05 * _r.normal(size=(400, 2))
TESTS = [
    (_full, _C, 0.3, 0.05),
    (_coll, _C, 0.3, 0.05),
    (_r.normal(size=(200, 2)) * 3, _C, 0.3, 0.05),
    (np.array([[10.0, 10.0]]), _C, 0.3, 0.05),
    (np.r_[_coll[:300], _C[[5]] + 0.01], _C, 0.3, 0.02),
    (_full, _C, 0.01, 0.05),
]

EXPLANATION = r"""
## What mode collapse looks like
The generator finds a few outputs the discriminator accepts and keeps producing them. Sample quality can look fine while diversity is gone, so you need a metric that sees the whole distribution, not single samples.

## Why a synthetic mixture
With real images you do not know the modes. A ring of Gaussians gives ground truth: count covered modes, measure the fraction of samples near any mode (quality), and the KL to uniform (balance). The three numbers separate the failure modes: low quality with full coverage, high quality with few modes, or a lopsided spread.

## At image scale
The same idea appears as precision and recall for generative models, and as FID, which is sensitive to both quality and diversity but cannot tell you which one dropped.

## Interview follow-ups
- Which training changes (minibatch discrimination, unrolled GANs, Wasserstein loss) target collapse, and how?
"""

import numpy as np

META = dict(title='CLIP-style cosine retrieval', tags=['Embeddings', 'Computer Vision'], diff='Easy', runtime='numpy',
            lessons=['ieng:26.03'], related=['softmax-rows', 'precision-recall-at-k', 'top-k-classification-accuracy'])

STATEMENT = r"""
A CLIP model embeds images and texts into the same space. Retrieval and zero-shot classification then reduce to cosine similarity: normalise every embedding to unit length, take dot products, and rank.

Write `clip_retrieve(image_emb, text_emb, k, logit_scale=100.0)` and return `(topk, probs)`.

## Input
- `image_emb`: float `(N, D)`; `text_emb`: float `(M, D)`. No row is all zeros.
- `k`: int with `1 <= k <= M`. `logit_scale`: positive float (CLIP's learned temperature, about 100).

## Output
- `topk`: int array `(N, k)`. Row `i` lists the `k` text indices with the highest cosine similarity to image `i`, best first. **Ties are broken by the smaller text index.**
- `probs`: `float64` array `(N, M)`, the softmax over texts of `logit_scale * cosine_similarity`, i.e. the zero-shot class probabilities when the texts are class prompts.

Cosine similarity is $\cos(a, b) = \dfrac{a \cdot b}{\lVert a\rVert\,\lVert b\rVert}$.
"""

STARTER = '''import numpy as np

def clip_retrieve(image_emb, text_emb, k, logit_scale=100.0):
    """Return (topk (N, k) text indices by cosine similarity, probs (N, M) softmax of scaled similarities)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def clip_retrieve(image_emb, text_emb, k, logit_scale=100.0):
    img = np.asarray(image_emb, dtype=np.float64)
    txt = np.asarray(text_emb, dtype=np.float64)
    img = img / np.linalg.norm(img, axis=1, keepdims=True)
    txt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    sim = img @ txt.T                                           # (N, M) cosine similarities
    topk = np.argsort(-sim, axis=1, kind='stable')[:, :k]       # stable: equal scores keep index order
    z = logit_scale * sim
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return topk, e / e.sum(axis=1, keepdims=True)
'''

HINTS = [
    'Once both sets of rows have unit length, the full similarity matrix is a single matrix product.',
    'For a descending ranking with ties to the smaller index, sort the **negated** scores with a stable sort: `np.argsort(-sim, axis=1, kind="stable")`.',
    'The softmax of `logit_scale * sim` needs the usual max subtraction: with scale 100, logits reach 100 and `exp` overflows float32 and gets close in float64.',
]

_r = np.random.default_rng(38)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([[3.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 2),
    (np.array([[2.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), 3, 10.0),
    (np.array([[1.0, 1.0, 1.0, 1.0]]), np.array([[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [-1.0, 0.0, 0.0, 0.0]]), 4),
    (_r.normal(size=(5, 16)), _r.normal(size=(7, 16)), 3),
    (_r.normal(size=(3, 8)) * 10, _r.normal(size=(4, 8)) * 0.01, 4, 50.0),
    (_r.normal(size=(6, 32)), _r.normal(size=(6, 32)), 1),
]

EXPLANATION = r"""
## Why normalise
CLIP's contrastive loss is computed on cosine similarities, so only an embedding's direction carries meaning; norms vary with irrelevant factors. Normalising first also makes the similarity scale fixed in $[-1, 1]$, which is why a single learned `logit_scale` can sharpen all the softmaxes at once.

## Zero-shot classification
Embed one prompt per class ("a photo of a {class}"), and `probs` is the classifier. Averaging the embeddings of several prompts per class (prompt ensembling) usually adds a few points of accuracy.

## Complexity
$O(NMD)$ for the similarities and $O(NM\log M)$ for the full sort. For large galleries use `np.argpartition` to get the top $k$ in $O(NM)$, then sort just those $k$; at web scale, an approximate nearest-neighbour index replaces the matrix product entirely.

## Common bugs
- Forgetting to normalise one side: the ranking for each image is unchanged if only images are unnormalised, but **not** if texts are (test 1 has text norms 3, 1 and $\sqrt 2$).
- `np.argsort(sim)[:, ::-1]`: descending, but ties now go to the **larger** index.
- Softmax over the wrong axis: zero-shot probabilities are over texts, per image.

## Interview follow-ups
- Implement the symmetric InfoNCE loss CLIP trains with, given a batch where image $i$ matches text $i$.
- Report recall@k for text-to-image retrieval with the same matrix. (Transpose it.)
"""

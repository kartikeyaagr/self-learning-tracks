import numpy as np

META = dict(title='CLIP zero-shot classification with prompt ensembling', tags=['Embeddings', 'Computer Vision'], diff='Medium',
            runtime='numpy', lessons=['ieng:26.03'],
            related=['clip-cosine-retrieval', 'symmetric-clip-contrastive-loss', 'siglip-pairwise-loss'])

STATEMENT = r"""
CLIP classifies an image without any task-specific training: embed one text prompt per class ("a photo of a dog"), embed the image, and pick the class whose text is most similar. Averaging the embeddings of several prompt templates per class (prompt ensembling) usually helps.

Write `clip_zero_shot(img_feats, txt_feats, W_img, W_txt, logit_scale, n_templates)` returning `(probs, preds)`.

## Inputs
- `img_feats` `(B, Di)`: image-encoder outputs; `W_img` `(Di, E)` projects them to the joint space.
- `txt_feats` `(K * T, Dt)`: text-encoder outputs for $K$ classes and $T$ = `n_templates` templates, **class-major** (rows $kT .. kT + T - 1$ belong to class $k$); `W_txt` `(Dt, E)`.
- `logit_scale`: the learned log-temperature $s$.

## Steps
1. Project and L2-normalise every image and every text embedding.
2. Class embedding: the mean of its $T$ normalised text embeddings, L2-normalised again.
3. Logits $= e^{s}\cdot$ (cosine similarity between each image and each class embedding), `(B, K)`.
4. `probs` = row-wise softmax of the logits; `preds` = argmax per row.
"""

STARTER = '''import numpy as np

def clip_zero_shot(img_feats, txt_feats, W_img, W_txt, logit_scale, n_templates):
    """Return (probs (B, K), preds (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _l2(u):
    return u / np.linalg.norm(u, axis=-1, keepdims=True)

def clip_zero_shot(img_feats, txt_feats, W_img, W_txt, logit_scale, n_templates):
    im = _l2(np.asarray(img_feats, dtype=float) @ W_img)
    tx = _l2(np.asarray(txt_feats, dtype=float) @ W_txt)
    cls = _l2(tx.reshape(-1, n_templates, tx.shape[1]).mean(1))
    lg = np.exp(logit_scale) * im @ cls.T
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p, p.argmax(1)
'''

HINTS = [
    'Normalise before averaging the templates (so each template counts equally) and again after (so the class embedding is a unit vector).',
    'Class-major order means `reshape(K, T, E)` groups the templates of each class together.',
    'The temperature is stored as a log; CLIP clips $e^{s}$ at 100 during training. A large scale makes the softmax nearly one-hot.',
]

_r = np.random.default_rng(2104)
def _c(B, K, T, Di, Dt, E, s):
    return (_r.normal(size=(B, Di)), _r.normal(size=(K * T, Dt)), _r.normal(size=(Di, E)) / np.sqrt(Di),
            _r.normal(size=(Dt, E)) / np.sqrt(Dt), s, T)
TESTS = [
    _c(2, 3, 1, 4, 5, 3, 0.0),
    _c(3, 4, 2, 6, 5, 4, np.log(100.0)),
    _c(5, 6, 4, 8, 8, 6, 2.0),
    _c(1, 2, 3, 3, 3, 2, 1.0),
    _c(4, 10, 5, 12, 10, 8, np.log(50.0)),
]

EXPLANATION = r"""
## Why it works
CLIP trains an image encoder and a text encoder on hundreds of millions of image-caption pairs with a contrastive loss: matching pairs should have high cosine similarity, mismatched pairs low. Any set of class names then defines a classifier, because the text encoder turns each name into a weight vector. The paper's zero-shot ImageNet accuracy matched a supervised ResNet-50.

## Prompt engineering and ensembling
Captions rarely consist of a bare class name, so "a photo of a {label}." works better than "{label}". Averaging embeddings over many templates ("a blurry photo of a", "a sketch of a", ...) builds a more robust class vector at no extra inference cost, since the class embeddings are computed once. The paper reports that prompt engineering and ensembling together add almost 5 points of ImageNet accuracy.

## Common bugs
- Averaging unnormalised text embeddings, which lets long or unusual prompts dominate.
- Using $s$ directly instead of $e^{s}$.
- Grouping the templates template-major when the data is class-major.

## Interview follow-ups
- How would you adapt this to a new domain with a few labelled examples (linear probe vs prompt tuning)?
- Why might zero-shot accuracy fall on classes whose names are rare or ambiguous?
"""

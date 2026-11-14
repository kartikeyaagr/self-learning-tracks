import numpy as np

META = dict(title='Masked diffusion language-model loss', tags=['Language Models', 'Diffusion Models'], diff='Medium',
            runtime='numpy',
            related=['masked-language-modeling', 'cross-entropy-logits', 'build-assistant-only-sft-loss-mask'])

STATEMENT = r"""
A masked diffusion language model (as in LLaDA) corrupts a sequence by replacing each token with `[MASK]` independently with probability $t$, for a noise level $t$ drawn uniformly from $(0, 1]$, and trains a bidirectional Transformer to predict the original tokens at the masked positions. The loss is weighted by $1/t$, which makes it an upper bound on the negative log-likelihood.

Write `masked_diffusion_loss(x0, t, u, logits, mask_id, prompt_len=None)` returning `(xt, loss)`.

## Inputs
- `x0`: `(B, L)` int clean tokens. `t`: `(B,)` noise levels in $(0, 1]$. `u`: `(B, L)` uniform draws in $[0, 1)$.
- Position $i$ of row $b$ is masked when $u_{b,i} < t_b$. If `prompt_len` `(B,)` is given (supervised fine-tuning), positions $i <$ `prompt_len[b]` are never masked.
- `logits`: `(B, L, V)`, the mask predictor's output on `xt` (supplied so you don't need a network).

## Output
- `xt`: `x0` with masked positions set to `mask_id`.
- `loss` (float): the batch mean of
  $$\ell_b = \frac{1}{t_b\,N_b}\sum_{i\ \text{masked}} -\log \mathrm{softmax}(\text{logits}_{b,i})[x_{0,b,i}]$$
  where $N_b = L$ in pre-training and $N_b = L - \text{prompt\_len}_b$ (the response length) in fine-tuning.
"""

STARTER = '''import numpy as np

def masked_diffusion_loss(x0, t, u, logits, mask_id, prompt_len=None):
    """Return (xt, loss)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def masked_diffusion_loss(x0, t, u, logits, mask_id, prompt_len=None):
    x0, t, u = np.asarray(x0), np.asarray(t, dtype=float), np.asarray(u, dtype=float)
    B, L = x0.shape
    masked = u < t[:, None]
    if prompt_len is None:
        N = np.full(B, L, dtype=float)
    else:
        pl = np.asarray(prompt_len)
        masked &= np.arange(L)[None, :] >= pl[:, None]
        N = (L - pl).astype(float)
    xt = np.where(masked, mask_id, x0)
    z = np.asarray(logits, dtype=float)
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    nll = -np.take_along_axis(logp, x0[..., None], axis=-1)[..., 0]
    per_row = (nll * masked).sum(1) / (t * N)
    return xt, float(per_row.mean())
'''

HINTS = [
    'Build the boolean mask `u < t[:, None]` first, then remove the prompt positions with a position comparison against `prompt_len[:, None]`.',
    'Gather the log-probability of the clean token at every position with `take_along_axis`, zero the unmasked ones, sum per row and divide by $t_b N_b$.',
]

_r = np.random.default_rng(2502)
_x = _r.integers(0, 6, size=(3, 5))
_lg = _r.normal(size=(3, 5, 7))
TESTS = [
    (_x, np.array([0.5, 0.5, 0.5]), _r.random((3, 5)), _lg, 6),
    (_x, np.array([1.0, 0.2, 0.05]), _r.random((3, 5)), _lg, 6),
    (_x, np.array([0.9, 0.9, 0.9]), _r.random((3, 5)), _lg, 6, np.array([2, 0, 4])),
    (_x[:1], np.array([0.01]), np.array([[0.5, 0.2, 0.9, 0.3, 0.6]]), _lg[:1], 6),
    (_r.integers(0, 9, size=(4, 8)), _r.uniform(0.05, 1, size=4), _r.random((4, 8)), _r.normal(size=(4, 8, 10)) * 3, 9, np.array([3, 3, 5, 0])),
    (_r.integers(0, 9, size=(2, 6)), np.array([1.0, 1.0]), np.zeros((2, 6)), _r.normal(size=(2, 6, 10)), 9),
]

EXPLANATION = r"""
## Why the $1/t$ weight
At noise level $t$ about $tL$ tokens are masked, so the raw sum of their losses grows with $t$. Dividing by $t$ makes every noise level contribute on the same per-token scale, and integrating the weighted loss over $t \sim U(0,1]$ gives an upper bound on $-\log p(x_0)$, which is what makes this a proper likelihood-based generative model rather than BERT with a random mask rate.

## Compared with BERT's MLM
BERT masks a fixed 15% and never generates. A masked diffusion model sees every mask rate from almost none to all, so at sampling time it can start from a fully masked sequence and unmask it over several steps.

## Fine-tuning
Only the response is ever masked; the prompt stays visible and acts as the condition. Normalising by response length keeps short and long answers on the same scale.

## Complexity
$O(BLV)$ for the log-softmax, as for any language-model loss.

## Common bugs
- Normalising by the number of masked tokens instead of $tL$ (that is a different, biased estimator).
- Letting prompt tokens be masked in fine-tuning.
- Including unmasked positions in the loss: the model sees them as input, so predicting them is trivial and distorts the objective.

## Interview follow-ups
- Why is drawing $t$ from $(0, 1]$ rather than $[0, 1]$ important for this loss?
- How does sampling from such a model work, and why can it unmask several tokens per step?
"""

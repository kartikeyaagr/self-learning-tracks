import numpy as np

META = dict(title='Masked diffusion sampling: blocks, low-confidence remasking and CFG', tags=['Language Models', 'Sampling'],
            diff='Hard', runtime='numpy',
            related=['masked-diffusion-lm-loss', 'classifier-free-guidance', 'text-generation-sampling'])

STATEMENT = r"""
A masked diffusion language model generates by starting with an all-`[MASK]` answer and unmasking it over several steps. LLaDA's best sampler is **semi-autoregressive with low-confidence remasking**: the answer is split into blocks filled left to right; at every step the model predicts all masked positions of the current block, and only the most confident predictions are kept.

Write `diffusion_generate(prompt, gen_len, block, steps, p, mask_id, cfg=0.0)` returning `(tokens, order)`.

## The mask predictor (given, bidirectional)
For a sequence $s$ of length $N$: $h_i = E[s_i] + P_i$, $c = \frac1N\sum_i h_i$, logits$_i = (h_i + c)\,W$. `p` has `E` `(V+1, D)` (row `mask_id` = $V$ is the mask embedding), `P` `(N_{max}, D)`, `W` `(D, V)`.

## Classifier-free guidance
If `cfg` $= w > 0$, also run the predictor on the sequence with **all prompt tokens replaced by `mask_id`** and use $(1 + w)\,\ell_{cond} - w\,\ell_{uncond}$ as the logits.

## Schedule
- `gen_len` is a multiple of `block`; there are $n_b$ = `gen_len // block` blocks and `steps` is a multiple of $n_b$, giving $k = $ `steps // n_b` steps per block. Each step unmasks `block // k` positions (divides exactly).
- At each step: compute logits for the whole current sequence; for each still-masked position **in the current block** take the argmax token and its softmax probability (confidence); unmask the `block // k` positions with the highest confidence (ties: lower position first), writing their argmax tokens.

## Output
- `tokens`: the prompt followed by the `gen_len` generated ids.
- `order`: int array `(gen_len,)`, the 0-based global step at which each generated position was unmasked.
"""

STARTER = '''import numpy as np

def diffusion_generate(prompt, gen_len, block, steps, p, mask_id, cfg=0.0):
    """Return (tokens, order)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _logits(seq, p):
    h = p['E'][seq] + p['P'][:len(seq)]
    return (h + h.mean(0)) @ p['W']

def diffusion_generate(prompt, gen_len, block, steps, p, mask_id, cfg=0.0):
    prompt = np.asarray(prompt)
    P0 = len(prompt)
    seq = np.concatenate([prompt, np.full(gen_len, mask_id)])
    order = np.full(gen_len, -1)
    per_block = steps // (gen_len // block)
    per_step = block // per_block
    step = 0
    for b0 in range(P0, P0 + gen_len, block):
        for _ in range(per_block):
            lg = _logits(seq, p)
            if cfg > 0:
                un = seq.copy()
                un[:P0] = mask_id
                lg = (1 + cfg) * lg - cfg * _logits(un, p)
            z = lg - lg.max(1, keepdims=True)
            prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            pos = [i for i in range(b0, b0 + block) if seq[i] == mask_id]
            conf = np.array([prob[i].max() for i in pos])
            keep = np.argsort(-conf, kind='stable')[:per_step]      # stable: lower position wins ties
            for j in keep:
                i = pos[j]
                seq[i] = int(np.argmax(lg[i]))
                order[i - P0] = step
            step += 1
    return seq, order
'''

HINTS = [
    'Keep one array `seq` with the prompt and `gen_len` mask ids, and loop over blocks, then over the steps of each block. Every step re-runs the predictor on the whole current sequence.',
    'Only masked positions inside the current block are candidates. Rank them by the maximum softmax probability and unmask the top `block // k` with their argmax tokens.',
    'For guidance, build the unconditional input by copying `seq` and setting its prompt positions to `mask_id`; combine logits as $(1+w)\\ell_c - w\\ell_u$ before the softmax.',
]

_r = np.random.default_rng(2511)
def _p(V, D, N):
    return dict(E=_r.normal(size=(V + 1, D)), P=_r.normal(size=(N, D)) * 0.5, W=_r.normal(size=(D, V)))
_p1, _p2 = _p(6, 4, 16), _p(10, 6, 24)
TESTS = [
    (np.array([1, 2]), 4, 4, 4, _p1, 6),
    (np.array([1, 2]), 4, 4, 2, _p1, 6),
    (np.array([1, 2]), 4, 2, 4, _p1, 6),
    (np.array([3, 0, 5]), 8, 4, 4, _p1, 6, 1.0),
    (np.array([7, 1, 4, 4]), 12, 4, 6, _p2, 10),
    (np.array([7, 1, 4, 4]), 12, 12, 4, _p2, 10, 0.5),
    (np.array([9]), 6, 3, 6, _p2, 10, 2.0),
]

EXPLANATION = r"""
## Why low-confidence remasking
Unmasking a fixed random subset ignores what the model knows. Committing first to the predictions the model is surest about, and leaving uncertain positions masked so they can be predicted again with more context, gives clearly better samples. (The "remasking" view: predict every masked token, then re-mask the least confident ones.)

## Why blocks
Pure low-confidence sampling over a long answer tends to commit end-of-text tokens early, because padding at the end is easy to predict. Filling blocks left to right adds an autoregressive bias at the block level while keeping parallel, bidirectional prediction inside each block.

## Steps vs quality
Fewer steps means more tokens committed per forward pass: faster, but each commitment is made with less context. `steps = gen_len` (one token per step) is the slowest and usually the best.

## Guidance
Classifier-free guidance here contrasts the prediction with and without the prompt, pushing the answer toward tokens the prompt makes more likely. It costs a second forward pass per step.

## Common bugs
- Choosing among masked positions in the whole answer instead of the current block.
- Recomputing logits once per block instead of once per step.
- Ranking positions by raw logits instead of softmax probabilities; logits are not comparable across positions.

## Interview follow-ups
- How could a KV cache be used with a bidirectional model whose earlier blocks are frozen?
- Compare the number of forward passes with autoregressive decoding of the same answer.
"""

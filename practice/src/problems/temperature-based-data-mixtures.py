import numpy as np

META = dict(title='Temperature-sampled data mixture with epoch caps', tags=['Data Processing', 'Language Models'],
            diff='Medium', runtime='numpy', lessons=['air-fanout-v1:10.03'],
            related=['extrapolate-data-mixture-scaling-curves', 'perform-stable-exact-deduplication', 'row-scaling'])

STATEMENT = r"""
Sampling training data in proportion to each source's size lets the biggest source (say, English web text) drown out small ones (low-resource languages, code in rare languages). **Temperature sampling** flattens the distribution:

$$w_i \propto n_i^{1/\tau}$$

where $n_i$ is the number of tokens in source $i$. $\tau = 1$ is proportional sampling; as $\tau \to \infty$ it approaches uniform. Upweighting a small source means repeating it, though, and too many epochs over the same text leads to memorisation, so each source is capped at `max_epochs` passes.

Write `temperature_mixture(tokens, tau, total_tokens, max_epochs)` and return a dict.

## Inputs
- `tokens`: float64 `(K,)`, positive source sizes.
- `tau`: float $> 0$. `total_tokens`: float, the training budget. `max_epochs`: float $> 0$.
- The tests guarantee $\text{total\_tokens} \le \text{max\_epochs} \cdot \sum_i n_i$, so a valid mixture exists.

## Capping (water-filling)
Source $i$'s weight may not exceed its cap $c_i = \text{max\_epochs} \cdot n_i / \text{total\_tokens}$. Repeat until nothing changes:
1. Sources already capped keep $w_i = c_i$.
2. The remaining mass $1 - \sum_{\text{capped}} c_i$ is shared among the uncapped sources in proportion to $n_i^{1/\tau}$.
3. Every uncapped source whose new weight **exceeds** its cap becomes capped.

## Output
| key | value |
|---|---|
| `"weights"` | float64 `(K,)`, summing to 1 |
| `"epochs"` | float64 `(K,)`, $w_i \cdot \text{total\_tokens} / n_i$ |
| `"capped"` | bool `(K,)` |

Compute $n_i^{1/\tau}$ as `tokens ** (1 / tau)`; for large sizes and small $\tau$ you may rescale `tokens` by its maximum first (the weights are scale-free).
"""

STARTER = '''import numpy as np

def temperature_mixture(tokens, tau, total_tokens, max_epochs):
    """Return a dict: weights (K,), epochs (K,), capped (K,) bool."""
    n = np.asarray(tokens, dtype=float)
    w = n ** (1 / tau)
    w = w / w.sum()                                  # ignores the epoch caps
    return {"weights": w, "epochs": w * total_tokens / n, "capped": np.zeros(len(n), dtype=bool)}
'''

SOLUTION = '''import numpy as np

def temperature_mixture(tokens, tau, total_tokens, max_epochs):
    n = np.asarray(tokens, dtype=float)
    base = (n / n.max()) ** (1.0 / tau)              # rescale first so large n and small tau can't overflow
    cap = max_epochs * n / total_tokens
    capped = np.zeros(len(n), dtype=bool)
    while True:
        w = np.where(capped, cap, 0.0)
        free = ~capped
        if not free.any():
            break
        w[free] = (1.0 - cap[capped].sum()) * base[free] / base[free].sum()
        over = free & (w > cap)
        if not over.any():
            break
        capped |= over                               # pin them and redistribute what's left
    return {"weights": w, "epochs": w * total_tokens / n, "capped": capped}
'''

HINTS = [
    'Without caps this is one line: normalise `tokens ** (1 / tau)`. Start from that and then enforce the caps.',
    'Keep a boolean `capped` array. Each round, fix the capped weights at their caps and give the leftover mass to the rest, proportional to their tempered sizes.',
    'Stop when no uncapped source exceeds its cap. Capping one source only *raises* the others, so sources never need to be uncapped, and the loop ends after at most $K$ rounds.',
]

TESTS = [
    (np.array([100.0, 1.0]), 1.0, 50.0, 10.0),
    (np.array([100.0, 1.0]), 2.0, 50.0, 4.0),
    (np.array([900.0, 90.0, 9.0, 1.0]), 1e6, 1000.0, 100.0),
    (np.array([5e11, 8e10, 2e10, 3e9, 4e8]), 3.3, 1e12, 4.0),
    (np.array([5e11, 8e10, 2e10, 3e9, 4e8]), 5.0, 2e12, 4.0),
    (np.array([1.0, 1.0, 1.0]), 0.7, 3.0, 1.5),
    (np.array([1e12, 1e9, 1e6]), 0.5, 1e12, 2.0),
    (np.array([40.0, 30.0, 20.0, 10.0]), 1.5, 120.0, 1.5),
]

EXPLANATION = r"""
## What the temperature does
Raising sizes to the power $1/\tau < 1$ compresses ratios: with $\tau = 2$, a source 100 times larger gets only 10 times the weight. Multilingual models typically use $\tau$ between 3 and 5. The caps are what keep that from becoming 50 epochs of a tiny language: test 4 caps every source except the largest, test 5 doubles the budget (which halves every cap, so the largest source absorbs the difference), and test 3 is close to uniform ($\tau = 10^6$) until its smallest source hits its cap.

## Why water-filling terminates and is correct
Capping a source removes it from the pool and frees the mass it would have taken above its cap, so every uncapped source's weight can only **increase** from round to round. A capped source therefore never needs to be released, each round caps at least one new source or stops, and the loop runs at most $K$ times. The result is the mixture closest to temperature sampling (in the sense of keeping uncapped ratios exact) that respects every cap.

## Complexity
$O(K^2)$ in the worst case, trivial for the tens of sources in a real mixture.

## Common bugs
- Clipping weights to their caps once and renormalising: renormalising pushes capped sources back above their caps.
- Redistributing freed mass in proportion to raw sizes $n_i$ instead of $n_i^{1/\tau}$.
- Overflow: $(10^{12})^{1/0.5} = 10^{24}$ is fine, but sizes of $10^{12}$ with $\tau = 0.01$ are not. Rescaling by the maximum first avoids it (test 7 uses $\tau < 1$, which *sharpens* the distribution).

## Interview follow-ups
- How would you pick $\tau$ for a multilingual model, and what do you measure to see whether low-resource languages are now over-fit?
- Some pipelines schedule the mixture over training (more high-quality data at the end). How does that interact with epoch caps?
"""

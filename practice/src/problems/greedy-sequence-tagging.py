import numpy as np

META = dict(title='Greedy decoding of a tagging model', tags=['NLP'], diff='Easy', runtime='numpy',
            related=['named-entity-recognition', 'pos-tagging', 'beam-search-decoding'])

STATEMENT = r"""
A linear-chain tagger scores a tag sequence $y_0, \dots, y_{T-1}$ as
$$\text{score}(y) = s_{y_0} + \sum_{t=0}^{T-1} e_{t, y_t} + \sum_{t=1}^{T-1} A_{y_{t-1}, y_t}$$
with start scores $s$, emission scores $e$ (how well tag $k$ fits token $t$) and transition scores $A$ (how well tag $j$ follows tag $i$). The cheapest decoder picks tags left to right, never revisiting a choice.

Write `greedy_decode(emissions, transitions, start)` that returns `(tags, score)`.

## Inputs
- `emissions`: float array `(T, K)` with $T \ge 1$.
- `transitions`: float array `(K, K)`, entry `[i, j]` is the score of tag `j` following tag `i`.
- `start`: float array `(K,)`.

## Algorithm
- $y_0 = \arg\max_k (s_k + e_{0,k})$.
- For $t \ge 1$: $y_t = \arg\max_k (A_{y_{t-1}, k} + e_{t,k})$.
- Ties go to the smallest tag index.

## Output
`tags` as a list of Python ints, and `score` as a float: $\text{score}(y)$ of the greedy path.
"""

STARTER = '''import numpy as np

def greedy_decode(emissions, transitions, start):
    """Return (list of tag ints, path score float)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def greedy_decode(emissions, transitions, start):
    E, A, s = (np.asarray(x, dtype=float) for x in (emissions, transitions, start))
    local = s + E[0]
    y = int(np.argmax(local))              # argmax returns the first maximum: smallest index wins ties
    tags, score = [y], local[y]
    for t in range(1, len(E)):
        local = A[y] + E[t]                # only the row of the previous choice matters
        y = int(np.argmax(local))
        tags.append(y)
        score += local[y]
    return tags, float(score)
'''

HINTS = [
    'The first tag only sees the start scores plus the first emission row. Every later tag sees the transition row of the tag you just chose plus its own emission row.',
    '`np.argmax` returns the first index among equal maxima, which is exactly the tie rule.',
    'Accumulate the score as you go: the value you maximised at each step is exactly that step\'s contribution to $\\text{score}(y)$.',
]

_r = np.random.default_rng(19)
TESTS = [
    (np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, -2.0], [0.0, 0.0]]), np.array([0.0, 0.0])),
    (np.array([[2.0, 2.0, 0.0], [1.0, 0.0, 3.0]]), np.zeros((3, 3)), np.array([0.0, 0.0, 1.0])),
    (np.array([[0.5, -0.5]]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([0.0, 2.0])),
    (np.zeros((4, 3)), np.zeros((3, 3)), np.zeros(3)),
    (np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 2.0]]), np.array([[0.0, -5.0], [-5.0, 0.0]]), np.zeros(2)),
    (_r.normal(size=(8, 5)), _r.normal(size=(5, 5)), _r.normal(size=5)),
    (_r.normal(size=(15, 9)) * 3, _r.normal(size=(9, 9)), _r.normal(size=9)),
]

EXPLANATION = r"""
## Why it works (and when it doesn't)
Each step maximises the next term of the score given the past choices, so the greedy path is a valid sequence with a correctly computed score. It is not always the **best** sequence. In test 5 the first token strongly prefers tag 0, but switching tags costs 5, so greedy stays on tag 0 and collects only 3. The path `1, 1, 1` would score 4: an early locally good choice blocked a better global one. Viterbi fixes this with dynamic programming.

## Complexity
$O(TK)$: one row of $K$ scores per step. Viterbi costs $O(TK^2)$ because it keeps the best score for every tag at every step.

## Common bugs
- Using the column `A[:, y]` instead of the row `A[y]` (the convention here is `[previous, next]`).
- Adding the start scores at every step.
- Breaking ties by sorting with an unstable sort.

## Interview follow-ups
- Implement Viterbi for the same score and show it is optimal. (Keep `best[t, k]` and a back-pointer, then trace back from the best final tag.)
- Where does beam search sit between greedy and Viterbi, and why is Viterbi not an option for Transformer language models? (The score of the next token depends on the entire prefix, not just the previous tag.)
"""

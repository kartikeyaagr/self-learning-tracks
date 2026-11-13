import numpy as np

META = dict(title='DETR-style Hungarian matching', tags=['Object Detection', 'Optimization'], diff='Hard', runtime='numpy',
            lessons=[], related=['generalized-iou-loss', 'iou-matrix', 'box-encode-and-decode'])

STATEMENT = r"""
Set-based detectors such as DETR predict a fixed number of boxes `Q` and must decide which prediction is responsible for which of the `T <= Q` ground-truth objects before computing the loss. They pick the one-to-one assignment with the lowest total cost, found with the Hungarian algorithm. Unmatched predictions are trained to say "no object".

Write `hungarian_match(probs, pred_boxes, tgt_labels, tgt_boxes, w_class=1.0, w_box=5.0)` and return `(query_idx, target_idx)`.

## Cost matrix
For prediction `q` and target `t`:

$$C[q, t] = -w_{class}\; \text{probs}[q, \text{label}_t] \;+\; w_{box} \sum_{k=0}^{3} \big|\text{pred\_boxes}[q, k] - \text{tgt\_boxes}[t, k]\big|$$

## Assignment
Choose distinct predictions $q_0, \dots, q_{T-1}$, one per target, minimising $\sum_t C[q_t, t]$. Each target gets exactly one prediction; `Q - T` predictions stay unmatched. Test cases are built so the optimum is unique.

Implement the assignment yourself (no SciPy). An exhaustive search over permutations is too slow for the larger tests; use the Hungarian method (potentials and shortest augmenting paths), which runs in $O(T^2 Q)$.

## Input / output
- `probs`: float `(Q, K)` class probabilities. `pred_boxes`: float `(Q, 4)`. `tgt_labels`: int `(T,)`. `tgt_boxes`: float `(T, 4)`. `0 <= T <= Q`.
- `query_idx`, `target_idx`: Python lists of `T` ints, the matched pairs **sorted by query index** (so `query_idx` is increasing). Empty lists when `T = 0`.
"""

STARTER = '''import numpy as np

def hungarian_match(probs, pred_boxes, tgt_labels, tgt_boxes, w_class=1.0, w_box=5.0):
    """Return (query_idx, target_idx): the min-cost one-to-one matching, sorted by query index."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _assign(a):
    """Min-cost assignment of every row of a (n x m, n <= m) to a distinct column.
    Shortest-augmenting-path Hungarian method with row/column potentials u, v. Returns col of each row."""
    n, m = a.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)          # p[j] = row (1-based) matched to column j; column 0 is a sentinel
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:                          # grow a shortest-path tree until it reaches a free column
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = a[i0 - 1] - u[i0] - v[1:]  # reduced costs from row i0
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta              # keep reduced costs non-negative along the tree
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:                            # flip the augmenting path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            col[p[j] - 1] = j - 1
    return col

def hungarian_match(probs, pred_boxes, tgt_labels, tgt_boxes, w_class=1.0, w_box=5.0):
    probs, pb, tb = (np.asarray(x, dtype=np.float64) for x in (probs, pred_boxes, tgt_boxes))
    labels = np.asarray(tgt_labels, dtype=int)
    if labels.size == 0:
        return [], []
    cost = -w_class * probs[:, labels] + w_box * np.abs(pb[:, None, :] - tb[None, :, :]).sum(-1)   # (Q, T)
    q_of_t = _assign(cost.T)                 # rows = targets, columns = queries
    order = np.argsort(q_of_t)
    return q_of_t[order].tolist(), order.tolist()
'''

HINTS = [
    'Build the `(Q, T)` cost matrix with broadcasting first. Then the problem is a pure rectangular assignment: give each target (row of `C.T`) a distinct query (column).',
    'The Hungarian method keeps potentials `u` (rows) and `v` (columns) so that `C[i, j] - u[i] - v[j] >= 0`, and adds rows one at a time by finding a shortest augmenting path (Dijkstra on reduced costs) to an unassigned column.',
    'For each new row: keep `minv[j]` (best reduced cost to reach column j) and `way[j]` (previous column). Repeatedly mark the closest unused column, shift potentials by that `delta`, and stop at a free column; then walk `way` back to flip the path. Finally sort the pairs by query index.',
]

_r = np.random.default_rng(34)
def _case(Q, T, K):
    probs = _r.dirichlet(np.ones(K), size=Q)
    return probs, _r.uniform(0, 1, (Q, 4)), _r.integers(0, K, size=T), _r.uniform(0, 1, (T, 4))
TESTS = [
    (np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]),
     np.array([1, 0]), np.array([[0.6, 0.6, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]])),
    (np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), np.array([[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]),
     np.array([0]), np.array([[0.8, 0.875, 0.125, 0.125]])),
    (*_case(4, 0, 3),),
    (*_case(5, 5, 4),),
    (*_case(8, 3, 5), 2.0, 1.0),
    (*_case(12, 7, 6),),
    (*_case(20, 20, 10), 1.0, 2.0),
]

EXPLANATION = r"""
## Why a one-to-one match
DETR has no anchors and no NMS. Instead, each ground-truth object is assigned to exactly one query, and every other query is pushed toward "no object". The bipartite matching makes the loss invariant to the order of the predictions and stops several queries from all regressing onto the same object, which is what removes the need for NMS.

## How the Hungarian method works
Keep dual potentials $u_i$ and $v_j$ with reduced costs $C_{ij} - u_i - v_j \ge 0$, and maintain an optimal matching for the rows added so far. To add row $i$, run Dijkstra over columns on the reduced costs: `minv[j]` is the cheapest way found to reach column $j$, `way[j]` remembers through which column. Each step settles the closest column and shifts potentials by `delta`, which keeps all reduced costs non-negative and makes the edges of the tree tight. When the newest column is unmatched, flipping the path along `way` grows the matching by one. Every row costs $O(TQ)$ (at most $T$ settle steps, each $O(Q)$), so the total is $O(T^2 Q)$; the inner step is vectorised over columns here.

## Common bugs
- Brute-forcing permutations: $20!$ is not an option (test 7).
- Greedy matching (repeatedly take the cheapest remaining pair) is not optimal: a cheap early pick can force an expensive later one.
- Forgetting that `T < Q` is the normal case and solving it as if the matrix were square.
- Returning pairs in target order when the caller expects query order.

## Interview follow-ups
- DETR's real cost also includes $-\mathrm{GIoU}$ between boxes. Add it. Why is the class term the negative **probability** rather than log-probability?
- Why does DETR train slowly, and how do later models (Deformable DETR, DINO) speed up the matching's convergence? (Denoising queries and better query initialisation make the assignment stable earlier.)
"""

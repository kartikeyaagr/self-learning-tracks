import numpy as np

META = dict(title='Capstone: train a graph classifier on a citation-style graph', tags=['Neural Networks', 'Classification'], diff='Hard', runtime='numpy',
            related=['normalized-adjacency', 'gcn-layer', 'masked-split-accuracy', 'oversmoothing-measure'])

STATEMENT = r"""
Train a simplified graph convolutional network (SGC): propagate features $K$ times, then fit a softmax classifier on the labelled nodes.

Write `train_sgc(edges, n, X, labels, train_mask, val_mask, test_mask, K, lr, steps, wd)`:

1. $\hat A = D^{-1/2}(A + I)D^{-1/2}$ for the undirected graph (duplicates and self-edges ignored), and $F = \hat A^K X$.
2. Logits $Z = FW + b$ with $W$ and $b$ starting at zero; $C$ = `labels.max() + 1` classes.
3. Loss: mean cross-entropy over **training nodes only** plus $\tfrac{wd}{2}\lVert W\rVert^2$ (no penalty on $b$).
4. Run `steps` steps of full-batch gradient descent with learning rate `lr`.
5. Report the final loss and the accuracy (argmax, lowest class on ties) on each of the three masks.

Return a dict with `loss`, `train_acc`, `val_acc`, `test_acc`.
"""

STARTER = '''import numpy as np

def train_sgc(edges, n, X, labels, train_mask, val_mask, test_mask, K, lr, steps, wd):
    """Propagate features K times with the normalised adjacency, then train softmax regression on labelled nodes."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ahat(edges, n):
    A = np.zeros((n, n))
    for i, j in edges:
        if i != j:
            A[i, j] = A[j, i] = 1.0
    A += np.eye(n)
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None, :]

def _acc(Z, y, m):
    m = np.asarray(m, bool)
    return float((Z.argmax(1) == y)[m].mean()) if m.any() else 0.0

def train_sgc(edges, n, X, labels, train_mask, val_mask, test_mask, K, lr, steps, wd):
    X, y = np.asarray(X, float), np.asarray(labels, int)
    tr = np.asarray(train_mask, bool)
    F = np.linalg.matrix_power(_ahat(edges, n), K) @ X
    C = int(y.max()) + 1
    W, b = np.zeros((X.shape[1], C)), np.zeros(C)
    Y = np.eye(C)[y[tr]]
    Ft = F[tr]
    m = len(Ft)
    def loss_grad(W, b):
        Z = Ft @ W + b
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        loss = float(-np.log((P * Y).sum(1)).mean() + 0.5 * wd * (W ** 2).sum())
        G = (P - Y) / m
        return loss, Ft.T @ G + wd * W, G.sum(0)
    for _ in range(steps):
        _, gW, gb = loss_grad(W, b)
        W, b = W - lr * gW, b - lr * gb
    loss = loss_grad(W, b)[0]
    Z = F @ W + b
    return {"loss": loss, "train_acc": _acc(Z, y, tr), "val_acc": _acc(Z, y, val_mask), "test_acc": _acc(Z, y, test_mask)}
'''

HINTS = [
    'Propagation has no parameters in SGC, so compute A_hat^K X once before training.',
    'Only training rows enter the loss and its gradient; validation and test nodes still shaped F through the graph.',
    'Compare K = 0 (plain logistic regression on the features) with K = 2 on the same split to see what the graph adds.',
]

def _graph(seed, n_per, C, p_in, p_out, f, noise, n_train_per):
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(C), n_per)
    n = len(y)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if g.random() < (p_in if y[i] == y[j] else p_out)]
    mu = 1.0 * g.normal(size=(C, f))
    X = mu[y] + noise * g.normal(size=(n, f))
    order = g.permutation(n)
    tr = np.zeros(n, bool)
    for c in range(C):
        tr[[i for i in order if y[i] == c][:n_train_per]] = True
    rest = [i for i in order if not tr[i]]
    va = np.zeros(n, bool); va[rest[:len(rest) // 2]] = True
    te = np.zeros(n, bool); te[rest[len(rest) // 2:]] = True
    return edges, n, X, y, tr, va, te

_G1 = _graph(1, 20, 3, 0.25, 0.02, 8, 2.5, 3)
_G2 = _graph(2, 15, 4, 0.3, 0.05, 6, 3.0, 2)

TESTS = [
    _G1 + (0, 0.5, 200, 1e-3),
    _G1 + (2, 0.5, 200, 1e-3),
    _G1 + (64, 0.5, 200, 1e-3),
    _G2 + (0, 0.5, 300, 1e-2),
    _G2 + (2, 0.5, 300, 1e-2),
    ([(0, 1), (2, 3)], 4, [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.3, 0.7]], [0, 0, 1, 1], [True, False, True, False], [False, True, False, False], [False, False, False, True], 1, 1.0, 50, 0.0),
]

EXPLANATION = r"""
## The whole pipeline
Building the normalised adjacency, propagating features, scoring on masked splits and checking for over-smoothing all feed this capstone. SGC (Wu et al., 2019) showed that on citation graphs most of a GCN's accuracy comes from propagation alone: removing the nonlinearities between layers and training one linear classifier on $\hat A^K X$ loses little and trains much faster.

## What the tests show
- **The graph helps.** With only 3 labelled nodes per class and noisy features, logistic regression on the raw features ($K = 0$) has poor validation and test accuracy. Two rounds of propagation average each node with its same-class neighbours, and accuracy jumps.
- **Too much propagation.** At $K = 64$ every node's features have converged to nearly the same direction (the over-smoothing step). Validation accuracy falls to about chance, and even the training loss stays near $\log 3$ because the classes can no longer be separated.
- **Training accuracy is not the target.** With $K = 0$ it is 1.0 while test accuracy is about 0.54 in the first test: with so few labels, the classifier memorises the training nodes. Only the validation and test numbers say whether the graph helped.

## Interview follow-ups
- How would you pick $K$ and the weight decay without looking at the test split, and how many random splits would you average over before reporting a result?
"""

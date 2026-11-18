TITLE = 'Find the bugs in a training script'
MODULES = ['08', '09']

STATEMENT = r'''
The starter code trains an L2-regularised logistic regression with early stopping. It runs without errors and prints plausible numbers. It contains **8 planted bugs**. Find and fix all of them in the editor. Keep the function names and signatures.

Intended behaviour:
- `make_split` shuffles rows (keeping X and y aligned) and holds out `val_frac` of them.
- `prepare` standardises features using training statistics only.
- `loss_and_grad` returns the mean log-loss plus $\frac{\lambda}{2}\lVert w\rVert^2$ (the bias is not regularised), and its exact gradient.
- `train` runs gradient descent, keeps the weights with the lowest validation loss, and stops after `patience` epochs in a row without improvement.
- `predict` returns 0/1 labels; `accuracy` returns the fraction correct.

The check tests each function separately and lists every probe that still fails.
'''

STARTER = '''def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def make_split(X, y, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))]
    y = y[rng.permutation(len(y))]
    n_val = int(len(X) * val_frac)
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]

def prepare(X_train, X_val):
    X_all = np.vstack([X_train, X_val])
    mu, sd = X_all.mean(axis=0), X_all.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_val - mu) / sd

def loss_and_grad(w, b, X, y, lam):
    z = X @ w + b
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * np.sum(w ** 2)
    r = sigmoid(z) - y
    dw = X.T @ r + lam * w
    db = np.mean(r) + lam * b
    return loss, dw, db

def train(X_tr, y_tr, X_va, y_va, lr=0.1, lam=1e-2, epochs=500, patience=20):
    w = np.zeros(X_tr.shape[1])
    b = 0.0
    best_val, best_w, best_b = np.inf, w, b
    bad, hist = 0, []
    for ep in range(epochs):
        _, dw, db = loss_and_grad(w, b, X_tr, y_tr, lam)
        w -= lr * dw
        b -= lr * db
        val, _, _ = loss_and_grad(w, b, X_va, y_va, 0.0)
        hist.append(val)
        if val < best_val:
            best_val, best_w, best_b = val, w, b
        else:
            bad += 1
            if bad >= patience:
                break
    return best_w, best_b, hist

def predict(w, b, X):
    return (X @ w + b > 0.5).astype(int)

def accuracy(w, b, X, y):
    pred = predict(w, b, X).reshape(-1, 1)
    return float(np.mean(pred == y))

# demo run
_rng = np.random.default_rng(0)
_X = _rng.normal(size=(1000, 5)) * [1, 10, 0.1, 1, 3] + [0, 50, 0, 0, 0]
_y = (_X @ np.array([1.0, 0.1, 5.0, -1.0, 0.0]) - 5 + _rng.normal(size=1000) > 0).astype(float)
Xtr, ytr, Xva, yva = make_split(_X, _y)
Xtr, Xva = prepare(Xtr, Xva)
w_hat, b_hat, hist = train(Xtr, ytr, Xva, yva)
print(f"epochs run: {len(hist)}, best val loss: {min(hist):.4f}, val accuracy: {accuracy(w_hat, b_hat, Xva, yva):.3f}")
'''

CHECK = r'''def _check():
    fails = []
    def probe(name, fn):
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"raised {type(e).__name__}: {e}"
        if not ok:
            fails.append(f"{name}: {msg}")
    rng = np.random.default_rng(6)

    def p_split():
        X = np.arange(40.0)[:, None] * np.ones((1, 3)); y = (np.arange(40) % 3 == 0).astype(float)
        a, ya, c, yc = make_split(X, y, 0.25, seed=1)
        ok = np.array_equal(ya, (a[:, 0] % 3 == 0)) and np.array_equal(yc, (c[:, 0] % 3 == 0)) and len(c) == 10
        return ok, "X and y rows no longer line up after the split"
    probe("make_split", p_split)

    def p_prep():
        A = rng.normal(3, 2, size=(50, 4)); B = rng.normal(-5, 7, size=(20, 4))
        a, b_ = prepare(A, B)
        mu, sd = A.mean(0), A.std(0)
        return np.allclose(a, (A - mu) / sd) and np.allclose(b_, (B - mu) / sd), "scaling statistics are not from the training set only"
    probe("prepare", p_prep)

    X = rng.normal(size=(30, 4)); y = (rng.random(30) < 0.4).astype(float)
    w0 = rng.normal(size=4); b0 = 0.7; lam = 0.5
    def ref_loss(w, b):
        z = X @ w + b
        return np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * np.sum(w ** 2)
    def p_loss():
        L, _, _ = loss_and_grad(w0, b0, X, y, lam)
        return abs(float(L) - ref_loss(w0, b0)) < 1e-10, "loss value is wrong"
    probe("loss_and_grad loss", p_loss)
    def p_dw():
        _, dw, _ = loss_and_grad(w0, b0, X, y, lam)
        num = np.array([(ref_loss(w0 + 1e-6 * e, b0) - ref_loss(w0 - 1e-6 * e, b0)) / 2e-6 for e in np.eye(4)])
        return np.shape(dw) == (4,) and np.allclose(dw, num, rtol=1e-5, atol=1e-7), "dw does not match finite differences of the loss"
    probe("loss_and_grad dw", p_dw)
    def p_db():
        _, _, db = loss_and_grad(w0, b0, X, y, lam)
        num = (ref_loss(w0, b0 + 1e-6) - ref_loss(w0, b0 - 1e-6)) / 2e-6
        return abs(float(db) - num) < 1e-6, "db does not match finite differences of the loss"
    probe("loss_and_grad db", p_db)

    def scripted(vals):
        it = iter(vals)
        def fake(w, b, Xa, ya, lam_):
            if Xa is Xv:
                return next(it), None, None
            return 0.0, np.ones_like(w), 1.0
        return fake
    Xt, Xv = np.zeros((5, 3)), np.zeros((4, 3))
    vals = [5, 4, 6, 6, 3, 6, 6, 6, 6, 6, 6, 6]
    real = globals()["loss_and_grad"]
    def p_best():
        globals()["loss_and_grad"] = scripted(vals)
        try:
            bw, bb, h = train(Xt, np.zeros(5), Xv, np.zeros(4), lr=0.1, lam=0.0, epochs=12, patience=3)
        finally:
            globals()["loss_and_grad"] = real
        return np.allclose(bw, -0.5) and abs(bb + 0.5) < 1e-12, "returned weights are not the ones from the best validation epoch"
    probe("train best weights", p_best)
    def p_pat():
        globals()["loss_and_grad"] = scripted(vals)
        try:
            _, _, h = train(Xt, np.zeros(5), Xv, np.zeros(4), lr=0.1, lam=0.0, epochs=12, patience=3)
        finally:
            globals()["loss_and_grad"] = real
        return len(h) == 8, f"early stopping fired after {len(h)} epochs; expected 8 for this validation curve"
    probe("train early stopping", p_pat)

    def p_pred():
        Xp = np.array([[0.2], [0.4], [-0.1], [3.0]])
        p = np.asarray(predict(np.array([1.0]), 0.0, Xp))
        return np.array_equal(p.ravel(), [1, 1, 0, 1]) and p.shape == (4,), f"predict gives {p.ravel()} for logits [0.2, 0.4, -0.1, 3.0]"
    probe("predict", p_pred)
    def p_acc():
        Xp = np.array([[1.0], [-1.0], [2.0], [-3.0], [0.5]]); yp = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
        a = accuracy(np.array([1.0]), 0.0, Xp, yp)
        return abs(a - 0.8) < 1e-12, f"accuracy {a}, expected 0.8"
    probe("accuracy", p_acc)

    expect(not fails, f"{len(fails)} of 9 probes fail:\n  - " + "\n  - ".join(fails))
'''

SOLUTION = '''def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))

def make_split(X, y, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))                    # fix 1: one permutation for both X and y
    X, y = X[perm], y[perm]
    n_val = int(len(X) * val_frac)
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]

def prepare(X_train, X_val):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)   # fix 2: training statistics only (no leakage)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_val - mu) / sd

def loss_and_grad(w, b, X, y, lam):
    z = X @ w + b
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * np.sum(w ** 2)
    r = sigmoid(z) - y
    dw = X.T @ r / len(y) + lam * w                   # fix 3: gradient of a mean divides by n
    db = np.mean(r)                                   # fix 4: the bias is not regularised
    return loss, dw, db

def train(X_tr, y_tr, X_va, y_va, lr=0.1, lam=1e-2, epochs=500, patience=20):
    w = np.zeros(X_tr.shape[1])
    b = 0.0
    best_val, best_w, best_b = np.inf, w.copy(), b
    bad, hist = 0, []
    for ep in range(epochs):
        _, dw, db = loss_and_grad(w, b, X_tr, y_tr, lam)
        w -= lr * dw
        b -= lr * db
        val, _, _ = loss_and_grad(w, b, X_va, y_va, 0.0)
        hist.append(val)
        if val < best_val:
            best_val, best_w, best_b = val, w.copy(), b   # fix 5: copy, since `w -=` mutates the array in place
            bad = 0                                       # fix 6: patience counts consecutive bad epochs
        else:
            bad += 1
            if bad >= patience:
                break
    return best_w, best_b, hist

def predict(w, b, X):
    return (X @ w + b > 0).astype(int)                # fix 7: logit > 0 is probability > 0.5

def accuracy(w, b, X, y):
    pred = predict(w, b, X)                           # fix 8: no reshape, which broadcast (n, 1) == (n,) to (n, n)
    return float(np.mean(pred == y))

# demo run
_rng = np.random.default_rng(0)
_X = _rng.normal(size=(1000, 5)) * [1, 10, 0.1, 1, 3] + [0, 50, 0, 0, 0]
_y = (_X @ np.array([1.0, 0.1, 5.0, -1.0, 0.0]) - 5 + _rng.normal(size=1000) > 0).astype(float)
Xtr, ytr, Xva, yva = make_split(_X, _y)
Xtr, Xva = prepare(Xtr, Xva)
w_hat, b_hat, hist = train(Xtr, ytr, Xva, yva)
print(f"epochs run: {len(hist)}, best val loss: {min(hist):.4f}, val accuracy: {accuracy(w_hat, b_hat, Xva, yva):.3f}")
'''

MCQ = [
    {"q": "Which one of these bugs would clearly show up in the **training-loss** curve?",
     "options": ["`prepare` computes the mean and standard deviation over training and validation rows together.",
                 "`predict` thresholds the logit at 0.5 instead of 0.",
                 "`make_split` shuffles `X` and `y` with two different permutations.",
                 "`accuracy` reshapes the predictions to `(n, 1)` before comparing them with `y`."],
     "answer": 2,
     "why": r"Independent shuffles pair each row with a random label, so nothing is learnable and the training loss stays near $\ln 2 \approx 0.69$. The other three leave training untouched: they leak statistics or corrupt only the reported metrics, so you catch them with unit tests and assertions on shapes and splits."},
    {"q": "In `train`, the buggy code does `best_w = w` on an improvement and later `w -= lr * dw`. What does `train` return as `best_w`?",
     "options": ["The weights from the best validation epoch: `-=` builds a new array, so `best_w` keeps the old values.",
                 "The weights from the last epoch run: `best_w` and `w` are the same array, which `-=` updates in place. Store `w.copy()` instead.",
                 "All zeros: `best_w` was bound before training started and never changes.",
                 "It raises an error, because the bias `b` is a float and cannot be updated in place."],
     "answer": 1,
     "why": r"`w -= ...` mutates the array that `best_w` also refers to, so the saved \"best\" weights silently follow training to the end. `b -= ...` rebinds a new float, which is why only the weights need the copy."},
]

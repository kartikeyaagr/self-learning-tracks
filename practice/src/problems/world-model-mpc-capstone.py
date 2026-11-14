import numpy as np

META = dict(title='Capstone: learn a dynamics model and control with it', tags=['Reinforcement Learning', 'Optimization'], diff='Hard', runtime='numpy',
            related=['episodes-to-transition-pairs', 'cem-planning-round', 'random-shooting-planner'])

STATEMENT = r"""
Fit a world model from logged transitions, then use it inside a model-predictive controller on the real system.

Write `fit_and_control(S, U, S_next, s0, noise, lam, n_elite, alpha, A_true, B_true)`:

1. **Fit.** Solve the least-squares problem $[S\ U]\,\Theta \approx S_{next}$ with `np.linalg.lstsq` (no bias term). Then $\hat A = \Theta_{[:d_s]}^\top$ and $\hat B = \Theta_{[d_s:]}^\top$.
2. **Control.** `noise` has shape `(T, I, N, H, du)`: $T$ control steps, $I$ CEM rounds per step, $N$ candidates, horizon $H$. At each control step:
   - start CEM from mean 0 and std 1 (shape `(H, du)`);
   - run $I$ rounds of CEM under $(\hat A, \hat B)$ exactly as in the CEM step (reward $-(\lVert s'\rVert^2 + \lambda\lVert a\rVert^2)$, elites with index tie-breaks, smoothing `alpha`);
   - execute the first action of the final mean on the **true** system, $s \leftarrow A_{true}s + B_{true}a$, and add that step's true reward.

Return a dict: `A_hat`, `B_hat` (rounded to 6 decimals), `states` (the $T+1$ true states visited, starting at `s0`), `actions` (the $T$ executed actions) and `total_reward`.
"""

STARTER = '''import numpy as np

def fit_and_control(S, U, S_next, s0, noise, lam, n_elite, alpha, A_true, B_true):
    """Least-squares dynamics model, then CEM model-predictive control on the real system."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def _ret(A, B, s0, seq, lam):
    s, R = np.asarray(s0, float), 0.0
    for a in seq:
        s = A @ s + B @ a
        R -= float(s @ s + lam * (a @ a))
    return R

def _cem(A, B, s0, mean, std, noise, lam, n_elite, alpha):
    C = mean + std * noise
    R = np.array([_ret(A, B, s0, c, lam) for c in C])
    E = C[np.lexsort((np.arange(len(R)), -R))[:n_elite]]
    return alpha * E.mean(0) + (1 - alpha) * mean, alpha * E.std(0) + (1 - alpha) * std

def fit_and_control(S, U, S_next, s0, noise, lam, n_elite, alpha, A_true, B_true):
    S, U, S_next = (np.asarray(a, float) for a in (S, U, S_next))
    A_true, B_true = np.asarray(A_true, float), np.asarray(B_true, float)
    ds = S.shape[1]
    Theta = np.linalg.lstsq(np.hstack([S, U]), S_next, rcond=None)[0]
    A_hat, B_hat = Theta[:ds].T, Theta[ds:].T
    noise = np.asarray(noise, float)
    T, I, N, H, du = noise.shape
    s = np.asarray(s0, float)
    states, actions, total = [s.tolist()], [], 0.0
    for t in range(T):
        mean, std = np.zeros((H, du)), np.ones((H, du))
        for i in range(I):
            mean, std = _cem(A_hat, B_hat, s, mean, std, noise[t, i], lam, n_elite, alpha)
        a = mean[0]
        s = A_true @ s + B_true @ a
        total -= float(s @ s + lam * (a @ a))
        states.append(s.tolist()); actions.append(a.tolist())
    return {"A_hat": np.round(A_hat, 6).tolist(), "B_hat": np.round(B_hat, 6).tolist(),
            "states": states, "actions": actions, "total_reward": total}
'''

HINTS = [
    'Stack the inputs as [S U] so one lstsq call fits A and B together; transpose because lstsq solves X Theta = Y with row vectors.',
    'Each control step replans from the true current state with a fresh CEM distribution.',
    'Collect the data with varied actions; if the logged actions never vary, B cannot be identified.',
]

A0 = np.array([[1.0, 0.1], [0.0, 1.0]])
B0 = np.array([[0.005], [0.1]])

def _case(seed, n_data, data_noise, T, I, N, H, ne, alpha, act_scale):
    g = np.random.default_rng(seed)
    S = g.normal(size=(n_data, 2))
    U = act_scale * g.normal(size=(n_data, 1))
    S_next = S @ A0.T + U @ B0.T + data_noise * g.normal(size=(n_data, 2))
    return (S, U, S_next, [1.0, 0.0], g.normal(size=(T, I, N, H, 1)), 0.01, ne, alpha, A0, B0)

TESTS = [
    _case(1, 50, 0.0, 5, 3, 32, 5, 4, 1.0, 1.0),
    _case(2, 50, 0.01, 8, 3, 32, 6, 4, 0.9, 1.0),
    _case(3, 10, 0.1, 6, 2, 16, 4, 2, 1.0, 1.0),
    _case(4, 200, 0.01, 10, 4, 64, 8, 8, 0.8, 1.0),
    _case(5, 50, 0.0, 5, 3, 32, 5, 4, 1.0, 0.0),
]

EXPLANATION = r"""
## The loop
This is model-based RL in miniature: turn logged experience into training pairs, fit a dynamics model, plan with CEM inside the model, act on the real system, and replan from the state you actually reached. Replanning is what keeps small model errors from compounding over the whole episode.

## What the tests show
- With clean, varied data (the first test) least squares recovers the true model to six decimals, and the controller pushes the state towards zero.
- With ten noisy samples (the third test), $\hat A$ and $\hat B$ are visibly wrong. Replanning from real states still keeps the controller working, which is the main argument for MPC over executing a whole plan open-loop.
- In the last test the logged actions are all zero. The input matrix $[S\ U]$ has a zero column, so lstsq returns the minimum-norm solution $\hat B = 0$: the model believes actions do nothing. The only thing that separates plans in its eyes is the action penalty, so the controller makes small aimless moves and the state drifts away from zero instead of towards it. A world model only knows what the data showed it.

## Toward real systems
Neural dynamics models replace least squares, ensembles flag uncertainty, and CEM runs on a GPU with thousands of candidates. PETS, PlaNet, Dreamer and TD-MPC are all variations on this loop.

## Interview follow-ups
- After deploying the controller you collect new data from its own behaviour. Why can retraining on that data alone make the model worse in states the controller rarely visits?
"""

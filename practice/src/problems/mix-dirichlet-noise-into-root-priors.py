import numpy as np

META = dict(title='Dirichlet noise on the root priors', tags=['Reinforcement Learning', 'Probability'], diff='Easy',
            runtime='numpy', lessons=[],
            related=['initialize-legal-mcts-edge-statistics', 'turn-root-visits-into-search-policy'])

STATEMENT = r"""
Self-play only discovers a good move if search ever tries it. To guarantee some exploration even where the network is confident, AlphaGo Zero mixes Dirichlet noise into the priors **at the root only**:

$$P'(a) = (1 - \varepsilon)\, P(a) + \varepsilon\, \eta_a, \qquad \eta \sim \mathrm{Dir}(\alpha, \ldots, \alpha) \text{ over the legal actions.}$$

A Dirichlet sample is a vector of independent $\mathrm{Gamma}(\alpha, 1)$ draws divided by their sum. To keep the result reproducible, the gamma draws are passed in.

Write `noisy_root_priors(P, legal, gamma_draws, eps)`.

## Inputs
- `P`: float `(A,)`, root priors, already `0` on illegal actions and summing to 1 over legal ones.
- `legal`: bool `(A,)`, at least one `True`.
- `gamma_draws`: float `(A,)` of positive values, one $\mathrm{Gamma}(\alpha, 1)$ draw per action. Only the entries at legal actions are used: $\eta_a = g_a / \sum_{b \text{ legal}} g_b$ for legal $a$.
- `eps`: mixing weight in $[0, 1]$.

## Output
Float array `(A,)`: $P'$ on legal actions, exactly `0.0` on illegal ones. It sums to 1.
"""

STARTER = '''import numpy as np

def noisy_root_priors(P, legal, gamma_draws, eps):
    """Return (1 - eps) * P + eps * eta on legal moves, 0 elsewhere; eta = normalised legal gamma draws."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def noisy_root_priors(P, legal, gamma_draws, eps):
    legal = np.asarray(legal, dtype=bool)
    g = np.where(legal, np.asarray(gamma_draws, dtype=float), 0.0)   # illegal draws are discarded
    eta = g / g.sum()                                                  # normalised gammas = a Dirichlet sample
    return np.where(legal, (1 - eps) * np.asarray(P, dtype=float) + eps * eta, 0.0)
'''

HINTS = [
    'Turn the gamma draws into a Dirichlet sample by normalising them, but only over legal actions.',
    'Zero the illegal draws with `np.where(legal, gamma_draws, 0.0)`, then divide by the sum.',
    'Return `np.where(legal, (1 - eps) * P + eps * eta, 0.0)`.',
]

_rng = np.random.default_rng(41)
_A = 26
_P = np.where(np.arange(_A) % 3 != 0, _rng.random(_A), 0.0); _P /= _P.sum()
_L = _P > 0
_G = _rng.gamma(0.3, 1.0, _A)

TESTS = [
    (np.array([0.5, 0.5, 0.0]), np.array([True, True, False]), np.array([1.0, 3.0, 2.0]), 0.25),
    (np.array([1.0, 0.0]), np.array([True, True]), np.array([2.0, 2.0]), 0.5),
    (np.array([0.2, 0.3, 0.5]), np.array([True, True, True]), np.array([0.4, 0.1, 0.5]), 0.0),
    (np.array([0.2, 0.3, 0.5]), np.array([True, True, True]), np.array([0.4, 0.1, 0.5]), 1.0),
    (np.array([0.0, 1.0, 0.0]), np.array([False, True, False]), np.array([5.0, 0.1, 7.0]), 0.25),
    (_P, _L, _G, 0.25),
    (_P, _L, _G, 0.6),
]

EXPLANATION = r"""
## Why noise at the root
MCTS spends visits where the prior points. If the network assigns a strong move probability $10^{-4}$, the search will almost never try it and self-play will never generate data showing it is good. Mixing in $\varepsilon = 0.25$ of noise guarantees every legal root move a share of prior mass, so the search occasionally checks it. Noise is added only at the root: the root decides the move actually played and the training target, while deeper nodes should use the best available prior.

## Why $\alpha$ is small
With $\alpha < 1$ a Dirichlet sample is spiky: most of the mass lands on a few random moves. That focuses exploration on a handful of alternatives per move instead of smearing it thinly across all of them. AlphaGo Zero used $\alpha = 0.03$ for 19x19 Go; a common heuristic is $\alpha \approx 10 / \text{(typical number of legal moves)}$.

## Gammas to Dirichlet
If $g_i \sim \mathrm{Gamma}(\alpha_i, 1)$ independently, then $g / \sum_i g_i \sim \mathrm{Dir}(\alpha_1, \ldots, \alpha_k)$. That is exactly how `rng.dirichlet` works; passing the draws in makes the exercise reproducible.

## Common bugs
- Normalising the noise over **all** actions, which leaks noise mass onto illegal moves and leaves the legal priors summing to less than 1.
- Adding noise at every node in the tree.
- Reusing the same noise vector across moves.

## Interview follow-ups
- KataGo replaced uniform-$\alpha$ noise with noise shaped by the policy. What problem with uniform noise does that address?
- How do Gumbel-based root selection methods (Gumbel AlphaZero) guarantee policy improvement with very few simulations?
"""

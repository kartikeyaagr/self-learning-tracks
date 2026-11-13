import numpy as np

META = dict(title='Epsilon-greedy actions and their probabilities', tags=['Reinforcement Learning', 'Probability'],
            diff='Easy', runtime='numpy', lessons=['air-fanout-v1:07.03'],
            related=['softmax-boltzmann-exploration', 'ucb1-for-multi-armed-bandits', 'expected-sarsa'])

STATEMENT = r"""
An $\epsilon$-greedy policy explores with probability $\epsilon$ (uniformly random action) and otherwise exploits the greedy action. To keep it testable, the randomness is passed in.

Write `epsilon_greedy(Q, epsilon, u, random_actions)` for a batch of states and return `(actions, probs)`.

## Inputs
- `Q`: float array `(B, A)`, one row of action values per state.
- `epsilon`: float in `[0, 1]`.
- `u`: float array `(B,)` of uniform draws in `[0, 1)`.
- `random_actions`: int array `(B,)` of uniform random actions in `[0, A)`.

## Rules
- The greedy action of a row is its `argmax`, ties to the **lowest** index.
- Row $i$ explores if `u[i] < epsilon`, and then takes `random_actions[i]` (which may happen to equal the greedy action). Otherwise it takes the greedy action.
- `probs[i]` is the probability the $\epsilon$-greedy policy assigns to the action actually taken:

$$\pi(a \mid s) = \frac{\epsilon}{A} + (1 - \epsilon)\,[a = a_{\text{greedy}}]$$

## Output
`actions`: int array `(B,)`. `probs`: float array `(B,)`. These probabilities are what an off-policy method needs for importance weights.
"""

STARTER = '''import numpy as np

def epsilon_greedy(Q, epsilon, u, random_actions):
    """Q: (B, A). Return (actions (B,) ints, probability of each taken action (B,))."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def epsilon_greedy(Q, epsilon, u, random_actions):
    Q = np.asarray(Q, dtype=float)
    A = Q.shape[1]
    greedy = Q.argmax(axis=1)                                   # first max on ties
    actions = np.where(np.asarray(u) < epsilon, random_actions, greedy)
    probs = epsilon / A + (1.0 - epsilon) * (actions == greedy)
    return actions, probs
'''

HINTS = [
    'Find each row\'s greedy action first; `argmax(axis=1)` already prefers the lowest index.',
    '`np.where(u < epsilon, random_actions, greedy)` picks every row\'s action at once.',
    'Every action gets $\\epsilon / A$; the greedy one gets an extra $1 - \\epsilon$. That holds whether you got there by exploring or exploiting.',
]

_r = np.random.default_rng(101)
QR = _r.normal(size=(8, 4))
TESTS = [
    (np.array([[1.0, 3.0, 2.0], [0.0, 0.0, -1.0]]), 0.1, np.array([0.5, 0.05]), np.array([2, 2])),
    (np.array([[1.0, 3.0, 2.0], [0.0, 0.0, -1.0]]), 0.1, np.array([0.05, 0.5]), np.array([1, 2])),
    (np.array([[5.0, 5.0, 5.0]]), 0.0, np.array([0.0]), np.array([2])),
    (np.array([[5.0, 1.0]]), 1.0, np.array([0.999]), np.array([0])),
    (np.array([[5.0, 1.0]]), 1.0, np.array([0.999]), np.array([1])),
    (np.array([[2.0]]), 0.3, np.array([0.1]), np.array([0])),
    (QR, 0.25, _r.uniform(size=8), _r.integers(0, 4, 8)),
    (QR, 0.9, _r.uniform(size=8), _r.integers(0, 4, 8)),
]

EXPLANATION = r"""
## Why the probability has two terms
There are two ways to end up on the greedy action: exploit (probability $1-\epsilon$) or explore and draw it at random ($\epsilon/A$). A non-greedy action can only be reached by exploring. Off-policy corrections such as importance sampling need this exact $\pi(a|s)$, and forgetting the $\epsilon/A$ share on the greedy action is a classic mistake (test 2, row 0, explores onto the greedy action).

## Edge cases
- $\epsilon = 0$: always greedy, probability 1 (test 3, also a three-way tie).
- $\epsilon = 1$: uniform, every action has probability $1/A$ (tests 4 and 5).
- A single action: probability 1 regardless of $\epsilon$ (test 6).

## Complexity
$O(BA)$ for the argmax, everything else is $O(B)$.

## Common bugs
- Comparing `u > epsilon` (explore and exploit swapped).
- Excluding the greedy action from the random draw. That's a different policy and changes the probabilities to $\epsilon/(A-1)$.
- Breaking ties randomly, which makes `probs` inconsistent with the action actually logged.

## Interview follow-ups
- Why decay $\epsilon$ over training, and what does GLIE require of the schedule?
- When does $\epsilon$-greedy explore badly compared with UCB or Thompson sampling? (It explores uniformly, ignoring how uncertain each action is.)
"""

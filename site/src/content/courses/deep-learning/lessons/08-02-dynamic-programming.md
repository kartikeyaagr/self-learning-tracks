---
id: "08.02"
course: "deep-learning"
module: "08"
slug: "dynamic-programming"
title: "Dynamic programming"
keyIdeas: ["Dynamic programming solves a known MDP by repeatedly applying **Bellman backups**: recompute each state's value from its successors' current values. It needs $P$ and $R$, and a dense sweep costs $O(|\\mathcal S|^2|\\mathcal A|)$.", "Iterative policy evaluation repeats the Bellman expectation backup until values stop changing. It converges because the backup is a $\\gamma$-contraction in the max norm, so the error shrinks by at least a factor $\\gamma$ per sweep.", "Policy improvement makes the policy greedy with respect to $Q^\\pi$. The policy improvement theorem guarantees the new policy is at least as good in every state, and if nothing changes the policy is optimal.", "Policy iteration alternates full evaluation and improvement and usually stops after a handful of rounds; value iteration folds them into one max-backup per sweep. Modified policy iteration, with a few evaluation sweeps per improvement, is often cheapest.", "The error after $k$ value-iteration sweeps is at most $\\gamma^k$ times the initial error, so $\\gamma=0.99$ needs roughly ten times more sweeps than $\\gamma=0.9$ for the same accuracy."]
question: "On a 10,000-state gridworld with $\\gamma=0.99$, value iteration takes about 1,800 sweeps to converge, while policy iteration converges in 6 iterations. Which is faster in wall-clock time, and what does the answer depend on? Suggest a middle ground."
answer: "It depends on the cost of each policy-iteration evaluation. If evaluation is done exactly by solving the linear system $(I-\\gamma P_\\pi)V=r_\\pi$, each iteration costs $O(|S|^3)\\approx 10^{12}$ operations for a dense matrix, versus about $|S|\\cdot|A|\\cdot$(successors per action) operations per value-iteration sweep; for a gridworld with 4 actions and 3 successors that is roughly $1.2\\times10^5$ per sweep, so 1,800 sweeps are about $2\\times10^8$ operations. A dense solve therefore loses badly. But gridworld transitions are sparse (each state has a few successors), and a sparse solver on a 10,000-state grid is cheap, so 6 sparse solves plus 6 greedy sweeps can beat 1,800 sweeps. If evaluation is iterative instead, each policy-iteration step itself needs hundreds of sweeps at $\\gamma=0.99$, and the totals become comparable. The middle ground is modified policy iteration: run a fixed small number $m$ of evaluation sweeps (say 5 to 50) between greedy improvements, warm-starting each evaluation from the previous values. It keeps policy iteration's fast policy convergence without paying for exact evaluation of early, poor policies, and on small grids it often needs about half the work of plain value iteration. In-place (Gauss-Seidel) sweeps ordered backwards from the goal speed up either method further."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapter 4: policy evaluation, the improvement theorem, policy and value iteration, GPI."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 3 (planning by DP), including the contraction-mapping proof."}]
practice: ["value-iteration", "policy-iteration", "value-iteration-step", "bellman-expectation-equation"]
tags: ["dynamic programming", "value iteration", "policy iteration"]
tools: ["NumPy"]
---

## What you will learn

- What **dynamic programming** (DP) means in reinforcement learning: solving an MDP exactly when you know its rules.
- How a **Bellman backup** works, and how repeating it state by state computes the value of a policy.
- How to turn values into a better policy (policy improvement), and why the new policy can never be worse.
- The two classic algorithms, **policy iteration** and **value iteration**, worked by hand and in NumPy.
- Why both converge, how many sweeps they need, and why DP stops being practical for large problems.

*Before you start:* you need lesson 08.01 (states, actions, policies, returns, $V^\pi$, $Q^\pi$ and the two Bellman equations). The code uses NumPy arrays of shape (states, actions, next states); the matrix form $V^\pi=(I-\gamma P_\pi)^{-1}r_\pi$ from 08.01 comes back once.

## From equations to an algorithm

Lesson 08.01 ended with two equations. The **Bellman expectation equation** says what the values of a fixed policy must satisfy; the **Bellman optimality equation** says what the best possible values must satisfy. Both describe the answer without saying how to compute it, and the max in the optimality equation rules out a single matrix inversion.

This lesson turns those equations into algorithms. The setting is **planning**: you know the model, meaning the transition probabilities $P(s'\mid s,a)$ and the rewards $R$. Think of a board game's rules or a gridworld you wrote yourself. Nothing is learned from experience here, yet this lesson underpins the rest of the module: Monte Carlo, TD learning, Q-learning and actor-critic (08.03 to 08.06) are all DP updates with sampled transitions in place of the known model.

## The idea in one picture

Take a small robot on a 4 by 4 floor. The top-right cell is a charger (the **goal**, reward +1 when the robot enters it); one cell in the second row is a hole (the **pit**, reward −1). Both end the episode. Every other move gives reward 0, and the discount is $\gamma=0.9$, so reaching the charger sooner is worth more.

The floor is slippery: the robot moves where it intended with probability 0.8, and slides to one of the two perpendicular directions with probability 0.1 each. Bumping into a wall leaves it where it is. So the model is

| Intended move | Goes that way | Slides left of it | Slides right of it
| --- | --- | --- | --- |
| up | 0.8 | 0.1 (left) | 0.1 (right)
| right | 0.8 | 0.1 (up) | 0.1 (down)

and similarly for down and left. Running the value iteration algorithm from this lesson for 32 sweeps gives the following.

~~~dg-fig 08.02-grid.svg
The answer DP computes: the best expected discounted return from each cell (number) and the best action there (arrow). Darker cells are closer, in expected time, to the charger.
~~~

Read it like a map. From the start cell (bottom left) the best you can expect is 0.49; next to the charger it is 0.96. The arrows mostly point toward the charger, but look at the cell just below the pit: its arrow points *down*, away from the goal. Moving right from there would risk a 10% slide up into the pit; moving down can only slide left or right, which is safe. The cell left of the pit is a close call: moving up (value 0.324) accepts a 10% slide into the pit, while pressing left against the wall (0.322) is safe but wastes time. DP found both judgements on its own, simply by computing expected returns.

Two terms:

- A **backup** for state $s$ recomputes $V(s)$ from the current values of the states $s$ can reach in one step. Information flows *backwards*, from successors to predecessors.
- A **sweep** applies one backup to every state once.

## Step 1: one backup for a fixed policy

Suppose you have a policy $\pi$ and a guess $V_k$ for its values. The Bellman expectation equation from 08.01 says what $V^\pi(s)$ should equal; just evaluate its right-hand side with the current guess:

$$V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V_k(s')\big]$$

In words: for each action the policy might take, for each place you might land, add the reward you get on the way to the discounted value of where you land, and average with the policy and model probabilities. The symbols:

- $V_k(s)$: the current estimate for state $s$ after $k$ sweeps;
- $\pi(a\mid s)$: the probability the policy picks action $a$ in $s$;
- $P(s'\mid s,a)$: the probability of landing in $s'$;
- $R(s,a,s')$: the reward for that transition;
- $\gamma$: the discount factor.

Start from $V_0=0$ and sweep repeatedly. This is **iterative policy evaluation**. It gives the same answer as the matrix solve $V^\pi=(I-\gamma P_\pi)^{-1}r_\pi$ from 08.01, but each sweep only needs one multiply by the transition matrix, which stays cheap when transitions are sparse (each grid cell has at most 3 successors per action). A dense sweep costs $O(|\mathcal S|^2|\mathcal A|)$ operations.

## Step 2: make the policy greedy

Once you know $V^\pi$, you can score every action with a one-step lookahead:

$$Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^\pi(s')\big]$$

This is the value of taking $a$ now and following $\pi$ afterwards. The improved policy picks the best: $\pi'(s)=\arg\max_a Q^\pi(s,a)$. This is **policy improvement**, and $\pi'$ is called **greedy** with respect to $V^\pi$.

The **policy improvement theorem** promises $V^{\pi'}(s)\ge V^\pi(s)$ in every state. The intuition: in state $s$, taking the better first action and then going back to $\pi$ is already at least as good as following $\pi$ from the start, because $\max_a Q^\pi(s,a)\ge\sum_a\pi(a\mid s)Q^\pi(s,a)=V^\pi(s)$. Using $\pi'$ in the later states too applies the same argument again at every step, so it can only help. And if greedy changes nothing, then $V^\pi$ already satisfies the Bellman optimality equation, so $\pi$ is optimal.

<div class="ex">
<div class="lab">Worked example</div>

**evaluate and improve a random policy in a corridor**

A corridor of three cells, 1, 2, 3, with the goal to the right of cell 3. Actions are left and right, moves are deterministic, moving left from cell 1 bumps the wall and stays, entering the goal gives +1 and ends the episode. $\gamma=0.9$. The policy is random: left or right with probability 0.5 each.

Bellman backups for this policy (the goal's value is 0 because nothing happens after it): $V(1)=0.5\cdot0.9V(1)+0.5\cdot0.9V(2)$, $V(2)=0.45V(1)+0.45V(3)$, $V(3)=0.45V(2)+0.5\cdot(1+0)$.

Sweep 1 from $V_0=(0,0,0)$: only cell 3 can reach the reward, so $V_1=(0,\ 0,\ 0.5)$.

Sweep 2: $V(2)=0.45\cdot0+0.45\cdot0.5=0.225$, $V(3)=0.45\cdot0+0.5=0.5$, so $V_2=(0,\ 0.225,\ 0.5)$.

Sweep 3: $V(1)=0.45\cdot0+0.45\cdot0.225=0.101$, $V(2)=0.225$, $V(3)=0.45\cdot0.225+0.5=0.601$. The values creep toward the exact solution $V^\pi=(0.429,\ 0.524,\ 0.736)$ (from solving the three linear equations).

Improve. In cell 1: $Q(1,\text{left})=0.9\cdot0.429=0.386$ and $Q(1,\text{right})=0.9\cdot0.524=0.472$. In cell 2: left $0.386$, right $0.9\cdot0.736=0.662$. In cell 3: left $0.9\cdot0.524=0.472$, right $1+0=1$. Greedy picks right everywhere.

The new policy's values are $(0.81,\ 0.9,\ 1)$, higher in every cell, as the theorem promised. Improving once more changes nothing, so this policy is optimal.

</div>

## Step 3: policy iteration

Alternate the two steps until the policy stops changing:

~~~dg-flow Policy iteration: evaluate the current policy, make it greedy, repeat until the greedy step changes nothing.
Policy pi -(evaluate)-> V pi -(greedy w.r.t. Q)-> *Improved pi -(repeat until stable)-> Optimal pi
~~~

Policy iteration must stop: there are finitely many deterministic policies, and each round strictly improves until the last, so none repeats. In practice the rounds are few. Here it is on the slippery grid, starting from the bad policy "always move left":

~~~dg-fig 08.02-pi.svg
Policy iteration on the 4 by 4 grid. Each panel is a policy (arrows) with its exact values. Always-left (iteration 0) never reaches the charger; four improvements later the policy is optimal and the start cell's value has risen from 0 to 0.49.
~~~

The number of changed arrows per round was 10, 9, 4, 2 and then 0. Round 1 (not drawn) only reached the charger from the three cells next to it and scored 0 almost everywhere else; the payoff appears once a path to the charger exists, and the last rounds only fine-tune cells near the start. The price is hidden inside each round: a full policy evaluation, either a linear solve or many sweeps.

## Step 4: value iteration

Policy evaluation does not have to finish before you improve. Do just one sweep of evaluation, and fold the greedy choice straight into it by taking the max over actions instead of averaging with $\pi$:

$$V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V_k(s')\big]$$

This is **value iteration**. It is the Bellman optimality equation used as an update rule, and the only change from Step 1 is $\sum_a\pi(a\mid s)$ becoming $\max_a$. Stop when the largest change in a sweep falls below a small threshold $\theta$, then read off the greedy policy.

~~~dg-fig 08.02-sweeps.svg
Value iteration on the grid, starting from all zeros. After sweep 1 only the two cells beside the charger know about it (0.8, its chance of arriving); each further sweep pushes value one more step back. By sweep 10 the values are within about 0.03 of the final ones.
~~~

Value travels outward from the reward, about one cell per sweep. The bottom-left start cell is 6 moves from the charger and stays at 0 through sweep 5 and first becomes positive (0.25) at sweep 6.

<div class="ex">
<div class="lab">Worked example</div>

**value iteration by hand in the corridor**

Same corridor, $\gamma=0.9$, $V_0=(0,0,0)$. Each backup takes the max of the left and right lookaheads.

Sweep 1: cell 3 can step right into the goal: $\max(0.9\cdot0,\ 1+0)=1$. Cells 1 and 2 see only zeros. $V_1=(0,\ 0,\ 1)$.

Sweep 2: cell 2: $\max(0.9\cdot0,\ 0.9\cdot1)=0.9$. $V_2=(0,\ 0.9,\ 1)$.

Sweep 3: cell 1: $\max(0.9\cdot0,\ 0.9\cdot0.9)=0.81$. $V_3=(0.81,\ 0.9,\ 1)$.

Sweep 4 changes nothing, so stop. These are the optimal values found by policy iteration above, and the greedy policy is right everywhere.

</div>

## DP in code

Store the model as two arrays: `P[s, a, s2]` for transition probabilities and `R[s, a]` for the expected immediate reward. Then one backup for all states and actions at once is `R + gamma * P @ V`, an array of shape (states, actions).

```python
import numpy as np

def make_grid(n=4, goal=(0, 3), pit=(1, 1), slip=0.1):
    # Slippery n x n gridworld. Actions 0..3 = up, right, down, left.
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    S = n * n; P = np.zeros((S, 4, S)); R = np.zeros((S, 4))
    end = {goal[0] * n + goal[1]: 1.0, pit[0] * n + pit[1]: -1.0}
    for s in range(S):
        if s in end: P[s, :, s] = 1.0; continue          # terminal: absorbing, reward 0
        r, c = divmod(s, n)
        for a in range(4):
            for d, p in [(a, 1 - 2 * slip), ((a + 1) % 4, slip), ((a + 3) % 4, slip)]:
                rr = min(max(r + moves[d][0], 0), n - 1)  # bumping a wall = stay put
                cc = min(max(c + moves[d][1], 0), n - 1)
                s2 = rr * n + cc
                P[s, a, s2] += p; R[s, a] += p * end.get(s2, 0.0)
    return P, R

def value_iteration(P, R, gamma, tol=1e-8):
    V = np.zeros(P.shape[0]); k = 0
    while True:
        V_new = (R + gamma * P @ V).max(1); k += 1   # Bellman optimality backup
        if np.abs(V_new - V).max() < tol:
            return V_new, (R + gamma * P @ V_new).argmax(1), k
        V = V_new

def policy_iteration(P, R, gamma):
    S = np.arange(P.shape[0]); pi = np.zeros(len(S), int); it = 0
    while True:
        V = np.linalg.solve(np.eye(len(S)) - gamma * P[S, pi], R[S, pi])  # exact evaluation
        new = (R + gamma * P @ V).argmax(1); it += 1                     # greedy improvement
        if (new == pi).all(): return V, pi, it
        pi = new

P, R = make_grid()
V, pi, k = value_iteration(P, R, 0.9)
print(k, V.reshape(4, 4).round(2))
# 32 [[0.55 0.65 0.96 0.  ]
#     [0.32 0.   0.85 0.96]
#     [0.44 0.51 0.74 0.83]
#     [0.49 0.56 0.65 0.72]]
V2, pi2, it = policy_iteration(P, R, 0.9)
print(it, np.abs(V - V2).max() < 1e-6, (pi == pi2).all())
# 5 True True     (5 evaluations from "always up": 4 improvements + 1 check)
```

Terminal states are absorbing with zero reward, so the backup never adds value after the episode ends (08.01 explains why). `P[S, pi]` picks each state's transition row for the policy's action: the matrix $P_\pi$ from 08.01.

## Checking the result

- **The Bellman residual.** Compute `np.abs((R + gamma * P @ V).max(1) - V).max()`. If it is $\epsilon$, then $V$ is within $\epsilon/(1-\gamma)$ of $V^*$ (a standard bound), so a small residual at $\gamma=0.99$ still allows an error 100 times larger.
- **Two methods, one answer.** Policy iteration and value iteration must agree on the values, as in the code. Policies may legitimately differ where two actions tie.
- **Sanity values.** No value can exceed the largest reward divided by $1-\gamma$. In the grid every value is below 1, since at most one +1 is ever collected.
- **Look at the arrows.** A region of exact zeros usually means value has not propagated there yet, or the model has a bug.

## How many sweeps?

Each value-iteration sweep shrinks the worst-case error by at least a factor $\gamma$:

$$\|V_k-V^*\|_\infty\le\gamma^k\,\|V_0-V^*\|_\infty,$$

where $\|\cdot\|_\infty$ means the largest absolute error over all states.

<div class="ex">
<div class="lab">Worked example</div>

**sweeps needed for a given accuracy**

Suppose the initial error is $\|V_0-V^*\|_\infty=10$ and you want the error below $10^{-3}$, a reduction by a factor $10^{-4}$. You need $\gamma^k\le10^{-4}$, so $k\ge\ln(10^{-4})/\ln\gamma$.

With $\gamma=0.9$: $\ln(10^{-4})=-9.210$ and $\ln0.9=-0.1054$, so $k\ge87.4$: 88 sweeps.

With $\gamma=0.99$: $\ln0.99=-0.01005$, so $k\ge916.4$: 917 sweeps, about ten times more. A longer horizon makes DP, and every TD method later, slower to spread value.

</div>

~~~dg-fig 08.02-converge.svg
Measured value-iteration error on the 4 by 4 grid (solid) against the $\gamma^k$ guarantee (dashed). The guarantee is a worst case; here episodes end at the charger or the pit, so the real error falls faster, yet $\gamma=0.99$ still needs far more sweeps than $\gamma=0.9$.
~~~

On the grid, the error first drops below $10^{-6}$ after 26 sweeps for $\gamma=0.9$ and after 125 sweeps for $\gamma=0.99$, far fewer than the bound's worst case but still about five times more.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Values grow without limit | $\gamma=1$ in a task that can loop forever, or a terminal state that keeps paying reward | use $\gamma<1$; make terminals absorbing with zero reward
| Converges, but the policy is silly near the goal | bootstrapping past the end (terminal value not fixed at 0) | zero the terminal rows, or mask $V(s')$ with a done flag
| Runs for thousands of sweeps | $\gamma$ close to 1 and a long horizon | in-place updates, modified policy iteration, or a smaller $\gamma$ if the task allows
| Policy flips between two actions every iteration | exact ties broken differently each time | break ties consistently (lowest index) or keep the old action unless strictly better
| Memory error building `P` | dense array of size $\lvert S\rvert^2\lvert A\rvert$ | sparse transitions (lists of successors), or sample-based methods
| Stopped early with a poor policy | threshold on the change per sweep too loose at high $\gamma$ | use the $\epsilon/(1-\gamma)$ bound to set $\theta$, or run until the greedy policy stops changing

## Going deeper

**Why the sweeps converge.** Write one sweep as an operator $T$ acting on a whole value vector. For policy evaluation, $T^\pi V = r_\pi+\gamma P_\pi V$. For any two value vectors $U$ and $V$,

$$\|T^\pi U-T^\pi V\|_\infty=\gamma\|P_\pi(U-V)\|_\infty\le\gamma\|U-V\|_\infty,$$

because each row of $P_\pi$ is a probability average, and an average can never be larger than the largest entry. An operator that shrinks all distances by a factor $\gamma<1$ is a **contraction**, and the Banach fixed-point theorem says it has exactly one fixed point, reached from any start. That fixed point is $V^\pi$. The optimality operator $T^*$ (with the max) is also a $\gamma$-contraction, since $|\max_a f(a)-\max_a g(a)|\le\max_a|f(a)-g(a)|$. Its fixed point is $V^*$.

**In-place updates.** Overwriting $V(s)$ as soon as it is computed, so later states in the same sweep already use it, is still a contraction and usually converges faster, especially if states are swept in an order that follows value backwards from the goal. **Asynchronous DP** goes further and updates states in any order, as long as every state keeps being updated.

**Modified policy iteration and generalised policy iteration.** Value iteration does 1 evaluation sweep per improvement; policy iteration does "until converged". Anything in between, $m$ sweeps per improvement, is **modified policy iteration**.

~~~dg-fig 08.02-mpi.svg
Total work to solve a 10 by 10 slippery grid with $\gamma=0.99$, counting one improvement step as a full value-iteration sweep and one evaluation sweep as a quarter of one (4 actions). About 5 to 10 evaluation sweeps per improvement is cheapest here, roughly half the work of plain value iteration.
~~~

On that grid, value iteration needed 48 sweeps; $m=10$ needed 6 improvements and the equivalent of 21 sweeps in total; very large $m$ wastes effort evaluating early, poor policies precisely. All of these fit one pattern, **generalised policy iteration** (GPI): values are pushed toward the current policy's values, the policy is pushed toward greedy, and the two settle together at optimality. GPI reappears in Q-learning (08.04), in actor-critic (08.06), and in AlphaGo Zero, where search improves the policy and training evaluates it (09.04).

| Method | Backup | Work per iteration | Iterations
| --- | --- | --- | --- |
| Policy evaluation | expectation, fixed $\pi$ | $O(\lvert S\rvert^2\lvert A\rvert)$ per sweep | until $\gamma^k$ is small
| Policy iteration | full evaluation, then greedy | many sweeps or one linear solve | few
| Value iteration | optimality (max) | one sweep | about $\log\epsilon/\log\gamma$
| Modified policy iteration | $m$ expectation sweeps, then greedy | $m$ cheap sweeps plus one greedy sweep | between the two

**The limits of DP.** DP needs the model, which you rarely have; the next lessons replace the sum over $s'$ with transitions the agent actually experiences (08.03). And it touches every state on every sweep, which is impossible when states are Go positions (about $10^{170}$) or camera images. That motivates function approximation (deep RL, covered in the AI Research course) and spending computation only on states that matter, as tree search does (Module 09). This is Bellman's "curse of dimensionality": the number of states grows exponentially with the number of variables describing one.

## Key terms

| Term | Meaning
| --- | --- |
| Dynamic programming | Solving an MDP exactly from a known model by repeated Bellman backups
| Bellman backup | Recomputing one state's value from its successors' current values
| Sweep | One backup for every state
| Policy evaluation | Computing $V^\pi$ for a fixed policy, by sweeps or a linear solve
| Policy improvement | Making the policy greedy with respect to $Q^\pi$; never makes it worse
| Policy iteration | Alternating full evaluation and improvement until the policy is stable
| Value iteration | Repeating the max (optimality) backup; one sweep per improvement
| Contraction | An update that shrinks the distance between any two value vectors by a factor $\gamma$
| Generalised policy iteration | Any interleaving of evaluation and improvement that converges to optimality

<div class="take">
<div class="lab">Key takeaway</div>

with a known model, Bellman backups are $\gamma$-contractions, so repeated sweeps converge to a unique answer; policy iteration alternates full evaluation with greedy improvement and needs few rounds, value iteration does one max-backup per sweep, and both are the template that sample-based RL approximates.

</div>

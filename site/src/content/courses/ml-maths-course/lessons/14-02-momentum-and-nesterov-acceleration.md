---
id: "14.02"
course: "ml-maths-course"
module: "14"
slug: "momentum-and-nesterov-acceleration"
title: "Momentum and Nesterov acceleration"
keyIdeas: ["Heavy-ball momentum keeps a **velocity**, $v_{k+1} = \\beta v_k - \\eta\\nabla f(x_k)$, $x_{k+1} = x_k + v_{k+1}$: gradients that keep their sign add up, gradients that flip sign cancel.", "Under a constant gradient the step settles at $\\eta/(1-\\beta)$: $\\beta = 0.9$ makes the **effective learning rate** 10 times larger and averages roughly the last 10 gradients.", "On a quadratic, tuned momentum contracts by $(\\sqrt\\kappa - 1)/(\\sqrt\\kappa + 1)$ per step, so step counts grow like $\\sqrt\\kappa$ instead of $\\kappa$: on the $\\kappa = 100$ bowl, 93 steps instead of 691.", "**Nesterov** takes the gradient at the look-ahead point $x_k + \\beta v_k$, braking before it overshoots; it achieves the optimal $O(1/k^2)$ rate on smooth convex problems, where heavy ball has no guarantee and can even cycle.", "Frameworks store momentum differently (PyTorch: $b \\leftarrow \\mu b + g$, $p \\leftarrow p - \\eta b$), equivalent to heavy ball for a constant learning rate. Switching momentum on without dividing $\\eta$ by about 10 is a classic way to make training diverge."]
question: "A model trains well with plain SGD at learning rate 0.1. A colleague switches on momentum 0.9 with the same learning rate and the loss immediately diverges. Explain why, why the quadratic stability analysis alone does not predict it, and what learning rate would be a sensible first try with momentum."
answer: "With momentum the steady-state step is $\\eta/(1-\\beta)$, so at $\\beta = 0.9$ the same $\\eta = 0.1$ behaves like a plain step of about 1.0, ten times larger: along every direction where the gradient keeps its sign, the weights now move ten times further per update. Interestingly, the quadratic stability condition alone would not predict divergence: heavy ball is stable for $\\eta\\lambda < 2(1+\\beta)$, which is looser than plain gradient descent's $\\eta\\lambda < 2$, so on a fixed quadratic any step that was stable without momentum stays stable with it. A network's loss is not a fixed quadratic. The tenfold larger moves carry the weights into regions of higher curvature, mini-batch noise is amplified along with the signal, and the accumulated velocity keeps pushing for several steps after the gradient has reversed, so the run overshoots and blows up (the two-moons experiment in the lesson shows exactly this at learning rate 2.0). A sensible first try keeps the effective step the same: $\\eta \\approx 0.1 \\times (1 - 0.9) = 0.01$, then tune upwards, since on many problems momentum can tolerate somewhat more than that. More generally, treat $\\beta$ and $\\eta$ as a pair: any change of $\\beta$ in a recipe should come with a learning-rate retune."
sources: [{"title": "Goh: Why Momentum Really Works (Distill)", "url": "https://distill.pub/2017/momentum/", "note": "Interactive derivation of the quadratic analysis, the sqrt(kappa) rate and the oscillation regimes."}, {"title": "Sutskever et al.: On the importance of initialization and momentum in deep learning", "url": "https://proceedings.mlr.press/v28/sutskever13.html", "note": "Classical versus Nesterov momentum in deep networks, and momentum schedules."}, {"title": "PyTorch: torch.optim.SGD", "url": "https://pytorch.org/docs/stable/generated/torch.optim.SGD.html", "note": "The exact momentum, dampening and Nesterov update the framework implements."}]
practice: ["classical-momentum", "nesterov-momentum", "custom-sgd-with-momentum", "optimizer-showdown"]
tags: ["momentum", "acceleration"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why plain gradient descent is slow on stretched valleys, and how adding a "velocity" fixes it.
- The heavy-ball update in words and symbols, and why $\beta = 0.9$ quietly multiplies your step size by 10.
- How to tune momentum on a quadratic, and why the step count then grows like $\sqrt\kappa$ instead of $\kappa$.
- What Nesterov's look-ahead gradient changes, and why it is the method with guarantees beyond quadratics.
- How PyTorch writes momentum, and the most common way switching it on breaks a training run.

*Before you start:* you need lesson 14.01 (gradient descent, the condition number $\kappa = L/\mu$, the $2/L$ stability limit and the eigenvector view of a quadratic). Eigenvalues come from module 07; a quadratic's Hessian from 09.08.

## From the last lesson: the stuck ball

Lesson 14.01 ended on the bowl $f(x, y) = \tfrac12(x^2 + 100y^2)$, with $\kappa = 100$. The steep direction forces $\eta < 2/100$, and with that small step the flat direction barely moves: about $\kappa$ steps per digit of accuracy. The gradient alone cannot fix this, because it points mostly *across* the valley. This lesson adds memory of previous steps and gets a square-root speed-up.

## The idea in one picture

Start both methods at the point $(-1, 0.12)$ on that bowl and count how far each is from the minimum after a number of steps:

| Steps taken | 10 | 25 | 40
| --- | --- | --- | --- |
| Gradient descent (best fixed step) | 0.82 | 0.61 | 0.45
| Heavy-ball momentum (tuned) | 0.49 | 0.052 | 0.0040

After 40 steps gradient descent has covered about half the distance; momentum is a hundred times closer.

~~~dg-fig 14.02-paths.svg
The same 40 steps on the bowl from lesson 14.01. Gradient descent (left) zig-zags across the valley while creeping along it; momentum (right) swings wider at first but rolls down the valley floor and lands on the minimum.
~~~

The momentum path swings *wider* across the valley, but those swings cancel out, while the small, consistent push along the valley floor adds up step after step. That is the whole idea.

Two new words:

- The **velocity** $v$ is a running memory of recent steps: each step keeps most of it and adds a little of the new gradient.
- The **momentum coefficient** $\beta$ (between 0 and 1) is how much old velocity is kept: 0 is plain gradient descent, 0.9 keeps 90%.

## Step 1: the heavy-ball update

Polyak's **heavy-ball method** (1964) replaces "step against the gradient" by two lines:

$$v_{k+1} = \beta v_k - \eta\nabla f(x_k), \qquad x_{k+1} = x_k + v_{k+1}.$$

In words: shrink the old velocity by $\beta$, add a gradient step of size $\eta$, then move by the result. The symbols:

- $x_k$ is the parameter vector at step $k$; $\nabla f(x_k)$ is the gradient there.
- $\eta$ is the learning rate, exactly as in gradient descent.
- $v_k$ is the velocity, starting at $v_0 = 0$.
- $\beta$ is the momentum coefficient.

Substituting the velocity away gives an equivalent one-line form that is easier to read:

$$x_{k+1} = x_k - \eta\nabla f(x_k) + \beta(x_k - x_{k-1}).$$

That is an ordinary gradient step plus a fraction $\beta$ of the previous displacement. Picture a heavy ball rolling on the loss surface: the gradient is the force, mass keeps it moving, and $1 - \beta$ acts as friction.

Number check, one dimension with $f(x) = \tfrac12 x^2$ (gradient $x$), $\eta = 0.1$, $\beta = 0.9$, $x_0 = 1$: $v_1 = 0 - 0.1 \times 1 = -0.1$, so $x_1 = 0.9$. Then $v_2 = 0.9 \times (-0.1) - 0.1 \times 0.9 = -0.18$, so $x_2 = 0.72$. Plain gradient descent would be at $0.81$ after two steps: momentum is already ahead.

## Step 2: momentum as a filter on the gradient

Unroll the velocity and you see it is a weighted sum of all past gradients:

$$v_{k} = -\eta\left(g_{k-1} + \beta g_{k-2} + \beta^2 g_{k-3} + \dots\right),$$

where $g_j = \nabla f(x_j)$. Recent gradients count fully, older ones fade by a factor $\beta$ per step. This is an **exponential moving average** (up to a scale factor) with a memory of roughly $1/(1-\beta)$ steps: about 10 steps at $\beta = 0.9$, about 100 at $\beta = 0.99$.

Its effect depends on whether the gradients are steady or alternating:

~~~dg-fig 14.02-filter.svg
Left: when the gradient is the same every step, the momentum step grows from 0.1 towards 1.0, ten times the plain step. Right: when the gradient flips sign every step, the momentum step settles at about ±0.053, roughly half the plain step.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the velocity by hand, $\eta = 0.1$, $\beta = 0.9$**

Steady gradient $g = 1$ every step. $v_1 = -0.1$. $v_2 = 0.9 \times (-0.1) - 0.1 = -0.19$. $v_3 = 0.9 \times (-0.19) - 0.1 = -0.271$. $v_4 = 0.9 \times (-0.271) - 0.1 = -0.3439$.

The pattern is $v_k = -0.1(1 + 0.9 + 0.9^2 + \dots + 0.9^{k-1})$, a geometric series whose limit is $-0.1/(1 - 0.9) = -1.0$. After 40 steps the run gives $-0.985$.

Alternating gradient $g = +1, -1, +1, \dots$ $v_1 = -0.1$. $v_2 = 0.9 \times (-0.1) + 0.1 = 0.01$. $v_3 = 0.9 \times 0.01 - 0.1 = -0.091$. $v_4 = 0.9 \times (-0.091) + 0.1 = 0.0181$.

Here the steps settle to an amplitude of $0.1/(1 + 0.9) \approx 0.053$ (the run gives 0.052 to 0.054). The steady direction got 10 times faster; the alternating one got about 1.9 times slower: a ratio of 19 in favour of the useful direction.

</div>

That is the opening figure explained: along the valley floor the gradient keeps its sign and is amplified; across the valley it flips and is damped. The left panel also gives the most practical fact in this lesson. In steady state the step is

$$\text{effective learning rate} = \frac{\eta}{1-\beta}.$$

At $\beta = 0.9$ that is $10\eta$; at $\beta = 0.99$, $100\eta$. Changing $\beta$ without touching $\eta$ changes the step size. With mini-batches the same averaging also smooths batch-to-batch noise.

~~~dg-flow Momentum filters the gradient sequence: persistent directions are amplified, alternating ones cancel.
Gradient g_k -(scale by eta)-> Velocity update -(add beta times v_k)-> *Velocity v_k+1 -(add to x)-> New iterate
~~~

## Step 3: how fast, and how to tune it

On a quadratic, lesson 14.01 split the error into independent eigendirections. Do the same here. In a direction with curvature $\lambda$, the error $e_k$ (distance to the minimum along that direction) obeys

$$e_{k+1} = (1 + \beta - \eta\lambda)\,e_k - \beta\,e_{k-1}.$$

Each step now depends on the two previous errors, so the behaviour is set by the two roots of

$$z^2 - (1 + \beta - \eta\lambda)\,z + \beta = 0.$$

The error shrinks like $|z|^k$ for the larger root. Two regimes matter:

- **Real roots** (small $\beta$): the error decays smoothly, like gradient descent, possibly slowly.
- **Complex roots** (large enough $\beta$): the error spirals in, and both roots have magnitude exactly $\sqrt\beta$, *whatever the curvature $\lambda$ is*.

That is the trick: put every direction, from the flattest $\mu$ to the steepest $L$, in the complex regime and all contract at $\sqrt\beta$, flat ones included. Doing so with the smallest $\beta$ gives

$$\eta = \frac{4}{(\sqrt L + \sqrt\mu)^2}, \qquad \beta = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2,$$

and the contraction per step

$$\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \approx 1 - \frac{2}{\sqrt\kappa}.$$

Compare gradient descent's best rate $(\kappa - 1)/(\kappa + 1) \approx 1 - 2/\kappa$. The $\kappa$ has turned into $\sqrt\kappa$.

<div class="ex">
<div class="lab">Worked example</div>

**the $\kappa = 100$ bowl from lesson 14.01**

Curvatures $\mu = 1$, $L = 100$, so $\sqrt\kappa = 10$.

Gradient descent with the best fixed step $\eta = 2/101$ contracts by $99/101 = 0.980$ per step. Steps for a millionfold reduction: $\ln(10^{-6})/\ln(0.980) = -13.82/-0.0200 \approx 690$. The run takes 691.

Tuned momentum: $\beta = (9/11)^2 = 0.669$ and $\eta = 4/(10 + 1)^2 = 4/121 = 0.033$. Contraction $9/11 = 0.818$ per step, so the rate predicts $-13.82/\ln(0.818) = -13.82/-0.201 \approx 69$ steps.

Check the flattest direction ($\lambda = 1$): $1 + \beta - \eta\lambda = 1 + 0.669 - 0.033 = 1.636$, and the discriminant $1.636^2 - 4 \times 0.669 = 2.678 - 2.678 \approx 0$. A double root at $1.636/2 = 0.818$: exactly on the edge of the complex regime, as designed. The steepest direction ($\lambda = 100$) gives $1.669 - 3.306 = -1.636$, a double root at $-0.818$.

The run takes 93 steps, not 69. Double roots make the error behave like $k \cdot 0.818^k$ rather than $0.818^k$, and the extra factor $k$ costs about 24 steps. Still 7.4 times fewer than gradient descent.

Note that $\eta = 0.033$ is above plain gradient descent's limit $2/L = 0.02$. Momentum's stability region is larger: $0 < \eta\lambda < 2(1 + \beta)$.

</div>

~~~dg-fig 14.02-rates.svg
Left: error against step on the kappa = 100 bowl. Both momentum methods reach a millionfold reduction in 93 to 155 steps; gradient descent needs 691. The dotted line is the predicted rate 0.818 per step. Right: measured step counts across condition numbers grow like kappa for gradient descent and like the square root of kappa for tuned momentum.
~~~

Going from $\kappa = 100$ to $10{,}000$ multiplies gradient descent's measured count by 100 but momentum's only by about 11.

| Method | Rate per step (quadratic) | Predicted steps for 1e-6, kappa = 100 | Measured, kappa = 100 | Measured, kappa = 10,000
| --- | --- | --- | --- | --- |
| Gradient descent, best fixed step | $(\kappa-1)/(\kappa+1)$ | about 690 | 691 | 69,078
| Heavy ball, tuned | $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ | about 69 | 93 | 1,056

In practice you rarely know $\mu$ and $L$, but the shape of the trade-off carries over:

~~~dg-fig 14.02-beta.svg
Left: with the learning rate fixed at 0.01, the worst contraction per step as beta varies. It falls until beta is just large enough (about 0.81, rate 0.90) and then rises along the square root of beta. Right: too little momentum is slow and smooth; too much rings for a long time.
~~~

With $\eta = 0.01$, plain gradient descent contracts by only 0.99 per step in the flat direction. Raising $\beta$ helps up to about 0.81 (rate 0.90); past that every direction rings at rate $\sqrt\beta$, creeping back towards 1. Too much momentum is not unstable here, just a ball with too little friction that keeps sloshing.

## Step 4: Nesterov's look-ahead

Heavy ball computes the gradient at the current point and then adds the velocity. But we already know the velocity will carry us to roughly $x_k + \beta v_k$. **Nesterov's accelerated gradient** (NAG) evaluates the gradient *there* instead:

$$v_{k+1} = \beta v_k - \eta\nabla f(x_k + \beta v_k), \qquad x_{k+1} = x_k + v_{k+1}.$$

The only change is where the gradient is taken: the **look-ahead point** $x_k + \beta v_k$. When the ball is about to roll up the far wall of the valley, heavy ball measures the slope where it stands; Nesterov measures it a little ahead, sees the wall rising, and brakes one step earlier.

On a quadratic the two are similar: with the textbook settings $\eta = 1/L$, $\beta = (\sqrt\kappa - 1)/(\sqrt\kappa + 1)$, Nesterov contracts by about $1 - 1/\sqrt\kappa$ per step (155 steps in the rates figure). The difference shows up off quadratics.

~~~dg-fig 14.02-nesterov.svg
Left: a smooth strongly convex function built from three quadratic pieces (condition number 25). Heavy ball with its quadratic-tuned settings falls into a 3-cycle and never converges; Nesterov converges. Right: a hard smooth convex problem without strong convexity. Nesterov's error falls like 1/k^2 and ends orders of magnitude below gradient descent's 1/k.
~~~

The left panel is a counterexample from Lessard, Recht and Packard (2016): a smooth strongly convex function ($\mu = 1$, $L = 25$) whose gradient is $25x$ for $x < 1$, $x + 24$ for $1 \le x < 2$ and $25x - 24$ beyond. Heavy ball with its quadratic-optimal settings ($\beta = 4/9$, $\eta = 1/9$), started at 3.2, cycles $2.116 \to 0.647 \to -1.802$ forever.

For smooth convex functions (no strong convexity), Nesterov used a growing momentum $\beta_k = (k-1)/(k+2)$ and proved

$$f(x_k) - f^\star \le \frac{2L\lVert x_0 - x^\star\rVert^2}{(k+1)^2},$$

where $f^\star$ is the minimum value and $x^\star$ a minimiser. Gradient descent only guarantees $O(1/k)$. No method that uses only gradients can beat $O(1/k^2)$ in the worst case, so this rate is optimal. In the right panel, after 1,000 steps gradient descent is at $2.5 \times 10^{-3}$ and Nesterov at $3.7 \times 10^{-6}$. Heavy ball has no comparable guarantee outside quadratics.

## Momentum in code

First the update itself, in NumPy, on the bowl from the examples:

```python
import numpy as np

H = np.array([1.0, 100.0])              # f(x, y) = 0.5 * (x^2 + 100 y^2)
def grad(x): return H * x

def run(eta, beta=0.0, nesterov=False, tol=1e-6, max_steps=5000):
    x = np.array([1.0, 1.0]); v = np.zeros(2); e0 = np.linalg.norm(x)
    for k in range(1, max_steps + 1):
        g = grad(x + beta * v) if nesterov else grad(x)   # look-ahead for Nesterov
        v = beta * v - eta * g                            # decayed memory + new push
        x = x + v
        if np.linalg.norm(x) < tol * e0:
            return k
    return None

print(run(eta=2/101))                            # 691  plain GD, best fixed step
print(run(eta=4/121, beta=(9/11)**2))            # 93   heavy ball, tuned
print(run(eta=1/100, beta=9/11, nesterov=True))  # 155  Nesterov, strongly convex settings
print(run(eta=0.02, beta=0.9))                   # 223  untuned defaults still beat GD
```

PyTorch's `torch.optim.SGD` keeps a buffer $b \leftarrow \mu b + g$ and steps $p \leftarrow p - \eta b$, where $\mu$ is its name for $\beta$. Set $v = -\eta b$ and you recover heavy ball exactly, as long as $\eta$ is constant. When a schedule changes $\eta$ mid-run, the two differ: PyTorch's buffer stores raw gradients, so a new learning rate applies to the whole remembered velocity at once. With `nesterov=True` PyTorch steps with $g + \mu b$, a rearranged Nesterov that needs only the gradient at the current parameters.

```python
import torch
H = torch.tensor([1.0, 100.0])
def steps(**kw):
    p = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = torch.optim.SGD([p], **kw)
    for k in range(1, 5001):
        opt.zero_grad(); (0.5 * (H * p**2).sum()).backward(); opt.step()
        if p.detach().norm() < 1e-6 * 2**0.5: return k

print(steps(lr=4/121, momentum=(9/11)**2))           # 93   same as heavy ball above
print(steps(lr=1/100, momentum=9/11, nesterov=True)) # 154
print(steps(lr=0.019))                               # 703  plain SGD
print(steps(lr=0.0019, momentum=0.9))                # 556  same effective step, small gain
```

In the last line the effective step $\eta/(1-\beta)$ equals plain SGD's 0.019, and momentum is still 21% faster; the tuned run uses a learning rate 17 times larger.

## Reading the result

- **Compare at equal effective step:** SGD at $\eta$ against momentum at $\eta(1-\beta)$. If they look identical, gradient noise dominates and momentum is mostly averaging.
- **Early bumps are normal:** the error spirals, so the loss can rise for a few steps (see the rates figure). A loss that keeps climbing is not a bump.
- **A slow, regular wobble** means too much momentum; lower $\beta$ or $\eta$.
- **Log the update size** $\lVert x_{k+1} - x_k\rVert$: it should be roughly $\eta/(1-\beta)$ times the typical gradient norm.

## What goes wrong

The classic accident: a model trains with plain SGD, someone switches on momentum 0.9 and keeps the learning rate, and the run blows up. Here it is on a small two-layer network on the two-moons data (batch 32, same seed for all three runs):

~~~dg-fig 14.02-switch.svg
Plain SGD at learning rate 2.0 trains (loss 0.16 after 300 steps). Adding momentum 0.9 at the same rate makes the loss explode (above 45). Dividing the rate by 10, which keeps the effective step the same, trains well (0.12).
~~~

On a fixed quadratic this would not happen: any $\eta$ below $2/L$ is also below momentum's limit $2(1+\beta)/L$. But the tenfold effective step carries the weights much further per update, into regions of higher curvature, and the velocity keeps pushing for several steps after the gradient has turned. Retune $\eta$ whenever you change $\beta$.

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Diverges right after enabling momentum | effective step $\eta/(1-\beta)$ is 10x the old step | start from $\eta(1-\beta)$ and tune upwards
| Loss rises for a few steps, then falls | normal transient of an underdamped spiral | nothing; judge after a few hundred steps
| Slow regular wobble in the loss | $\beta$ too high for the curvature | lower $\beta$ (0.99 to 0.9) or lower $\eta$
| No gain over plain SGD | tiny batches, noise dominates | larger batch, or accept that momentum mostly averages
| Behaviour changes when the LR schedule kicks in | framework stores raw gradients in the buffer | expected in PyTorch; warm up or decay smoothly
| Heavy ball cycles on a non-quadratic problem | quadratic-tuned settings outside their theory | use Nesterov, or smaller $\eta$ and $\beta$

## Going deeper

**Why $\sqrt\kappa$ is the best possible.** For any method whose iterates stay in the span of past gradients, Nesterov built quadratics (like the tridiagonal one in the Nesterov figure) on which the error after $k$ steps is at least of order $\left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^{k}$. Tuned heavy ball matches this on quadratics. Behind it: $k$ steps of such a method apply a degree-$k$ polynomial $p$ with $p(0) = 1$ to the Hessian, and the best worst case over $[\mu, L]$ is a rescaled Chebyshev polynomial, the same fact conjugate gradients exploit.

**Continuous-time view.** For small steps heavy ball approaches a damped oscillator $\ddot x + a\dot x + \nabla f(x) = 0$; Nesterov's convex method approaches $\ddot x + \frac{3}{t}\dot x + \nabla f(x) = 0$ (Su, Boyd and Candès), friction that fades over time, matching $\beta_k \to 1$.

**Stochastic gradients.** Noise is averaged over about $1/(1-\beta)$ steps, but the step is also $1/(1-\beta)$ times larger, so when noise dominates, momentum at $\eta$ behaves much like plain SGD at $\eta/(1-\beta)$ and the $\sqrt\kappa$ gain mostly disappears. Sutskever et al. (2013) found that momentum (Nesterov's form, $\beta$ ramped towards 0.99) plus good initialisation made SGD competitive for deep and recurrent networks; $\beta = 0.9$ is the common default. Adam (14.03) keeps this velocity as its "first moment".

## Key terms

| Term | Meaning
| --- | --- |
| Velocity $v$ | Running, decaying memory of recent steps that is added to the parameters
| Momentum coefficient $\beta$ | Fraction of the old velocity kept each step; 0 is plain gradient descent
| Heavy-ball method | Polyak's momentum: gradient step at the current point plus $\beta$ times the last move
| Effective learning rate | $\eta/(1-\beta)$, the steady-state step under a constant gradient
| Exponential moving average | Weighted average where each older value counts $\beta$ times less
| Underdamped | Error spirals in (complex roots, magnitude $\sqrt\beta$); can overshoot and ring
| Nesterov accelerated gradient | Momentum with the gradient taken at the look-ahead point $x_k + \beta v_k$
| Optimal rate | $O(1/k^2)$ for smooth convex, $\sqrt\kappa$ scaling for strongly convex: no gradient method does better in the worst case

<div class="take">
<div class="lab">Key takeaway</div>

momentum keeps a decaying sum of past gradients, so steady directions speed up and zig-zags cancel; tuned, it needs about $\sqrt\kappa$ steps where gradient descent needs $\kappa$. Remember that $\beta$ multiplies the effective step by $1/(1-\beta)$, retune $\eta$ when you change it, and prefer Nesterov's look-ahead when you want guarantees beyond quadratics.

</div>

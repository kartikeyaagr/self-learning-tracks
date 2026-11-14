---
id: "14.03"
course: "ml-maths-course"
module: "14"
slug: "adaptive-methods-adagrad-rmsprop-adam-adamw"
title: "Adaptive methods: AdaGrad, RMSProp, Adam, AdamW"
keyIdeas: ["Adaptive methods divide each coordinate's gradient by a running estimate of its typical size: a cheap **diagonal preconditioner** learned from gradients alone, so parameters with tiny and huge gradients both move.", "AdaGrad divides by the square root of the *sum* of all past squared gradients, which suits sparse features but makes the step shrink like $1/\\sqrt t$; RMSProp uses an exponential moving average instead, so the step stays alive.", "Adam combines momentum ($m$) with RMSProp ($v$), bias-corrects both because they start at zero, and steps $\\eta\\,\\hat m/(\\sqrt{\\hat v} + \\epsilon)$. The ratio has no units, so each coordinate moves roughly $\\eta$ per step when its gradient is consistent and less when it flips sign.", "An L2 penalty inside Adam gets divided by $\\sqrt{\\hat v}$, so weights with large gradients are barely regularised and weights with small gradients are crushed; AdamW decouples weight decay as $\\theta \\leftarrow \\theta - \\eta\\lambda\\theta$, the standard for transformers.", "Costs and caveats: two extra states per parameter (8 bytes in fp32), steps that do not shrink on their own (use a schedule), possible non-convergence on adversarial problems (AMSGrad), and an $\\epsilon$ that matters in low precision."]
question: "You fine-tune a model with `torch.optim.Adam(params, lr=1e-4, weight_decay=0.01)` and notice that the embedding rows of frequent tokens keep growing while rows of rarely seen tokens shrink steadily towards zero. Explain the mechanism, what switching to AdamW changes, and what else you would adjust."
answer: "In PyTorch, `Adam`'s `weight_decay` adds $\\lambda\\theta$ to the gradient, i.e. it is an L2 penalty that passes through the adaptive scaling. The update for each coordinate is $\\eta\\,(\\hat m + \\ldots)/\\sqrt{\\hat v}$, where the penalty enters $\\hat m$ and $\\hat v$ along with the data gradient. Frequent tokens receive large, frequent data gradients, so their $\\sqrt{\\hat v}$ is large and the penalty contribution $\\lambda\\theta/\\sqrt{\\hat v}$ is tiny: these rows are effectively unregularised and drift outward. Rare tokens receive data gradients only occasionally, so most of the time their gradient *is* the penalty. Then $\\hat m \\approx \\lambda\\theta$, and once the old data gradients fade from $\\hat v$ we get $\\sqrt{\\hat v} \\approx \\lambda|\\theta|$, so the ratio approaches $\\mathrm{sign}(\\theta)$ and Adam pulls them towards zero by roughly $\\eta$ every step, a much stronger shrinkage than $\\lambda$ intended (the lesson's simulation shows weights with small gradients reaching zero while those with large gradients keep about 90% of their size). The regularisation therefore depends on gradient statistics instead of being uniform. AdamW applies decay outside the ratio, $\\theta \\leftarrow \\theta - \\eta\\lambda\\theta$, so every row loses the same fraction per step regardless of its gradient history (with $\\eta = 10^{-4}$ and $\\lambda = 0.01$ that is a factor of $(1 - 10^{-6})$ per step, gentle). Other adjustments: retune $\\lambda$ because its meaning changed (AdamW values like 0.01 to 0.1 are typical); use parameter groups to exclude biases and normalisation gains, and often embeddings, from decay; and keep a learning-rate schedule with warmup, since Adam's step does not shrink on its own."
sources: [{"title": "Kingma and Ba: Adam: A Method for Stochastic Optimization", "url": "https://arxiv.org/abs/1412.6980", "note": "The algorithm, the bias-correction argument and default hyperparameters."}, {"title": "Loshchilov and Hutter: Decoupled Weight Decay Regularization", "url": "https://arxiv.org/abs/1711.05101", "note": "Why L2 and weight decay differ under Adam, and AdamW."}, {"title": "Duchi, Hazan and Singer: Adaptive Subgradient Methods (AdaGrad)", "url": "https://jmlr.org/papers/v12/duchi11a.html", "note": "The original per-coordinate method and its regret bounds."}, {"title": "Reddi, Kale and Kumar: On the Convergence of Adam and Beyond", "url": "https://arxiv.org/abs/1904.09237", "note": "Counterexamples to Adam's convergence and the AMSGrad fix."}]
practice: ["adam-step", "adamw-decoupled-weight-decay", "adagrad", "rmsprop", "adam-optimizer"]
tags: ["Adam", "weight decay", "preconditioning"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why one learning rate for every parameter is a poor fit when gradients differ in size by orders of magnitude.
- How AdaGrad and RMSProp give each coordinate its own step size from the history of its squared gradients.
- How Adam combines that with momentum, why it needs bias correction, and why its learning rate behaves like a step length.
- Why an L2 penalty inside Adam is not weight decay, and what AdamW does instead.
- The practical costs and failure modes: optimizer memory, $\epsilon$, non-convergence, and steps that never shrink.

*Before you start:* you need gradient descent and the condition number $\kappa$ from 14.01, and momentum as an exponential moving average from 14.02. Element-wise operations on vectors (squaring, dividing, square roots) are used throughout.

Lesson 14.02 added memory to gradient descent so that it builds up speed along a valley floor. It still uses one learning rate $\eta$ for every parameter. This lesson lets each parameter choose its own.

## The idea in one picture

Take the stretched bowl from 14.01, with one steep and one flat direction:

$$f(w_1, w_2) = \tfrac12\,(20\,w_1^2 + 0.2\,w_2^2).$$

The **curvature** (how fast the slope changes) is 20 along $w_1$ and 0.2 along $w_2$, so $\kappa = 100$. Start at $(-0.8, 1)$. The gradient is $(20 w_1, 0.2 w_2) = (-16, 0.2)$: the two coordinates disagree in size by a factor of 80.

| | Gradient | Plain GD step ($\eta = 0.09$) | Adam's first step ($\eta = 0.1$)
| --- | --- | --- | --- |
| $w_1$ (steep) | $-16$ | $+1.44$ | $+0.1$
| $w_2$ (flat) | $0.2$ | $-0.018$ | $-0.1$

Gradient descent cannot use a bigger $\eta$ (above $2/20 = 0.1$ it diverges along $w_1$), so it overshoots across the steep direction while barely moving the flat one. Adam moves *both* coordinates by about $0.1$, whatever the size of their gradients.

~~~dg-fig 14.03-ravine.svg
Left: gradient descent (orange) bounces across the steep axis and creeps down the flat one; the adaptive methods head almost straight for the minimum. Right: the cost of that, in loss per step. Note RMSProp (gold) reaching 1e-10 and then bouncing back to a plateau near 0.025.
~~~

On this problem gradient descent needs 191 steps to push the loss below $10^{-4}$; Adam needs 71 and RMSProp 11. The adaptive methods all share one trick: divide each coordinate's gradient by a running estimate of that coordinate's typical gradient size. That turns a step "proportional to the gradient" into a step "of roughly fixed length in the gradient's direction".

In the language of 14.01 this is **preconditioning**, stepping with $P^{-1}\nabla f$. Adaptive methods build a **diagonal** $P$, one number per parameter, from gradients alone, at $O(d)$ cost for $d$ parameters. Below, every vector operation is element-wise and $g_t$ is the (mini-batch) gradient at step $t$.

## Step 1: AdaGrad, divide by the history of the gradient

**AdaGrad** (Duchi, Hazan and Singer, 2011) keeps, for every parameter, the running *sum* of its squared gradients:

$$G_t = G_{t-1} + g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon}\,g_t.$$

- $\theta_t$ is the parameter vector at step $t$; $\eta$ is the base learning rate.
- $G_t$ starts at zero and only grows; $\sqrt{G_t}$ measures how much gradient this coordinate has seen so far.
- $\epsilon$ (say $10^{-8}$) only prevents division by zero.

The quantity $\eta/\sqrt{G_t}$ is the **effective learning rate** of that coordinate. A coordinate whose gradients are large gets a small effective rate; one that is rarely active keeps a large one. That is exactly what you want for **sparse features**: a word that appears once in ten thousand documents should learn a lot from each appearance.

Number check: a dense coordinate that sees $g = 1$ every step has $G_t = t$, so its effective rate is $\eta/\sqrt t$: after 100 steps it is one tenth of $\eta$, after 10,000 steps one hundredth.

~~~dg-fig 14.03-effective-lr.svg
Effective learning rate on a stream of random gradients (lr = 0.01). AdaGrad's rate for a dense coordinate follows the lr/√t line downwards forever; a coordinate that is active only 2% of the time keeps about ten times more. RMSProp's rate stops falling and levels off.
~~~

That $1/\sqrt t$ decay is AdaGrad's flaw. It is ideal for convex problems, where the step *should* shrink as you approach the answer, but a deep network may need thousands of steps of real movement, and by then AdaGrad has throttled itself. In the figure, after 2,000 steps the dense coordinate's rate has fallen to 0.00022, about 45 times smaller than the base rate $\eta = 0.01$.

## Step 2: RMSProp, forget old gradients

**RMSProp** (proposed in Hinton's 2012 lecture notes, never formally published) keeps the idea but replaces the sum by an **exponential moving average** (EMA), the same kind of fading memory as the momentum buffer in 14.02:

$$v_t = \rho\, v_{t-1} + (1-\rho)\, g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon}\,g_t.$$

- $v_t$ estimates the recent mean of $g^2$, so $\sqrt{v_t}$ is the **root mean square** (RMS) of recent gradients, hence the name.
- $\rho$ (typically 0.9 to 0.99) sets the memory: roughly the last $1/(1-\rho)$ steps count, 100 steps for $\rho = 0.99$.

Because old gradients fade, $v_t$ tracks the current scale instead of the lifetime total, and the step size stops shrinking (gold curve in the figure, which settles near $\eta = 0.01$). If gradients are steady at size $s$, then $v_t \to s^2$ and the step is about $\eta\, g/s \approx \eta$: one learning rate's worth of movement per step, in every coordinate.

## Step 3: Adam, momentum plus RMSProp

**Adam** (Kingma and Ba, 2014) runs two EMAs side by side. The first averages the gradient itself (momentum), the second averages its square (RMSProp):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2.$$

- $m_t$ is the **first moment**, a smoothed gradient direction; $\beta_1$ is usually 0.9 (memory of about 10 steps).
- $v_t$ is the **second moment**, a smoothed squared size; $\beta_2$ is usually 0.999 (memory of about 1,000 steps).

Both start at zero, so early on they are too small. Adam divides each by the fraction of the average that has been "filled in" so far, then takes the step:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and $\eta = 10^{-3}$ as a common starting point.

~~~dg-flow Adam's data flow for one parameter tensor: two moving averages, bias correction, then a scale-free ratio.
Gradient g -(EMA beta1)-> m -(divide by 1 - beta1^t)-> m hat -(ratio)-> *Step = lr x m hat / (sqrt(v hat) + eps)
Gradient g -(square, EMA beta2)-> v -(divide by 1 - beta2^t)-> v hat
~~~

## Step 4: why the bias correction is there

Suppose the gradient is a constant $g$. Unrolling the EMA gives $m_t = (1-\beta_1^t)\,g$: after one step only 10% of $g$, after 10 steps 65%. Dividing by $1 - \beta_1^t$ recovers $g$ exactly, and the same holds for $v_t$ and $g^2$. That is the whole correction.

It matters most for $v$. With $\beta_2 = 0.999$, the raw $v$ is still only 18% filled after 200 steps. Since $v$ sits in the denominator, an underestimate makes steps *too big*, just when the optimizer knows least about the landscape.

~~~dg-fig 14.03-bias.svg
Constant gradient g = 1. Left: the raw averages start at zero; m catches up in about 30 steps, v is still far below 1 after 200. Right: without correction the step peaks at 6.6 times the learning rate around step 12; with correction it is exactly one learning rate from the first step.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**Adam's first two steps by hand**

Two parameters with very different gradients. Step 1 sees $g_1 = (4,\ 0.01)$, step 2 sees $g_2 = (2,\ 0.03)$. Use $\eta = 0.1$ and the default betas; ignore $\epsilon$.

Step 1, first moment: $m_1 = 0.9\cdot 0 + 0.1\cdot(4, 0.01) = (0.4,\ 0.001)$. Correction factor $1 - 0.9^1 = 0.1$, so $\hat m_1 = (4,\ 0.01)$.

Step 1, second moment: $v_1 = 0.001\cdot(16,\ 0.0001) = (0.016,\ 10^{-7})$. Correction factor $1 - 0.999 = 0.001$, so $\hat v_1 = (16,\ 0.0001)$ and $\sqrt{\hat v_1} = (4,\ 0.01)$.

Step 1, update: $0.1\cdot(4/4,\ 0.01/0.01) = (0.1,\ 0.1)$. Both parameters move by exactly $\eta$, although their gradients differ 400-fold. Plain SGD with the same $\eta$ would move them by $0.4$ and $0.001$.

Without correction the update would be $0.1\cdot 0.4/\sqrt{0.016} = 0.1 \cdot 0.4/0.1265 = 0.316$, more than three times larger.

Step 2, first moment: $m_2 = 0.9\cdot(0.4, 0.001) + 0.1\cdot(2, 0.03) = (0.56,\ 0.0039)$. Factor $1 - 0.81 = 0.19$, so $\hat m_2 = (2.947,\ 0.0205)$.

Step 2, second moment: $v_2 = 0.999\cdot(0.016, 10^{-7}) + 0.001\cdot(4,\ 0.0009) = (0.019984,\ 9.999\times10^{-7})$. Factor $1 - 0.999^2 = 0.001999$, so $\hat v_2 = (9.997,\ 0.000500)$ and $\sqrt{\hat v_2} = (3.162,\ 0.0224)$.

Step 2, update: $0.1\cdot(2.947/3.162,\ 0.0205/0.0224) = (0.0932,\ 0.0918)$. The first gradient halved and the second tripled, yet both steps stay just under $\eta$: Adam reacts to the *consistency* of a gradient, not its size.

</div>

That example shows Adam's key property. The ratio $\hat m/\sqrt{\hat v}$ has no units: multiply every gradient of a coordinate by 1,000 and it does not change. When a gradient keeps the same sign, the ratio is close to $\pm 1$ and the coordinate moves about $\eta$ per step. When the sign keeps flipping (noise, or bouncing across a valley), $\hat m$ averages towards zero while $\sqrt{\hat v}$ does not, so the step shrinks automatically. The learning rate has become a **step length in parameter units**, which is why one $\eta$ works across layers whose raw gradients differ by orders of magnitude.

## Step 5: AdamW, weight decay done right

**Weight decay** shrinks every weight a little each step, $\theta \leftarrow \theta(1 - \eta\lambda)$, to keep weights small. With plain SGD it is the same as adding an **L2 penalty** $\tfrac\lambda2\lVert\theta\rVert^2$ to the loss, because that penalty's gradient $\lambda\theta$ gets multiplied by the same $\eta$.

With Adam the two differ. The penalty gradient $\lambda\theta$ is added to $g_t$ *before* the division by $\sqrt{\hat v_t}$. So a weight whose data gradients are large has a large $\sqrt{\hat v}$ and its penalty is divided down to almost nothing; a weight with tiny data gradients has its penalty amplified. Regularisation strength ends up depending on gradient statistics.

**AdamW** (Loshchilov and Hutter, published at ICLR 2019) applies the decay outside the adaptive ratio:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_t\right).$$

Now every weight shrinks by the same fraction $\eta\lambda$ per step, whatever its gradient history.

~~~dg-fig 14.03-adamw.svg
Simulation: weights start at 1 and receive pure-noise data gradients of different sizes for 1000 steps (lr 0.01, λ = 0.1, averaged over 400 runs). With L2 inside Adam, weights with small gradients are driven to 0 while weights with gradient size 10 keep 90% of their value. AdamW shrinks all of them to the predicted 0.37.
~~~

AdamW is the standard optimizer for transformers. In PyTorch, `torch.optim.AdamW` defaults to `weight_decay=0.01`; large-language-model recipes commonly use around 0.1. Biases and normalisation gains are usually excluded from decay, since shrinking them has no regularising benefit. The AdamW authors also found the best $\lambda$ much less tied to the best $\eta$. Lesson 14.06 returns to weight decay as regularisation.

## Adaptive methods in code

A from-scratch Adam is ten lines. Setting `wd > 0` turns it into AdamW. Running it on the stretched bowl and comparing with PyTorch gives identical numbers:

```python
import numpy as np, torch

def adam(grad, x, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, steps=100):
    m = np.zeros_like(x); v = np.zeros_like(x)
    for t in range(1, steps + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g              # running mean of gradients
        v = b2 * v + (1 - b2) * g * g          # running mean of squared gradients
        m_hat = m / (1 - b1 ** t)              # bias correction
        v_hat = v / (1 - b2 ** t)
        x = x - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * x)   # wd > 0 gives AdamW
    return x

A = np.array([20.0, 0.2])                      # curvatures of the stretched bowl
x = adam(lambda x: A * x, np.array([-0.8, 1.0]))
print(x, 0.5 * np.sum(A * x**2))               # [0.00191 0.00294] 3.75e-05

p = torch.tensor([-0.8, 1.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([p], lr=0.1)
for _ in range(100):
    opt.zero_grad(); (0.5 * (torch.tensor(A) * p**2).sum()).backward(); opt.step()
print(p.detach().numpy())                      # [0.00191 0.00294]  (same)

# AdamW with the usual split: decay weight matrices, not biases or norm gains
net = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LayerNorm(64), torch.nn.Linear(64, 10))
decay    = [q for q in net.parameters() if q.ndim >= 2]
no_decay = [q for q in net.parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                         {'params': no_decay, 'weight_decay': 0.0}], lr=1e-3)
print(sum(q.numel() for q in decay), sum(q.numel() for q in no_decay))   # 4736 202
```

The `ndim >= 2` rule is a common heuristic: weight matrices are 2-D, biases and LayerNorm scales 1-D.

## Reading the result: what adaptive methods buy in practice

To see the methods on a real (small) network, we trained a two-hidden-layer MLP (64 units each) on scikit-learn's 8x8 digits, batch size 64, 30 epochs, with each optimizer at nine learning rates from $10^{-4}$ to 1.

~~~dg-fig 14.03-digits.svg
Left: training loss after 30 epochs against learning rate. The adaptive methods (RMSProp, Adam) work at rates 10 to 100 times smaller than SGD; at 1e-4 they already make progress while SGD barely moves. Right: training curves at each method's best rate; all reach about 97% test accuracy.
~~~

How to read it:

- **The usable range moves.** SGD only gets the loss below 0.1 at $\eta$ = 0.1 to 0.3; Adam and RMSProp do so from $10^{-3}$ to $3\times10^{-2}$. Because Adam's $\eta$ is a step length, its good range transfers between models, so $10^{-3}$ is a reasonable default.
- **Tuned, they end up close.** On this small, well-scaled network, SGD with momentum at its best rate (training loss 0.0011) is as good as Adam at its best (0.0008), and test accuracy is about 97% for every method. Adaptive methods shine when gradient scales differ wildly between parameter groups, as in transformers.
- **Constant-rate curves are spiky.** A decaying schedule (14.04) turns the bouncing into convergence.

In real training, also log the gradient norm and the per-layer update-to-weight ratio $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ (around $10^{-3}$ is a commonly quoted healthy value).

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Training slows to a crawl after a while with AdaGrad | $G_t$ keeps growing, effective rate falls like $1/\sqrt t$ | Use RMSProp or Adam; or restart AdaGrad's accumulator
| Loss plateaus and jitters, never reaches the minimum | Step is about $\eta$ no matter how small the gradient (RMSProp in the first figure) | Decay the learning rate (14.04); Adam's long $v$ memory helps but is no guarantee
| Loss spikes in the first few hundred steps | Too large $\eta$ while $\hat v$ is still a poor estimate | Learning-rate warmup (14.04); check bias correction is on
| Rare parameters shrink to zero, frequent ones grow | L2 penalty passed to `Adam(weight_decay=...)`, which is L2, not decoupled decay | Switch to `AdamW`; exclude biases and norm gains
| Sudden instability in fp16/bf16 | $\epsilon$ too small relative to the precision, or $\sqrt{\hat v}$ underflows | Raise $\epsilon$ (e.g. $10^{-6}$), keep optimizer state in fp32
| Out of memory although the model fits | $m$ and $v$ double the stored state per parameter | Sharded optimizer state, 8-bit Adam, Adafactor
| Loss explodes long into training | Rare large gradient after $v$ has decayed to small values | Gradient clipping, lower $\beta_2$ (e.g. 0.95), AMSGrad

**Memory in numbers**: Adam stores $m$ and $v$ for every parameter. For a 7-billion-parameter model in fp32 that is $2 \times 7\times10^9 \times 4$ bytes = 56 GB, four times the 14 GB of bf16 weights. **$\epsilon$ is not cosmetic**: where $\sqrt{\hat v}$ falls to near $\epsilon$, the update becomes $\eta g/\epsilon$, effectively SGD with a huge learning rate.

## Going deeper

**How large can a step get?** Kingma and Ba show that, ignoring $\epsilon$, the per-coordinate step is bounded by $\eta\,(1-\beta_1)/\sqrt{1-\beta_2}$ when $1-\beta_1 > \sqrt{1-\beta_2}$, and by $\eta$ otherwise. With the defaults, $0.1 > 0.0316$, so the worst case is about $3.2\,\eta$, reached only when one large gradient arrives after a long quiet period. In the common case the step is at most about $\eta$, which is what makes $\eta$ a trust-region radius in parameter units.

**Sign descent as a limit.** With $\beta_1 = \beta_2 = 0$ (and $\epsilon = 0$) Adam's step is $\eta\, g/|g| = \eta\,\mathrm{sign}(g)$. Adam is therefore a smoothed version of **sign descent**, which explains its bouncing at a fixed $\eta$: a sign step never shrinks near the minimum. It ends up oscillating within about $\eta$ of the optimum: the RMSProp plateau in the first figure.

**Scale invariance and its limit.** Rescaling a coordinate's gradients by any constant $c > 0$ leaves $\hat m/\sqrt{\hat v}$ unchanged. This is a diagonal preconditioner that adapts to scale, not to curvature: it cannot undo a stretched valley that is *rotated* relative to the axes, because a diagonal $P$ can only stretch along the coordinate axes. Full-matrix AdaGrad and Shampoo-style methods (14.05 covers second-order ideas) use non-diagonal preconditioners for that, at higher cost.

**Convergence theory.** AdaGrad has an $O(\sqrt T)$ regret bound for online convex problems, which is optimal, with constants that improve when gradients are sparse. Adam does not have an unconditional guarantee: Reddi, Kale and Kumar (ICLR 2018) built simple convex problems where it converges to the wrong point, because a large but rare informative gradient is forgotten by $v$ before it can act. Their **AMSGrad** fix uses $\max(\hat v_{t-1}, v_t)$ so the denominator never shrinks. In practice a decaying $\eta$ usually suffices.

**Generalisation.** Wilson et al. (2017) found well-tuned SGD with momentum generalising better than adaptive methods on several image benchmarks; for transformers, adaptive methods are clearly better.

| Method | State per parameter | Step scale | Typical use
| --- | --- | --- | --- |
| SGD + momentum | 1 (velocity) | proportional to gradient | CNNs, well-conditioned problems
| AdaGrad | 1 (sum of squares) | shrinks like $1/\sqrt t$ | sparse convex problems
| RMSProp | 1 (EMA of squares) | about $\eta$ | RL, older RNN recipes
| Adam / AdamW | 2 ($m$ and $v$) | about $\eta$, bias-corrected | transformers, most deep learning

## Key terms

| Term | Meaning
| --- | --- |
| Adaptive method | Optimizer that gives each parameter its own step size from its gradient history
| Diagonal preconditioner | One scaling factor per coordinate applied to the gradient, $P^{-1}g$ with diagonal $P$
| Effective learning rate | The actual multiplier on a coordinate's gradient, e.g. $\eta/\sqrt{G_t}$
| Exponential moving average | Running average $a_t = \beta a_{t-1} + (1-\beta)x_t$ that weights recent values most
| First / second moment | Adam's averages of $g$ ($m$) and of $g^2$ ($v$)
| Bias correction | Dividing $m_t$, $v_t$ by $1-\beta^t$ to undo their start at zero
| Decoupled weight decay | Shrinking weights by $\eta\lambda\theta$ outside the adaptive ratio (AdamW)
| $\epsilon$ | Small constant in the denominator; sets how Adam treats near-zero gradient scales

<div class="take">
<div class="lab">Key takeaway</div>

adaptive methods divide each coordinate's gradient by a running RMS of its own gradients, turning the learning rate into a per-coordinate step length; Adam adds momentum and bias correction, AdamW keeps weight decay outside that division, and both still need a learning-rate schedule and two extra fp32 numbers per parameter.

</div>

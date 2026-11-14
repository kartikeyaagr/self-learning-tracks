---
id: "14.03"
course: "ml-maths-course"
module: "14"
slug: "adaptive-methods-adagrad-rmsprop-adam-adamw"
title: "Adaptive methods: AdaGrad, RMSProp, Adam, AdamW"
keyIdeas: ["Adaptive methods divide each coordinate's step by a running estimate of its gradient scale, a diagonal preconditioner learned from gradients alone.", "AdaGrad sums squared gradients forever, which suits sparse features but shrinks the step towards zero; RMSProp uses an exponential average instead, so the step stays alive.", "Adam combines momentum ($m$) with RMSProp ($v$), bias-corrects both, and steps $\\eta\\,\\hat m/(\\sqrt{\\hat v} + \\epsilon)$; each step is roughly bounded by $\\eta$ per coordinate.", "L2 regularisation inside Adam is scaled down for parameters with large gradients; AdamW decouples weight decay as $\\theta \\leftarrow \\theta(1 - \\eta\\lambda)$, which is what modern training uses.", "Costs and caveats: two extra states per parameter (8 bytes each in fp32), possible non-convergence on adversarial problems, and $\\epsilon$ and $\\beta_2$ matter more than their defaults suggest."]
question: "You fine-tune a model with Adam plus an L2 penalty and notice that the large embedding rows for frequent tokens keep growing while rarely seen rows shrink towards zero. Explain the mechanism, and how switching to AdamW changes it."
answer: "With L2 inside Adam, the penalty gradient $\\lambda\\theta$ is added to the data gradient and then divided by $\\sqrt{\\hat v}$. Frequent tokens have large, frequent gradients, so their $\\hat v$ is large and the penalty term is scaled down to almost nothing: their rows are effectively unregularised and grow. Rare tokens get few data gradients, so $\\hat v$ is small and the same penalty is amplified, pulling those rows strongly to zero on every step even though there is little data to oppose it. The regularisation strength therefore depends on gradient statistics rather than being uniform. AdamW applies decay outside the adaptive scaling, $\\theta \\leftarrow \\theta - \\eta\\lambda\\theta$, so every row shrinks by the same fraction per step regardless of its gradient history. You would then retune $\\lambda$, often excluding embeddings, biases and normalisation parameters from decay."
sources: [{"title": "Kingma and Ba: Adam: A Method for Stochastic Optimization", "url": "https://arxiv.org/abs/1412.6980", "note": "The algorithm, the bias-correction argument and default hyperparameters."}, {"title": "Loshchilov and Hutter: Decoupled Weight Decay Regularization", "url": "https://arxiv.org/abs/1711.05101", "note": "Why L2 and weight decay differ under Adam, and AdamW."}, {"title": "Duchi, Hazan and Singer: Adaptive Subgradient Methods (AdaGrad)", "url": "https://jmlr.org/papers/v12/duchi11a.html", "note": "The original per-coordinate method and its regret bounds."}, {"title": "Reddi, Kale and Kumar: On the Convergence of Adam and Beyond", "url": "https://arxiv.org/abs/1904.09237", "note": "Counterexamples to Adam's convergence and the AMSGrad fix."}]
practice: ["adam-step", "adamw-decoupled-weight-decay", "adagrad", "rmsprop", "adam-optimizer"]
tags: ["Adam", "weight decay", "preconditioning"]
tools: ["NumPy", "PyTorch"]
---

## Per-coordinate step sizes

Momentum (14.02) still uses one learning rate for every parameter. But the parameters of a network see gradients of very different sizes: embedding rows for rare tokens get a gradient once in a thousand steps, the final layer bias gets one every step, and layers differ by orders of magnitude in typical gradient norm. Lesson 14.01 showed that the ideal fix is preconditioning, stepping with $P^{-1}\nabla f$. Adaptive methods build a **diagonal** $P$ from the history of squared gradients, one scale per coordinate, at $O(d)$ cost. The AI Research course walks through Adam's code; this lesson focuses on why each piece exists. Below, all operations on vectors are element-wise, and $g_t$ is the mini-batch gradient at step $t$.

## AdaGrad and RMSProp

**AdaGrad** (Duchi, Hazan and Singer, 2011) accumulates squared gradients:

$$G_t = G_{t-1} + g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon}\,g_t.$$

Rarely active coordinates have small $G$ and so keep large steps: good for sparse features in linear models and for word embeddings. The flaw is that $G$ only grows, so for dense gradients the effective rate decays like $1/\sqrt t$ and training can stall long before the optimum of a non-convex network.

**RMSProp** (Hinton's lecture notes, 2012) replaces the sum with an exponential moving average, so old gradients are forgotten:

$$v_t = \rho v_{t-1} + (1-\rho) g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon}\,g_t,$$

with $\rho$ around 0.9 to 0.99. The step now tracks the recent gradient scale and does not die out.

## Adam

Adam (Kingma and Ba, 2014) adds a momentum-style average of the gradient itself:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Why the bias correction: both averages start at zero, so early on they underestimate. After $t$ steps of a constant gradient, $m_t = (1 - \beta_1^t)g$; dividing by $1 - \beta_1^t$ recovers $g$ exactly. Without the correction, $v$ (with $\beta_2 = 0.999$) is tiny for the first thousand steps and the update would be enormous.

<div class="ex">
<div class="lab">Worked example</div>

**Adam's very first step**

At $t = 1$: $m_1 = 0.1\,g$ and $v_1 = 0.001\,g^2$. Bias correction gives $\hat m_1 = g$ and $\hat v_1 = g^2$.

The update is $\eta\,g/(|g| + \epsilon) \approx \eta\cdot\mathrm{sign}(g)$: every coordinate moves by about $\eta$, whatever its gradient scale.

Without correction it would be $\eta\cdot0.1g/\sqrt{0.001g^2} \approx 3.2\eta$ per coordinate, over three times larger, on the step where the curvature is least known.

</div>

That example shows Adam's key property: the ratio $\hat m/\sqrt{\hat v}$ is scale-free and, when gradients are consistent, close to $\pm1$, so each coordinate moves at most about $\eta$ per step. The learning rate becomes a step length in parameter units, which makes it far easier to transfer between layers and models than an SGD learning rate. When the gradient's sign keeps flipping, $\hat m$ shrinks relative to $\sqrt{\hat v}$ and the step drops automatically.

~~~dg-flow Adam's data flow for one parameter tensor: two moving averages, bias correction, then a scale-free ratio.
Gradient g -(EMA beta1)-> m -(divide by 1 - beta1^t)-> m hat -(ratio)-> *Step = lr x m hat / (sqrt(v hat) + eps)
Gradient g -(square, EMA beta2)-> v -(divide by 1 - beta2^t)-> v hat
~~~

## AdamW: decoupled weight decay

With plain SGD, adding $\frac\lambda2\lVert\theta\rVert^2$ to the loss and multiplying weights by $(1 - \eta\lambda)$ each step are the same thing. With Adam they are not. The L2 gradient $\lambda\theta$ goes through the adaptive scaling, so it is divided by $\sqrt{\hat v}$: parameters with large gradient history are barely regularised, those with small gradients are regularised heavily. Loshchilov and Hutter's **AdamW** applies the decay outside the adaptive step:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\theta_t\right).$$

This gives every weight the same relative shrinkage and makes the best $\lambda$ largely independent of $\eta$. AdamW is the default for transformers. PyTorch's `AdamW` uses a default `weight_decay` of 0.01 (typical LLM recipes use 0.1), and biases and normalisation gains are usually excluded from decay. Lesson 14.06 returns to weight decay as regularisation.

## Caveats

- **Memory.** Adam keeps $m$ and $v$ for every parameter: for a 7-billion-parameter model in fp32 that is $2\times7\times10^9\times4$ bytes = 56 GB of optimizer state, more than the bf16 weights themselves. This drives sharded optimizers and memory-light variants (Adafactor, 8-bit Adam).
- **$\epsilon$ is not cosmetic.** Where $\sqrt{\hat v}$ is near $\epsilon$, Adam degrades towards SGD with step $\eta/\epsilon$; in low precision a too-small $\epsilon$ can cause instabilities.
- **Convergence.** Reddi et al. built convex problems where Adam fails to converge, because $v$ can forget large rare gradients; AMSGrad keeps a running maximum of $v$. In practice a decaying learning rate usually suffices.
- **Generalisation.** Wilson et al. found tuned SGD with momentum generalising better than adaptive methods on some vision benchmarks; for transformers, adaptive methods are clearly better, partly because gradient scales differ so much between parameter types.

| Method | State per parameter | Step scale | Typical use
| --- | --- | --- | --- |
| SGD + momentum | 1 (velocity) | proportional to gradient | CNNs, well-conditioned problems
| AdaGrad | 1 (sum of squares) | shrinks like $1/\sqrt t$ | sparse convex problems
| RMSProp | 1 (EMA of squares) | about $\eta$ | RL, older RNN recipes
| Adam / AdamW | 2 (m and v) | about $\eta$, bias-corrected | transformers, most deep learning

<div class="take">
<div class="lab">Key takeaway</div>

adaptive methods are cheap diagonal preconditioners that make the learning rate a per-coordinate step length; use AdamW (decoupled weight decay) rather than Adam with L2, and budget two fp32 states per parameter.

</div>

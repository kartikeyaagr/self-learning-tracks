---
id: "14.02"
course: "ml-maths-course"
module: "14"
slug: "momentum-and-nesterov-acceleration"
title: "Momentum and Nesterov acceleration"
keyIdeas: ["Heavy-ball momentum keeps a velocity, $v_{k+1} = \\beta v_k - \\eta\\nabla f(x_k)$, $x_{k+1} = x_k + v_{k+1}$: consistent gradients accumulate, oscillating ones cancel.", "On a quadratic, tuned momentum contracts at $(\\sqrt\\kappa - 1)/(\\sqrt\\kappa + 1)$, so iterations scale with $\\sqrt\\kappa$ instead of $\\kappa$: 10x fewer steps at $\\kappa = 100$.", "In steady state the effective step is $\\eta/(1-\\beta)$; $\\beta = 0.9$ multiplies the step by 10 and averages roughly the last 10 gradients.", "Nesterov evaluates the gradient at the look-ahead point $x_k + \\beta v_k$, which corrects overshoot earlier and gives the optimal $O(1/k^2)$ rate for smooth convex problems.", "Frameworks write momentum differently (PyTorch: $b \\leftarrow \\mu b + g$, $p \\leftarrow p - \\eta b$); the forms are equivalent up to a rescaling of the learning rate."]
question: "A model trains well with plain SGD at learning rate 0.1. A colleague switches on momentum 0.9 with the same learning rate and the loss immediately diverges. Explain why, and what learning rate would be a sensible first try with momentum."
answer: "With momentum the steady-state step is $\\eta/(1-\\beta)$, so at $\\beta = 0.9$ the same $\\eta = 0.1$ behaves like a plain step of about 1.0, ten times larger. If 0.1 was near the stability edge for the sharpest directions, ten times that is far past it and the iterates blow up. The larger stability region of heavy ball ($\\eta < 2(1+\\beta)/L$) only buys a factor of about 1.9, not 10. A sensible first try keeps the effective step similar: $\\eta \\approx 0.1\\cdot(1 - 0.9) = 0.01$, then tune upward. The same logic says that changing beta in any recipe should come with a learning-rate retune, since beta and eta together set the effective step."
sources: [{"title": "Goh: Why Momentum Really Works (Distill)", "url": "https://distill.pub/2017/momentum/", "note": "Interactive derivation of the quadratic analysis, the sqrt(kappa) rate and the oscillation regimes."}, {"title": "Sutskever et al.: On the importance of initialization and momentum in deep learning", "url": "https://proceedings.mlr.press/v28/sutskever13.html", "note": "Classical versus Nesterov momentum in deep networks, and momentum schedules."}, {"title": "PyTorch: torch.optim.SGD", "url": "https://pytorch.org/docs/stable/generated/torch.optim.SGD.html", "note": "The exact momentum, dampening and Nesterov update the framework implements."}]
practice: ["classical-momentum", "nesterov-momentum", "custom-sgd-with-momentum", "optimizer-showdown"]
tags: ["momentum", "acceleration"]
tools: ["NumPy", "PyTorch"]
---

## The problem momentum solves

Lesson 14.01 showed that gradient descent on a stretched bowl zig-zags across the sharp direction and crawls along the flat one, needing about $\kappa$ steps per digit of accuracy. The gradient alone cannot fix this: it points mostly across the valley. Momentum adds memory. Gradient components that keep the same sign from step to step (along the valley floor) build up speed; components that flip sign (across the valley) cancel.

## Heavy-ball dynamics

Polyak's heavy-ball method keeps a velocity:

$$v_{k+1} = \beta v_k - \eta\nabla f(x_k), \qquad x_{k+1} = x_k + v_{k+1}.$$

Equivalently, $x_{k+1} = x_k - \eta\nabla f(x_k) + \beta(x_k - x_{k-1})$: a gradient step plus a fraction $\beta$ of the previous displacement. The physical picture is a ball with mass rolling on the loss surface with friction $1-\beta$.

Two quick consequences:

- If the gradient is constant, $g$, the velocity converges to $-\eta g/(1-\beta)$. The **effective learning rate is $\eta/(1-\beta)$**: 10x at $\beta = 0.9$, 100x at $\beta = 0.99$. Changing $\beta$ without adjusting $\eta$ changes the step size.
- The velocity is an exponential moving average of past gradients with a horizon of about $1/(1-\beta)$ steps, which also averages out mini-batch noise.

~~~dg-flow Momentum filters the gradient sequence: persistent directions are amplified, alternating ones cancel.
Gradient g_k -(scale by eta)-> Velocity update -(add beta times v_k)-> *Velocity v_k+1 -(add to x)-> New iterate
~~~

## Why it is faster: the quadratic analysis

On a quadratic, each eigendirection with curvature $\lambda$ now follows a two-term linear recurrence, $e_{k+1} = (1 + \beta - \eta\lambda)e_k - \beta e_{k-1}$. Its behaviour is set by the roots of $z^2 - (1 + \beta - \eta\lambda)z + \beta = 0$. When the roots are complex, both have magnitude exactly $\sqrt\beta$, regardless of $\lambda$. Choosing

$$\eta = \frac{4}{(\sqrt L + \sqrt\mu)^2}, \qquad \beta = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$$

puts every direction, from $\mu$ to $L$, in that regime, giving a contraction of

$$\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \approx 1 - \frac{2}{\sqrt\kappa}\ \text{per step.}$$

The iteration count drops from order $\kappa$ to order $\sqrt\kappa$. The price is oscillation: the error spirals rather than decaying monotonically, and the loss can rise temporarily.

| Method | Rate per step (quadratic) | Steps for 1e-6 at kappa = 100 | Steps for 1e-6 at kappa = 10,000
| --- | --- | --- | --- |
| Gradient descent, best fixed step | $(\kappa-1)/(\kappa+1)$ | about 690 | about 69,000
| Heavy ball, tuned | $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ | about 69 | about 690

<div class="ex">
<div class="lab">Worked example</div>

**the bowl from lesson 14.01, $\kappa = 100$**

Gradient descent with the best fixed step contracts by $99/101 = 0.980$ per step: $\ln(10^{-6})/\ln(0.980) \approx 690$ steps to reduce the error a millionfold.

Tuned momentum uses $\sqrt\kappa = 10$: $\beta = (9/11)^2 = 0.669$, $\eta = 4/(10 + 1)^2 = 0.033$, and contracts by $9/11 = 0.818$ per step: about 69 steps.

Note that $\eta = 0.033$ exceeds the plain-GD stability limit of $2/L = 0.02$. Momentum's stability region is larger: $\eta < 2(1+\beta)/L$.

</div>

## Nesterov's accelerated gradient

Nesterov's method computes the gradient where momentum is about to take you, not where you are:

$$v_{k+1} = \beta v_k - \eta\nabla f(x_k + \beta v_k), \qquad x_{k+1} = x_k + v_{k+1}.$$

If the velocity is carrying the iterate up the far wall of the valley, the look-ahead gradient already points back and brakes earlier. For general smooth convex functions (not just quadratics), with an increasing schedule $\beta_k = (k-1)/(k+2)$, Nesterov proved $f(x_k) - f^\star \le 2L\lVert x_0 - x^\star\rVert^2/(k+1)^2$: $O(1/k^2)$ against gradient descent's $O(1/k)$, and no first-order method can do better in the worst case. Heavy ball has no such guarantee outside quadratics and can even fail to converge on some smooth convex functions.

## Momentum in deep learning frameworks

PyTorch's `torch.optim.SGD` stores a buffer $b \leftarrow \mu b + g$ and steps $p \leftarrow p - \eta b$; with `nesterov=True` it steps with $g + \mu b$ instead, a reparameterised Nesterov that only needs the gradient at the current point. Multiply out and this is heavy ball with the learning rate folded inside the buffer; the two are equivalent for a constant $\eta$ but differ when a schedule changes $\eta$ mid-run.

```
def sgd_momentum(params, grads, bufs, lr=0.1, mu=0.9, nesterov=False):
    for p, g, b in zip(params, grads, bufs):
        b *= mu; b += g                       # b = mu * b + g
        p -= lr * (g + mu * b if nesterov else b)
```

Sutskever et al. (2013) showed that momentum with a well-chosen $\beta$ (often ramped up to 0.99) is what made SGD competitive for training deep and recurrent networks. In practice $\beta = 0.9$ is the near-universal default, and with mini-batch noise the benefit is partly acceleration and partly averaging; when the gradient noise dominates, the theoretical $\sqrt\kappa$ speed-up shrinks. The Distill article in the sources animates all of this.

<div class="take">
<div class="lab">Key takeaway</div>

momentum turns gradient descent into a damped oscillator whose tuned rate depends on $\sqrt\kappa$ instead of $\kappa$; remember that it multiplies the effective step by $1/(1-\beta)$, and prefer Nesterov's look-ahead form when you want guarantees beyond quadratics.

</div>

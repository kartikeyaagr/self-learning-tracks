---
id: "09.55"
course: "ml-maths-course"
module: "09"
slug: "momentum-and-adaptive-methods"
title: "Momentum and Adaptive Methods"
keyIdeas: ["Momentum keeps a running velocity: ⟦3161⟧, then ⟦3162⟧.", "It damps zig-zags across a narrow valley and speeds up travel along it.", "Adam divides each coordinate's step by a running estimate of that coordinate's gradient size, so badly scaled coordinates move at similar rates."]
question: "With ⟦3160⟧ and a constant gradient ⟦2832⟧, what velocity does momentum settle at?"
answer: "Step 1, at steady state ⟦3165⟧. Step 2, so ⟦3166⟧ and ⟦3167⟧. Momentum moves 10 times faster than plain gradient descent on a steady slope."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Constrained and unconstrained optimization"]
tools: ["NumPy", "PyTorch"]
topic: {"id": "09.08", "title": "Constrained and unconstrained optimization"}
---

## The big picture

Plain gradient descent is slow on long, narrow valleys (lesson 09.54). Nearly every optimiser used in practice adds momentum, a per-coordinate scale, or both.

## Momentum

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>v</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mi>β</mi><msub><mi>v</mi><mi>k</mi></msub><mo>−</mo><mi>η</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>k</mi></msub><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><msub><mi>x</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>x</mi><mi>k</mi></msub><mo>+</mo><msub><mi>v</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow></math></span></span>
Across the valley the gradient flips sign each step, so the velocity cancels out. Along the valley it points the same way each step, so the velocity builds up.

```
A = np.diag([1.0, 25.0])
def steps(eta, beta):
    x = np.array([1.0, 1.0]); v = np.zeros(2)
    for k in range(1, 10000):
        v = beta * v - eta * (A @ x); x = x + v
        if np.linalg.norm(x) < 1e-6:
            return k
print(steps(0.04, 0), steps(0.077, 0), steps(1 / 9, 0.444))  # 339 180 47
```

~~~dg-raw
%%viz f_bars v=339,180,47 l=GD_η_=_0.04,GD_η_=_0.077,momentum hi=none yl=steps_to_reach_10⁻⁶ | Steps to converge on f = ½(x² + 25y²). Even the best plain step size needs 180 steps; momentum with tuned settings needs 47.
~~~

## Adaptive methods: Adam

Adam keeps two running averages per coordinate: the gradient <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi></mrow></math></span> and the squared gradient <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi></mrow></math></span>. The step is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>x</mi><mo>←</mo><mi>x</mi><mo>−</mo><mi>η</mi><mtext> </mtext><mfrac><mover accent="true"><mi>m</mi><mo>^</mo></mover><mrow><msqrt><mover accent="true"><mi>s</mi><mo>^</mo></mover></msqrt><mo>+</mo><mi>ϵ</mi></mrow></mfrac></mrow></math></span></span>
where the hats correct for starting the averages at zero. A coordinate with large gradients gets a smaller multiplier, so each coordinate moves about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi></mrow></math></span> per step at first.

```
g = np.array([1.0, 25.0]); eta = 0.01
m = 0.1 * g; s = 0.001 * g ** 2                # first update of both averages
m_hat = m / (1 - 0.9); s_hat = s / (1 - 0.999)  # bias correction
print(np.round(eta * m_hat / (np.sqrt(s_hat) + 1e-8), 4))  # [0.01 0.01]
```

<div class="take">
<div class="lab">Key takeaway</div>

momentum smooths the path through narrow valleys; Adam rescales each coordinate so badly scaled problems move evenly.

</div>

## Where this breaks in practice

- Adam with default settings trains most networks well, but tuned SGD with momentum sometimes generalises better. Try both when the result matters.

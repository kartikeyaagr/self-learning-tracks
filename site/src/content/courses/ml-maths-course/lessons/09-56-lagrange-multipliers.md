---
id: "09.56"
course: "ml-maths-course"
module: "09"
slug: "lagrange-multipliers"
title: "Lagrange Multipliers"
keyIdeas: ["To optimise ⟦90⟧ subject to ⟦3170⟧, solve ⟦3171⟧ together with ⟦3170⟧.", "At the optimum the gradient of ⟦90⟧ is perpendicular to the constraint, just as ⟦3172⟧ is.", "⟦559⟧ measures how fast the best value changes when the constraint is relaxed."]
question: "Find the minimum-norm vector ⟦378⟧ with ⟦3168⟧ for a fixed non-zero ⟦364⟧. What is the minimum of ⟦3169⟧?"
answer: "Step 1, ⟦3185⟧ and ⟦3186⟧, so ⟦3187⟧. Step 2, substitute into ⟦3168⟧: ⟦3188⟧, so ⟦3189⟧ and ⟦3190⟧. Step 3, ⟦3191⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Constrained and unconstrained optimization"]
tools: ["NumPy", "SciPy"]
topic: {"id": "09.08", "title": "Constrained and unconstrained optimization"}
---

## The big picture

Often there are extra rules: probabilities must add to 1, a direction must have length 1. PCA, for instance, maximises variance over unit vectors. Lagrange multipliers handle equality constraints like these.

## The condition

At the optimum you cannot improve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> by sliding along the constraint. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> has no component along it and is perpendicular to it, like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>g</mi></mrow></math></span>. Two perpendicular vectors in the plane are parallel:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo>⋆</mo></msup><mo stretchy="false">)</mo><mo>=</mo><mi>λ</mi><mtext> </mtext><mi mathvariant="normal">∇</mi><mi>g</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo>⋆</mo></msup><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><mi>g</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo>⋆</mo></msup><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span></span>

~~~dg-raw
%%viz lagr live | Slide p along the constraint. At the best point, ∇f and ∇g line up.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the largest x + y on the unit circle**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>g</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mi>x</mi><mo separator="true">,</mo><mn>2</mn><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>. The condition gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>=</mo><mn>2</mn><mi>λ</mi><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>=</mo><mn>2</mn><mi>λ</mi><mi>y</mi></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow></math></span>.

On the circle <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msup><mi>x</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>y</mi><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mn>2</mn></msqrt></mrow></math></span> and the best value is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>2</mn></msqrt><mo>≈</mo><mn>1.414</mn></mrow></math></span>.

</div>

```
th = np.linspace(0, 2 * np.pi, 100001)
vals = np.cos(th) + np.sin(th)
print(round(vals.max(), 4), round(th[vals.argmax()], 4))  # 1.4142 0.7854
```

## PCA is a Lagrange problem

Maximise <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>S</mi><mi>w</mi></mrow></math></span> subject to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn></mrow></math></span>. The condition is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>S</mi><mi>w</mi><mo>=</mo><mn>2</mn><mi>λ</mi><mi>w</mi></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi><mi>w</mi><mo>=</mo><mi>λ</mi><mi>w</mi></mrow></math></span>: the optimal directions are eigenvectors of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>, and the variance captured is the eigenvalue.

```
from scipy.optimize import minimize
res = minimize(lambda w: w @ w, x0=np.array([1.0, 1.0]), constraints={'type': 'eq', 'fun': lambda w: np.array([3.0, 4]) @ w - 1})
print(np.round(res.x, 4))  # [0.12 0.16]
```

A numerical optimiser agrees with the exercise: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mo>⋆</mo></msup><mo>=</mo><mi>a</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∥</mi><mi>a</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>25</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

at a constrained optimum, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> is a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>g</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- The condition finds every constrained critical point, including maxima and saddles. Compare the candidates' values.

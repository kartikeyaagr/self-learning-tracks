---
id: "09.52"
course: "ml-maths-course"
module: "09"
slug: "critical-points-in-several-variables"
title: "Critical Points in Several Variables"
keyIdeas: ["A critical point is where ⟦2189⟧: the tangent plane is flat.", "Every local minimum of a differentiable ⟦90⟧ is a critical point, but not every critical point is a minimum.", "Critical points can be minima, maxima or saddles; lesson 09.53 tells them apart."]
question: "Find the critical points of ⟦3129⟧."
answer: "Step 1, ⟦3137⟧. Step 2, set both to zero: ⟦3138⟧, ⟦1074⟧. Step 3, the critical points are ⟦310⟧ and ⟦822⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Constrained and unconstrained optimization"]
tools: ["NumPy"]
topic: {"id": "09.08", "title": "Constrained and unconstrained optimization"}
---

## The big picture

Training is optimisation: find the weights that make the loss smallest. The first question is where the ground is flat, because a minimum can only be there.

## The first-order condition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo>⋆</mo></msup><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span></span>
If any partial derivative were non-zero, you could step against it and go lower, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mo>⋆</mo></msup></mrow></math></span> would not be a minimum.

~~~dg-raw
%%viz f_curve f=x*x*x*x-x*x x=-1.5,1.5 y=-0.4,1 tk=0.5 pts=-0.707,min;0,max;0.707,min | f(x) = x⁴ − x² has three flat points: two minima at ±1/√2 and a local maximum at 0. A zero derivative alone does not say which is which.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**solving ∇f = 0**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mi>x</mi><mi>y</mi><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>−</mo><mn>3</mn><mi>x</mi></mrow></math></span>. The gradient is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>x</mi><mo>+</mo><mi>y</mi><mo>−</mo><mn>3</mn><mo separator="true">,</mo><mtext> </mtext><mi>x</mi><mo>+</mo><mn>2</mn><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>.

The second equation gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo>−</mo><mn>2</mn><mi>y</mi></mrow></math></span>. Substituting: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>4</mn><mi>y</mi><mo>+</mo><mi>y</mi><mo>−</mo><mn>3</mn><mo>=</mo><mn>0</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mo>−</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span>.

The only critical point is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
A = np.array([[2.0, 1], [1, 2]]); b = np.array([3.0, 0])
print(np.linalg.solve(A, b))  # [ 2. -1.]
```

For a quadratic, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo>=</mo><mn>0</mn></mrow></math></span> is a linear system.

<div class="take">
<div class="lab">Key takeaway</div>

minima live where the gradient vanishes, but a zero gradient is only a candidate.

</div>

## Where this breaks in practice

- In a deep network, almost all critical points are saddles, not minima. A small gradient means "flat here", not "done".

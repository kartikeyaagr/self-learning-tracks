---
id: "09.53"
course: "ml-maths-course"
module: "09"
slug: "the-second-derivative-test"
title: "The Second-Derivative Test"
keyIdeas: ["At a critical point, the Hessian's eigenvalues classify it: all positive is a minimum, all negative a maximum, mixed signs a saddle.", "A zero eigenvalue makes the test inconclusive.", "A function is convex when its Hessian is positive semidefinite everywhere; then every local minimum is global."]
question: "Classify the critical points of ⟦3129⟧."
answer: "Step 1, the Hessian is ⟦3145⟧. Step 2, at ⟦310⟧ the eigenvalues are ⟦3146⟧: a minimum. Step 3, at ⟦822⟧ they are ⟦3147⟧: a saddle."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Constrained and unconstrained optimization"]
tools: ["NumPy"]
topic: {"id": "09.08", "title": "Constrained and unconstrained optimization"}
---

## The big picture

The gradient finds flat points; the Hessian (lesson 09.33) says which way the ground curves there.

## Reading the eigenvalues

| Hessian eigenvalues | Point | Shape |
| --- | --- | --- |
| all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&gt;</mo><mn>0</mn></mrow></math></span> | local minimum | bowl |
| all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo><mn>0</mn></mrow></math></span> | local maximum | dome |
| some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&gt;</mo><mn>0</mn></mrow></math></span>, some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo><mn>0</mn></mrow></math></span> | saddle | up one way, down another |
| some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mn>0</mn></mrow></math></span> | inconclusive | flat in some direction |

~~~dg-raw
%%viz f_curve f=x*x g=-x*x x=-1.5,1.5 y=-2.3,4 tk=0.5 fl=slice_along_x:_curves_up gl=slice_along_y:_curves_down | Two slices through the saddle f(x, y) = x² − y² at the origin. Along x it curves up, along y it curves down, so the origin is neither a minimum nor a maximum.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**three quadratics at the origin**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></math></span>: Hessian eigenvalues <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo separator="true">,</mo><mn>2</mn></mrow></math></span>. Minimum.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><msup><mi>x</mi><mn>2</mn></msup><mo>−</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>2</mn></mrow></math></span>. Maximum.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>−</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>2</mn></mrow></math></span>. Saddle.

</div>

```
H = np.array([[6.0, 0], [0, 2]])
print(np.linalg.eigvalsh(H), np.linalg.eigvalsh(np.array([[-6.0, 0], [0, 2]])))  # [2. 6.] [-6.  2.]
```

## Convexity

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is **convex** if the line between any two points on its graph lies on or above the graph: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>t</mi><mi>x</mi><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>t</mi><mo stretchy="false">)</mo><mi>y</mi><mo stretchy="false">)</mo><mo>≤</mo><mi>t</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>t</mi><mo stretchy="false">)</mo><mi>f</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>. For twice-differentiable <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> this is the same as a positive semidefinite Hessian everywhere. Least squares and logistic regression are convex, so gradient descent finds their global minimum.

<div class="take">
<div class="lab">Key takeaway</div>

the signs of the Hessian's eigenvalues classify a critical point; convexity makes every local minimum global.

</div>

## Where this breaks in practice

- Neural network losses are not convex. Their Hessians usually have a few large positive eigenvalues, many near zero and some negative.

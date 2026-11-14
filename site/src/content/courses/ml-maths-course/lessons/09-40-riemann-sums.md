---
id: "09.40"
course: "ml-maths-course"
module: "09"
slug: "riemann-sums"
title: "Riemann Sums"
keyIdeas: ["A Riemann sum is ⟦2960⟧ for any choice of points ⟦2961⟧ in the strips.", "Left, right, midpoint and trapezoid rules are different choices; all converge to the same limit for continuous ⟦90⟧.", "The midpoint rule's error shrinks like ⟦2962⟧; left and right rules like ⟦365⟧."]
question: "With ⟦2383⟧ on ⟦4⟧ for ⟦2328⟧, compute the midpoint sum."
answer: "Step 1, midpoints ⟦2964⟧. Step 2, squares ⟦2965⟧, sum ⟦2966⟧. Step 3, times ⟦131⟧: ⟦2967⟧, already within ⟦2968⟧ of ⟦2954⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

There are many ways to pick the rectangle heights. They all agree in the limit, but some get close much faster. This is the first taste of numerical integration.

## The general sum

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>S</mi><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>f</mi><mo stretchy="false">(</mo><msubsup><mi>x</mi><mi>i</mi><mo>∗</mo></msubsup><mo stretchy="false">)</mo><mtext> </mtext><mi mathvariant="normal">Δ</mi><mi>x</mi></mrow></math></span></span>

~~~dg-raw
%%viz riem live | More strips, better fit. Compare the four rules.
~~~

```
f = lambda x: x ** 2
for n in (4, 40):
    dx = 1 / n; left = np.arange(n) * dx
    L = f(left).sum() * dx; R = f(left + dx).sum() * dx; M = f(left + dx / 2).sum() * dx
    print(n, round(L, 5), round(M, 5), round(R, 5))
```

Ten times more strips makes the left and right errors 10 times smaller, but the midpoint error 100 times smaller.

<div class="take">
<div class="lab">Key takeaway</div>

Riemann sums with any choice of points converge to the same area; midpoints converge fastest of the simple rules.

</div>

---
id: "09.43"
course: "ml-maths-course"
module: "09"
slug: "numerical-integration-rules"
title: "Numerical Integration Rules"
keyIdeas: ["The trapezoid rule joins neighbouring points by straight lines; its error shrinks like ⟦2962⟧.", "Simpson's rule fits parabolas through triples of points; its error shrinks like ⟦3001⟧.", "`scipy.integrate.quad` adapts the spacing automatically and is very accurate in one dimension."]
question: "With ⟦3000⟧ strips on ⟦4⟧ for ⟦2328⟧, compute the trapezoid and Simpson estimates."
answer: "Step 1, points ⟦3007⟧ with values ⟦3008⟧. Step 2, trapezoid: ⟦3009⟧. Step 3, Simpson: ⟦3010⟧, exact, because Simpson is exact for cubics and below."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy", "SciPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

When there is no antiderivative, compute the integral numerically. Smarter rules use the same function values to get far more accuracy.

## Rules

| Rule | Shape between points | Error |
| --- | --- | --- |
| left or right | flat | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∝</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span> |
| trapezoid | straight line | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∝</mo><mn>1</mn><mi mathvariant="normal">/</mi><msup><mi>n</mi><mn>2</mn></msup></mrow></math></span> |
| Simpson | parabola | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∝</mo><mn>1</mn><mi mathvariant="normal">/</mi><msup><mi>n</mi><mn>4</mn></msup></mrow></math></span> |

```
from scipy.integrate import quad, simpson, trapezoid
f = lambda x: np.exp(-x ** 2)
x = np.linspace(0, 2, 11)
exact = quad(f, 0, 2)[0]
print(f"{abs(trapezoid(f(x), x=x) - exact):.0e}", f"{abs(simpson(f(x), x=x) - exact):.0e}")
```

With the same 11 points, Simpson is about a hundred times more accurate than the trapezoid rule here.

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x) x=0,2 y=0,1 tk=0.5 pts=0;0.5;1;1.5;2 | The integrand e^(−x²) on [0, 2] with five sample points. The trapezoid rule joins the dots with straight lines; Simpson's rule fits a parabola through each consecutive triple.
~~~

## Higher dimensions

Grid rules need <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>m</mi><mi>d</mi></msup></mrow></math></span> points for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi></mrow></math></span> per axis in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions: 10 per axis in 100 dimensions is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>100</mn></msup></mrow></math></span>. That is why high-dimensional integrals use Monte Carlo (lesson 09.45).

<div class="take">
<div class="lab">Key takeaway</div>

trapezoid and Simpson rules use the same samples with better interpolation; in 1D, `quad` is the default.

</div>

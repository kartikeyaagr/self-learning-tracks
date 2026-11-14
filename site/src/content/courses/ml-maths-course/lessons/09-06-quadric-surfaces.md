---
id: "09.06"
course: "ml-maths-course"
module: "09"
slug: "quadric-surfaces"
title: "Quadric Surfaces"
keyIdeas: ["Quadratic functions ⟦2500⟧ have level sets that are ellipsoids (⟦8⟧ positive definite), hyperboloids or saddles (⟦8⟧ indefinite).", "Near a minimum every smooth loss looks like a quadric (lesson 07.04).", "Gaussian density contours ⟦2501⟧ are ellipsoids."]
question: "Classify the level sets of ⟦2498⟧. What does the level set ⟦2499⟧ look like?"
answer: "Step 1, ⟦2506⟧ is indefinite, so ⟦90⟧ is a saddle. Step 2, ⟦2507⟧ gives hyperbolas. Step 3, ⟦2499⟧ means ⟦2508⟧: the two lines ⟦2509⟧ and ⟦2510⟧ crossing at the saddle point."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

After planes, the next shapes are quadratic. They describe local loss bowls, saddles and Gaussian contours. Module 07 showed their eigenvalue structure; here we look at them as surfaces.

## Three cases

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> | Surface | Level sets |
| --- | --- | --- |
| positive definite | bowl | ellipses |
| indefinite | saddle | hyperbolas |
| negative definite | dome | ellipses |

~~~dg-raw
%%viz f_plane x=-3,3 y=-3,3 tk=1 ln=1,-1,0,f_=_0,m,0.9;1,1,0,,m,0.9 pg=3.00:-2.83_2.78:-2.59_2.56:-2.36_2.35:-2.12_2.14:-1.89_1.93:-1.65_1.73:-1.42_1.55:-1.18_1.37:-0.94_1.22:-0.71_1.11:-0.47_1.03:-0.24_1.00:0.00_1.03:0.24_1.11:0.47_1.22:0.71_1.37:0.94_1.55:1.18_1.73:1.42_1.93:1.65_2.14:1.89_2.35:2.12_2.56:2.36_2.78:2.59_3.00:2.83_3.00:2.83_3.00:-2.83,a;-3.00:-2.83_-2.78:-2.59_-2.56:-2.36_-2.35:-2.12_-2.14:-1.89_-1.93:-1.65_-1.73:-1.42_-1.55:-1.18_-1.37:-0.94_-1.22:-0.71_-1.11:-0.47_-1.03:-0.24_-1.00:0.00_-1.03:0.24_-1.11:0.47_-1.22:0.71_-1.37:0.94_-1.55:1.18_-1.73:1.42_-1.93:1.65_-2.14:1.89_-2.35:2.12_-2.56:2.36_-2.78:2.59_-3.00:2.83_-3.00:2.83_-3.00:-2.83,a;-2.83:3.00_-2.59:2.78_-2.36:2.56_-2.12:2.35_-1.89:2.14_-1.65:1.93_-1.42:1.73_-1.18:1.55_-0.94:1.37_-0.71:1.22_-0.47:1.11_-0.24:1.03_0.00:1.00_0.24:1.03_0.47:1.11_0.71:1.22_0.94:1.37_1.18:1.55_1.42:1.73_1.65:1.93_1.89:2.14_2.12:2.35_2.36:2.56_2.59:2.78_2.83:3.00_2.83:3.00_-2.83:3.00,f;-2.83:-3.00_-2.59:-2.78_-2.36:-2.56_-2.12:-2.35_-1.89:-2.14_-1.65:-1.93_-1.42:-1.73_-1.18:-1.55_-0.94:-1.37_-0.71:-1.22_-0.47:-1.11_-0.24:-1.03_0.00:-1.00_0.24:-1.03_0.47:-1.11_0.71:-1.22_0.94:-1.37_1.18:-1.55_1.42:-1.73_1.65:-1.93_1.89:-2.14_2.12:-2.35_2.36:-2.56_2.59:-2.78_2.83:-3.00_2.83:-3.00_-2.83:-3.00,f | The saddle x² − y². Green: where f ≥ 1, bounded by hyperbolas opening left and right. Orange: where f ≤ −1, opening up and down. The grey lines y = ±x are the level set f = 0.
~~~

```
f = lambda x, y: x ** 2 - y ** 2
t = np.linspace(-1, 1, 5)
print(np.allclose(f(np.cosh(t), np.sinh(t)), 1))  # True
```

The points <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>cosh</mi><mo>⁡</mo><mi>t</mi><mo separator="true">,</mo><mi>sinh</mi><mo>⁡</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span> trace one branch of the hyperbola <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

## Other shapes worth recognising

| Set | Equation | Where it shows up |
| --- | --- | --- |
| ball | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>≤</mo><mi>r</mi></mrow></math></span> | weight constraints, trust regions |
| ellipsoid | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo><mo>≤</mo><mi>c</mi></mrow></math></span> | Gaussian confidence regions |
| half-space | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo>≥</mo><mn>0</mn></mrow></math></span> | one class of a linear classifier |

<div class="take">
<div class="lab">Key takeaway</div>

quadratic functions give ellipses around minima and hyperbolas around saddles; the eigenvalues decide which.

</div>

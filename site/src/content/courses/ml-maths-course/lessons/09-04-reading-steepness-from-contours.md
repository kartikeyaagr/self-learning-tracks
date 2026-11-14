---
id: "09.04"
course: "ml-maths-course"
module: "09"
slug: "reading-steepness-from-contours"
title: "Reading Steepness from Contours"
keyIdeas: ["Contours drawn at equal steps of ⟦119⟧ that are close together mean steep; far apart means flat.", "Round contours mean equally steep in every direction; long thin ellipses mean steep one way and flat another.", "Long thin contours are the ill-conditioned case where gradient descent zig-zags."]
question: "For ⟦2484⟧, how far must you walk along each axis from the origin to reach ⟦2485⟧?"
answer: "Step 1, along ⟦7⟧: ⟦431⟧, so 1 unit. Step 2, along ⟦509⟧: ⟦2488⟧, so 0.1 units. Step 3, the contours are 10 times tighter in ⟦509⟧: the function is 10 times steeper that way."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

A contour map tells you at a glance where the function is steep, where it is flat and whether the bowl is round or stretched. Those three facts predict how gradient descent will behave.

## Spacing is steepness

If contours are drawn for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>=</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span>, the distance between neighbours is roughly <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mtext>slope</mtext></mrow></math></span>. Tight spacing means a large slope.

~~~dg-raw
%%viz f_plane x=-2.2,2.2 y=-1,1 tk=1 pg=2.00:0.00_1.98:0.03_1.90:0.06_1.78:0.09_1.62:0.12_1.41:0.14_1.18:0.16_0.91:0.18_0.62:0.19_0.31:0.20_0.00:0.20_-0.31:0.20_-0.62:0.19_-0.91:0.18_-1.18:0.16_-1.41:0.14_-1.62:0.12_-1.78:0.09_-1.90:0.06_-1.98:0.03_-2.00:0.00_-1.98:-0.03_-1.90:-0.06_-1.78:-0.09_-1.62:-0.12_-1.41:-0.14_-1.18:-0.16_-0.91:-0.18_-0.62:-0.19_-0.31:-0.20_0.00:-0.20_0.31:-0.20_0.62:-0.19_0.91:-0.18_1.18:-0.16_1.41:-0.14_1.62:-0.12_1.78:-0.09_1.90:-0.06_1.98:-0.03,m;1.00:0.00_0.99:0.02_0.95:0.03_0.89:0.05_0.81:0.06_0.71:0.07_0.59:0.08_0.45:0.09_0.31:0.10_0.16:0.10_0.00:0.10_-0.16:0.10_-0.31:0.10_-0.45:0.09_-0.59:0.08_-0.71:0.07_-0.81:0.06_-0.89:0.05_-0.95:0.03_-0.99:0.02_-1.00:0.00_-0.99:-0.02_-0.95:-0.03_-0.89:-0.05_-0.81:-0.06_-0.71:-0.07_-0.59:-0.08_-0.45:-0.09_-0.31:-0.10_-0.16:-0.10_0.00:-0.10_0.16:-0.10_0.31:-0.10_0.45:-0.09_0.59:-0.08_0.71:-0.07_0.81:-0.06_0.89:-0.05_0.95:-0.03_0.99:-0.02,a | Contours f = 1 and f = 4 of x² + 100y². The bowl is 10 times steeper in y than in x, so the ellipses are 10 times longer than they are tall.
~~~

## Shapes

~~~dg-flow Reading a contour plot of a loss.
Tight contours -(steep)-> large gradient, small safe step
Wide contours -(flat)-> *small gradient, slow progress
Elongated ellipses -(ill-conditioned)-> zig-zagging gradient descent
~~~

```
f = lambda x, y: x ** 2 + 100 * y ** 2
print(f(1, 0), round(f(0, 0.1), 10))  # 1 1.0
```

<div class="take">
<div class="lab">Key takeaway</div>

closely spaced contours mean steep, widely spaced mean flat, and stretched ellipses warn of slow, zig-zagging descent.

</div>

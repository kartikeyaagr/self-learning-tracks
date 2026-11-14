---
id: "09.03"
course: "ml-maths-course"
module: "09"
slug: "level-sets-and-contour-maps"
title: "Level Sets and Contour Maps"
keyIdeas: ["A level set is ⟦2472⟧: every input where ⟦90⟧ takes the value ⟦119⟧.", "Level sets of ⟦2461⟧ for several ⟦119⟧ form a contour map, like height lines on a hiking map.", "For ⟦2451⟧ the level sets are circles; for ⟦2473⟧ they are ellipses."]
question: "Describe the level set ⟦2471⟧ and check that ⟦849⟧ and ⟦5⟧ lie on it."
answer: "Step 1, divide by 4: ⟦2481⟧, an ellipse crossing the ⟦7⟧-axis at ⟦2482⟧ and the ⟦509⟧-axis at ⟦1373⟧. Step 2, ⟦849⟧: ⟦2483⟧. Step 3, ⟦5⟧: ⟦1647⟧. Both lie on it."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

A contour map shows a whole surface on flat paper: each curve joins points of equal height. Contour plots of losses are how most people picture optimisation.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>:</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>c</mi><mo stretchy="false">}</mo></mrow></math></span></span>
In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> dimensions a level set is usually an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>-dimensional surface.

~~~dg-raw
%%viz f_plane x=-3.2,3.2 y=-2.2,2.2 tk=1 pg=3.00:0.00_2.96:0.23_2.85:0.46_2.67:0.68_2.43:0.88_2.12:1.06_1.76:1.21_1.36:1.34_0.93:1.43_0.47:1.48_0.00:1.50_-0.47:1.48_-0.93:1.43_-1.36:1.34_-1.76:1.21_-2.12:1.06_-2.43:0.88_-2.67:0.68_-2.85:0.46_-2.96:0.23_-3.00:0.00_-2.96:-0.23_-2.85:-0.46_-2.67:-0.68_-2.43:-0.88_-2.12:-1.06_-1.76:-1.21_-1.36:-1.34_-0.93:-1.43_-0.47:-1.48_0.00:-1.50_0.47:-1.48_0.93:-1.43_1.36:-1.34_1.76:-1.21_2.12:-1.06_2.43:-0.88_2.67:-0.68_2.85:-0.46_2.96:-0.23,m;2.00:0.00_1.98:0.16_1.90:0.31_1.78:0.45_1.62:0.59_1.41:0.71_1.18:0.81_0.91:0.89_0.62:0.95_0.31:0.99_0.00:1.00_-0.31:0.99_-0.62:0.95_-0.91:0.89_-1.18:0.81_-1.41:0.71_-1.62:0.59_-1.78:0.45_-1.90:0.31_-1.98:0.16_-2.00:0.00_-1.98:-0.16_-1.90:-0.31_-1.78:-0.45_-1.62:-0.59_-1.41:-0.71_-1.18:-0.81_-0.91:-0.89_-0.62:-0.95_-0.31:-0.99_0.00:-1.00_0.31:-0.99_0.62:-0.95_0.91:-0.89_1.18:-0.81_1.41:-0.71_1.62:-0.59_1.78:-0.45_1.90:-0.31_1.98:-0.16,a;1.00:0.00_0.99:0.08_0.95:0.15_0.89:0.23_0.81:0.29_0.71:0.35_0.59:0.40_0.45:0.45_0.31:0.48_0.16:0.49_0.00:0.50_-0.16:0.49_-0.31:0.48_-0.45:0.45_-0.59:0.40_-0.71:0.35_-0.81:0.29_-0.89:0.23_-0.95:0.15_-0.99:0.08_-1.00:0.00_-0.99:-0.08_-0.95:-0.15_-0.89:-0.23_-0.81:-0.29_-0.71:-0.35_-0.59:-0.40_-0.45:-0.45_-0.31:-0.48_-0.16:-0.49_0.00:-0.50_0.16:-0.49_0.31:-0.48_0.45:-0.45_0.59:-0.40_0.71:-0.35_0.81:-0.29_0.89:-0.23_0.95:-0.15_0.99:-0.08,b pt=2,0,(2~_0),k | Level sets of f(x, y) = x² + 4y² for c = 1, 4 and 9 (inner to outer). Each is an ellipse twice as wide as it is tall; (2, 0) lies on the c = 4 curve.
~~~

```
f = lambda x, y: x ** 2 + 4 * y ** 2
th = np.linspace(0, 2 * np.pi, 7)
print(np.allclose(f(2 * np.cos(th), np.sin(th)), 4))  # True
```

Every point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>cos</mi><mo>⁡</mo><mi>t</mi><mo separator="true">,</mo><mi>sin</mi><mo>⁡</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span> lies on the level set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>=</mo><mn>4</mn></mrow></math></span>.

## Gaussian contours are ellipses

A 2D Gaussian with covariance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">diag</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> has density proportional to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>exp</mi><mo>⁡</mo><mo stretchy="false">(</mo><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><msup><mi>x</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mn>4</mn><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math></span>. The density is constant exactly where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mn>4</mn><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></math></span> is constant, so its contours are ellipses twice as wide as tall.

<div class="take">
<div class="lab">Key takeaway</div>

a level set collects all inputs with the same output; a stack of them is a contour map.

</div>

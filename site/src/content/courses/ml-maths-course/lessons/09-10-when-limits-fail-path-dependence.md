---
id: "09.10"
course: "ml-maths-course"
module: "09"
slug: "when-limits-fail-path-dependence"
title: "When Limits Fail: Path Dependence"
keyIdeas: ["If ⟦90⟧ approaches different values along two paths into ⟦364⟧, the limit does not exist.", "Checking along the axes alone is not enough; try lines ⟦2544⟧ and curves like ⟦2545⟧.", "Classic example: ⟦2546⟧ is 0 on the axes and ⟦2547⟧ on the diagonal."]
question: "Show that ⟦2543⟧ does not exist."
answer: "Step 1, along ⟦1074⟧: ⟦2550⟧. Step 2, along ⟦2509⟧: ⟦2551⟧. Step 3, along ⟦2544⟧: ⟦2552⟧, which depends on ⟦958⟧. Different paths give different values, so there is no limit."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

In one variable you check left and right. In two variables there are infinitely many directions, and a function can behave differently along each. This lesson shows the standard way to prove a limit does not exist.

## The method

Find two paths into <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> with different limiting values.

~~~dg-raw
%%viz f_curve f=x/(1+x*x) x=-4,4 y=-0.6,0.6 tk=1 mark=max ml=m_=_1:_0.5 | The value of xy/(x² + y²) along the line y = mx, as a function of the slope m. Each line gives a different constant, so there is no single limit at the origin.
~~~

```
f = lambda x, y: x * y / (x ** 2 + y ** 2)
for m in (0, 1, -1, 2):
    print(m, round(f(1e-8, m * 1e-8), 6))
```

## Lines are not always enough

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="true" scriptlevel="0"><mfrac><mrow><msup><mi>x</mi><mn>2</mn></msup><mi>y</mi></mrow><mrow><msup><mi>x</mi><mn>4</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></mfrac></mstyle></mrow></math></span> tends to 0 along every straight line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mi>m</mi><mi>x</mi></mrow></math></span>, but along the parabola <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span> it equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>4</mn></msup><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mn>2</mn><msup><mi>x</mi><mn>4</mn></msup><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>2</mn></mrow></math></span>. So the limit still does not exist.

```
g = lambda x, y: x ** 2 * y / (x ** 4 + y ** 2)
x = 1e-4
print(round(g(x, 3 * x), 6), round(g(x, x ** 2), 6))  # 3.3e-05 0.5
```

<div class="take">
<div class="lab">Key takeaway</div>

two paths with different limiting values prove a limit does not exist; try lines and curves, not just the axes.

</div>

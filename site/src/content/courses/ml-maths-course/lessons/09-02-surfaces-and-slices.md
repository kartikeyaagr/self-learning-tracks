---
id: "09.02"
course: "ml-maths-course"
module: "09"
slug: "surfaces-and-slices"
title: "Surfaces and Slices"
keyIdeas: ["The graph of ⟦2461⟧ is a surface: above each point ⟦116⟧ sits height ⟦2461⟧.", "A slice fixes all but one variable and gives an ordinary curve, such as ⟦2462⟧.", "Loss-landscape plots are 1D or 2D slices through weight space along chosen directions."]
question: "For ⟦2451⟧, write the slice at ⟦831⟧ and the slice along the line through the origin in direction ⟦2428⟧."
answer: "Step 1, at ⟦831⟧: ⟦2468⟧, a parabola one unit up. Step 2, along ⟦2469⟧: ⟦2470⟧, a parabola through 0."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

You cannot see a surface in a million dimensions, but you can always walk along a straight line through it and look at the height. That is a slice, and it is the main way people visualise loss landscapes.

## Slices along an axis

Fix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msub><mi>y</mi><mn>0</mn></msub></mrow></math></span> and vary only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>: the curve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>↦</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><msub><mi>y</mi><mn>0</mn></msub><mo stretchy="false">)</mo></mrow></math></span>. Partial derivatives (lesson 09.12) are the slopes of these curves.

~~~dg-raw
%%viz f_curve f=x*x+1 g=x*x+4 x=-2,2 y=0,9 tk=1 fl=slice_y_=_1:_x²+1 gl=slice_y_=_2:_x²+4 | Two slices of the bowl f(x, y) = x² + y². Fixing y gives a parabola in x; a larger y lifts it.
~~~

## Slices along any direction

Pick a point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span> and a direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span>. The function <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>↦</mo><mi>f</mi><mo stretchy="false">(</mo><mi>θ</mi><mo>+</mo><mi>t</mi><mi>d</mi><mo stretchy="false">)</mo></mrow></math></span> is an ordinary one-variable function. A 2D version uses two directions:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>α</mi><mo separator="true">,</mo><mi>β</mi><mo stretchy="false">)</mo><mo>=</mo><mi>L</mi><mo stretchy="false">(</mo><mi>θ</mi><mo>+</mo><mi>α</mi><msub><mi>d</mi><mn>1</mn></msub><mo>+</mo><mi>β</mi><msub><mi>d</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span></span>

```
f = lambda x, y: x ** 2 + y ** 2
d = np.array([1, 1]) / np.sqrt(2)
for t in (-1.0, 0.0, 2.0):
    p = t * d; print(t, round(f(*p), 6))
```

Along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><msqrt><mn>2</mn></msqrt></mrow></math></span> the bowl is again <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>t</mi><mn>2</mn></msup></mrow></math></span>.

## Loss-landscape plots

Plots of a network's loss "landscape" are slices like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>α</mi><mo separator="true">,</mo><mi>β</mi><mo stretchy="false">)</mo></mrow></math></span> with random or chosen directions. They help intuition, but they show only a 2D cut through a space with millions of dimensions.

<div class="take">
<div class="lab">Key takeaway</div>

a slice turns a function of many variables into an ordinary curve along one line; that is how you look at a loss.

</div>

## Where this breaks in practice

- Slices depend heavily on how the directions are scaled. "Filter-normalised" directions were introduced because naive ones can make sharp and flat minima look arbitrarily different.

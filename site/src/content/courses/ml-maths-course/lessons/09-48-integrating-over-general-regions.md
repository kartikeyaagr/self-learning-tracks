---
id: "09.48"
course: "ml-maths-course"
module: "09"
slug: "integrating-over-general-regions"
title: "Integrating Over General Regions"
keyIdeas: ["For a region that is not a rectangle, the inner limits depend on the outer variable.", "Always sketch the region first; read the limits off the sketch.", "Swapping the order of integration keeps the region and changes the limits."]
question: "Write ⟦3071⟧ for the triangle with corners ⟦685⟧, ⟦310⟧, ⟦111⟧ in both orders."
answer: "Step 1, outer ⟦7⟧ from 0 to 1, inner ⟦509⟧ from 0 to ⟦7⟧: ⟦3082⟧. Step 2, outer ⟦509⟧ from 0 to 1, inner ⟦7⟧ from ⟦509⟧ to 1: ⟦3083⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Integration over two variables"]
tools: ["SciPy"]
topic: {"id": "09.07", "title": "Integration over two variables"}
---

## The big picture

Events like "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>+</mo><mi>Y</mi><mo>≤</mo><mn>1</mn></mrow></math></span>" or "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>≤</mo><mi>X</mi></mrow></math></span>" are regions with slanted edges. To get their probability you integrate over them, and the only new skill is getting the limits right.

## Limits from a sketch

For each value of the outer variable, draw the slice and ask where it enters and leaves the region. Those two points are the inner limits.

~~~dg-raw
%%viz f_plane x=-0.3,1.4 y=-0.3,1.4 tk=1 pg=0:0_1:0_1:1,a ln=-1,1,0,y_=_x,k,0.85 sg=0.6,0,0.6,0.6,f pt=0.6,0,x,f | The triangle 0 ≤ y ≤ x ≤ 1. At a fixed x the slice (orange) runs from y = 0 up to y = x.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a triangular region**

Integrate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>x</mi></mrow></math></span> over the triangle <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>≤</mo><mi>y</mi><mo>≤</mo><mi>x</mi><mo>≤</mo><mn>1</mn></mrow></math></span>.

Outer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> from 0 to 1; inner <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> from 0 to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>. Inner: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mi>x</mi></msubsup><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span>. Outer: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msup><mi>x</mi><mn>2</mn></msup><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span>.

Reversed: outer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> from 0 to 1; inner <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> to 1. Inner: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mi>y</mi><mn>1</mn></msubsup><mi>x</mi><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msup><mi>y</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></math></span>. Outer: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msup><mi>y</mi><mn>2</mn></msup><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><mfrac><mn>1</mn><mn>3</mn></mfrac></mrow></math></span>.

</div>

```
from scipy.integrate import dblquad
val, err = dblquad(lambda y, x: x, 0, 1, lambda x: 0, lambda x: x)
print(round(val, 6))  # 0.333333
```

## Swapping the order

Sometimes one order is impossible by hand and the other is easy. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msubsup><mo>∫</mo><mi>y</mi><mn>1</mn></msubsup><msup><mi>e</mi><msup><mi>x</mi><mn>2</mn></msup></msup><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span> has no antiderivative inside. Swap to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msubsup><mo>∫</mo><mn>0</mn><mi>x</mi></msubsup><msup><mi>e</mi><msup><mi>x</mi><mn>2</mn></msup></msup><mtext> </mtext><mi>d</mi><mi>y</mi><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mi>x</mi><msup><mi>e</mi><msup><mi>x</mi><mn>2</mn></msup></msup><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mfrac><mrow><mi>e</mi><mo>−</mo><mn>1</mn></mrow><mn>2</mn></mfrac></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

sketch the region, read the inner limits from a slice, and keep the same region when you swap the order.

</div>

## Where this breaks in practice

- The most common error is the limits, not the integration. After a swap, check that both versions give the region the same area.

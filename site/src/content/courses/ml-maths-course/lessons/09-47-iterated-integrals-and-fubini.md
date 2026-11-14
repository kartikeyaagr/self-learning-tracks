---
id: "09.47"
course: "ml-maths-course"
module: "09"
slug: "iterated-integrals-and-fubini"
title: "Iterated Integrals and Fubini"
keyIdeas: ["Fubini's theorem: over a rectangle, integrate one variable at a time, in either order.", "In the inner integral the outer variable is a constant, just as in a partial derivative.", "The order can be swapped when ⟦3059⟧, which holds for anything bounded on a bounded region."]
question: "Compute ⟦3057⟧ over ⟦3058⟧ both ways."
answer: "Step 1, inner in ⟦509⟧: ⟦3067⟧, then ⟦3068⟧. Step 2, inner in ⟦7⟧: ⟦3069⟧, then ⟦3070⟧. Same answer."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Integration over two variables"]
tools: ["SciPy"]
topic: {"id": "09.07", "title": "Integration over two variables"}
---

## The big picture

Nobody evaluates a double integral as a limit of grids. Fubini turns it into two ordinary integrals, so everything from lesson 09.42 applies.

## Two single integrals

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mo>∬</mo><mrow><mo stretchy="false">[</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">]</mo><mo>×</mo><mo stretchy="false">[</mo><mi>c</mi><mo separator="true">,</mo><mi>d</mi><mo stretchy="false">]</mo></mrow></msub><mi>f</mi><mtext> </mtext><mi>d</mi><mi>A</mi><mo>=</mo><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mtext> ⁣</mtext><mrow><mo fence="true">(</mo><msubsup><mo>∫</mo><mi>c</mi><mi>d</mi></msubsup><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mo fence="true">)</mo></mrow><mi>d</mi><mi>x</mi><mo>=</mo><msubsup><mo>∫</mo><mi>c</mi><mi>d</mi></msubsup><mtext> ⁣</mtext><mrow><mo fence="true">(</mo><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mo fence="true">)</mo></mrow><mi>d</mi><mi>y</mi></mrow></math></span></span>
The inner integral gives the area of one slice of the solid. The outer integral adds up the slices.

~~~dg-raw
%%viz f_plane x=-0.5,1.5 y=-0.5,2.5 tk=1 pg=0:0_1:0_1:2_0:2,a sg=0.6,0,0.6,2,f,d pt=0.6,1,slice_at_fixed_x,f | Fix x (dashed line), integrate along the slice in y, then add the slices up from x = 0 to x = 1.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a rectangle, both orders**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∬</mo><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>A</mi></mrow></math></span> over <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo><mo>×</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></math></span>.

Inner in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>, with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> fixed: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>2</mn></msubsup><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>2</mn></mrow></math></span>. Outer: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mo stretchy="false">(</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>2</mn><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mn>1</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>3</mn></mrow></math></span>.

Other order: inner <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>+</mo><mi>y</mi></mrow></math></span>, outer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>2</mn></msubsup><mo stretchy="false">(</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><mn>1</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>3</mn></mrow></math></span>.

</div>

```
from scipy.integrate import dblquad
val, err = dblquad(lambda y, x: x + y, 0, 1, 0, 2)
print(round(val, 6))  # 3.0
```

`dblquad` takes the integrand as `f(y, x)`, inner variable first, then the outer limits, then the inner limits.

## Products split apart

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mi>h</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> on a rectangle, the double integral is the product of two single integrals: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∬</mo><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mi>h</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>A</mi><mo>=</mo><mo>∫</mo><mi>g</mi><mtext> </mtext><mi>d</mi><mi>x</mi><mo>⋅</mo><mo>∫</mo><mi>h</mi><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span>. This is why independent random variables are easy to work with.

<div class="take">
<div class="lab">Key takeaway</div>

integrate the inside variable with the outside one held fixed, then integrate the result.

</div>

## Where this breaks in practice

- `dblquad` argument order (`y` before `x`) is a classic source of silent errors. Test it on a case you can do by hand.

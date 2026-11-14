---
id: "09.49"
course: "ml-maths-course"
module: "09"
slug: "polar-coordinates"
title: "Polar Coordinates"
keyIdeas: ["Polar coordinates: ⟦3085⟧, ⟦3086⟧, with ⟦3087⟧ and ⟦3088⟧.", "The area element is ⟦3089⟧. The extra ⟦118⟧ is the Jacobian determinant from lesson 09.32.", "Use polar coordinates for discs, rings and anything that depends only on ⟦3090⟧."]
question: "Compute ⟦3084⟧ over the unit disc."
answer: "Step 1, in polar, ⟦3098⟧ and ⟦3089⟧. Step 2, ⟦3099⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Integration over two variables"]
tools: ["NumPy"]
topic: {"id": "09.07", "title": "Integration over two variables"}
---

## The big picture

A disc is awkward in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> (the limits involve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mn>1</mn><mo>−</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></msqrt></mrow></math></span>) but simple in polar: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> from 0 to 1, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span> all the way round. Rotationally symmetric functions, like the Gaussian, become one-variable problems.

## Why the extra r

A small polar box of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>r</mi></mrow></math></span> by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>θ</mi></mrow></math></span> is a curved patch about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>r</mi></mrow></math></span> wide and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mtext> </mtext><mi>d</mi><mi>θ</mi></mrow></math></span> long. Its area is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mtext> </mtext><mi>d</mi><mi>r</mi><mtext> </mtext><mi>d</mi><mi>θ</mi></mrow></math></span>, so boxes far from the centre are bigger.

~~~dg-raw
%%viz f_plane x=-0.3,1.8 y=-0.3,1.8 tk=1 pg=1.05:0.58_1.01:0.64_0.97:0.71_0.92:0.77_0.86:0.83_0.81:0.89_0.75:0.94_0.99:1.25_1.08:1.18_1.15:1.11_1.22:1.03_1.29:0.95_1.35:0.86_1.40:0.77,a sg=0,0,1.05,0.58,m,d;0,0,0.75,0.94,m,d | A polar box: r from 1.2 to 1.6 and θ from 0.5 to 0.9. Its width is dr = 0.4 and its arc length is about r·dθ, so its area grows with r.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**area of a disc**

Disc of radius <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mrow><mn>2</mn><mi>π</mi></mrow></msubsup><msubsup><mo>∫</mo><mn>0</mn><mi>R</mi></msubsup><mi>r</mi><mtext> </mtext><mi>d</mi><mi>r</mi><mtext> </mtext><mi>d</mi><mi>θ</mi><mo>=</mo><mn>2</mn><mi>π</mi><mo>⋅</mo><mfrac><msup><mi>R</mi><mn>2</mn></msup><mn>2</mn></mfrac><mo>=</mo><mi>π</mi><msup><mi>R</mi><mn>2</mn></msup></mrow></math></span>.

Without the factor <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> you would get <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>π</mi><mi>R</mi></mrow></math></span>, which is the circumference, not the area.

</div>

```
from scipy.integrate import dblquad
val, err = dblquad(lambda r, t: r ** 3, 0, 2 * np.pi, 0, 1)
print(round(val, 6), round(np.pi / 2, 6))  # 1.570796 1.570796
```

<div class="take">
<div class="lab">Key takeaway</div>

in polar coordinates, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>A</mi><mo>=</mo><mi>r</mi><mtext> </mtext><mi>d</mi><mi>r</mi><mtext> </mtext><mi>d</mi><mi>θ</mi></mrow></math></span>; never drop the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- `np.arctan(y / x)` loses the quadrant. Use `np.arctan2(y, x)` to recover <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span>.

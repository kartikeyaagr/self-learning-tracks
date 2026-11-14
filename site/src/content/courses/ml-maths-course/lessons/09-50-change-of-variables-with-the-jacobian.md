---
id: "09.50"
course: "ml-maths-course"
module: "09"
slug: "change-of-variables-with-the-jacobian"
title: "Change of Variables With the Jacobian"
keyIdeas: ["If ⟦3101⟧ is smooth and invertible, ⟦3102⟧.", "⟦3103⟧ is the local area-scaling factor of ⟦196⟧ (lessons 03.02 and 09.32).", "The same factor turns up when you transform random variables and in normalising flows."]
question: "Under ⟦3100⟧, what is the area of the image of the unit square?"
answer: "Step 1, ⟦3109⟧. Step 2, ⟦3110⟧. Step 3, the image has area ⟦1606⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Integration over two variables"]
tools: ["NumPy"]
topic: {"id": "09.07", "title": "Integration over two variables"}
---

## The big picture

Polar coordinates are one change of variables. The general rule lets you pick whatever coordinates make the region or the integrand simple, as long as you pay the area-scaling toll.

## The rule

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mo>∬</mo><mrow><mi>T</mi><mo stretchy="false">(</mo><mi>S</mi><mo stretchy="false">)</mo></mrow></msub><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><msub><mo>∬</mo><mi>S</mi></msub><mi>f</mi><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">(</mo><mi>u</mi><mo separator="true">,</mo><mi>v</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mtext> </mtext><mo stretchy="false">∣</mo><mi>det</mi><mo>⁡</mo><msub><mi>J</mi><mi>T</mi></msub><mo stretchy="false">(</mo><mi>u</mi><mo separator="true">,</mo><mi>v</mi><mo stretchy="false">)</mo><mo stretchy="false">∣</mo><mtext> </mtext><mi>d</mi><mi>u</mi><mtext> </mtext><mi>d</mi><mi>v</mi></mrow></math></span></span>
For polar coordinates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>J</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mi>r</mi><mi>sin</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>sin</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>r</mi><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>J</mi><mo>=</mo><mi>r</mi></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-0.5,3.5 y=-1.5,1.5 tk=1 pg=0:0_1:0_1:1_0:1,m;0:0_2:-1_3:0_1:1,a | The unit square (grey) and its image under T(u, v) = (2u + v, −u + v) (green). The determinant is 3, so the green parallelogram has area 3.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a slanted region made square**

Integrate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>=</mo><mn>1</mn></mrow></math></span> over the parallelogram above. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> the limits are four slanted lines. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>u</mi><mo separator="true">,</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></math></span> it is the unit square: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mn>3</mn><mtext> </mtext><mi>d</mi><mi>u</mi><mtext> </mtext><mi>d</mi><mi>v</mi><mo>=</mo><mn>3</mn></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
T = np.array([[2.0, 1], [-1, 1]])                  # (u, v) -> (2u + v, -u + v)
pts = rng.random((1_000_000, 2)) * [3, 2] + [0, -1]  # uniform on the box [0, 3] x [-1, 1], area 6
uv = np.linalg.solve(T, pts.T).T                   # map each point back to (u, v)
print(round(((uv >= 0) & (uv <= 1)).all(1).mean() * 6, 2))  # 3.0
```

Sampling a box around the parallelogram and keeping the points whose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>u</mi><mo separator="true">,</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></math></span> lands in the unit square gives an area of 3.

<div class="take">
<div class="lab">Key takeaway</div>

change coordinates freely, but multiply by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">∣</mo><mi>det</mi><mo>⁡</mo><mi>J</mi><mo stretchy="false">∣</mo></mrow></math></span> to account for stretched area.

</div>

## Where this breaks in practice

- The map must be one-to-one on the region. If two <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>u</mi><mo separator="true">,</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></math></span> points land on the same <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>, that area is counted twice.

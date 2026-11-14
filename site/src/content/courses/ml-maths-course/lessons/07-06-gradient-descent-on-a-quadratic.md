---
id: "07.06"
course: "ml-maths-course"
module: "07"
slug: "gradient-descent-on-a-quadratic"
title: "Gradient Descent on a Quadratic"
keyIdeas: ["On ⟦2210⟧, gradient descent is ⟦2211⟧; eigen-coordinate ⟦274⟧ is multiplied by ⟦2212⟧ each step.", "Stable only if ⟦1778⟧. The best fixed step ⟦2213⟧ shrinks error by ⟦2214⟧ per step.", "Newton's step ⟦2215⟧ divides by curvature and solves a quadratic in one step; Adam and momentum are cheaper partial fixes."]
question: "Gradient descent on ⟦2207⟧ with ⟦2155⟧. With ⟦2208⟧, what happens to each coordinate, and about how many steps shrink the first coordinate 1000 times? What happens with ⟦2209⟧?"
answer: "Step 1, each step multiplies ⟦1370⟧ by ⟦2225⟧ and ⟦697⟧ by ⟦2226⟧. Step 2, with ⟦2208⟧: factors ⟦2227⟧ and ⟦2228⟧; ⟦697⟧ flips sign and shrinks. Step 3, ⟦2229⟧ gives ⟦2230⟧ steps. Step 4, with ⟦2209⟧ the factor for ⟦697⟧ is ⟦2231⟧: it grows 10% per step and diverges. The cut-off is ⟦2232⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Curvature and quadratic objectives"]
tools: ["NumPy"]
topic: {"id": "07.01", "title": "Curvature and quadratic objectives"}
---

## The big picture

Why does training blow up when the learning rate rises a little, and crawl when it is lowered? On a quadratic bowl you can solve gradient descent exactly, and both answers fall out.

## The exact solution

For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi></mrow></math></span> the gradient is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span>, so
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>x</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>x</mi><mi>k</mi></msub><mo>−</mo><mi>η</mi><mi>A</mi><msub><mi>x</mi><mi>k</mi></msub><mo>=</mo><mo stretchy="false">(</mo><mi>I</mi><mo>−</mo><mi>η</mi><mi>A</mi><mo stretchy="false">)</mo><msub><mi>x</mi><mi>k</mi></msub></mrow></math></span></span>
In eigen-coordinates each direction evolves alone:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>y</mi><mrow><mi>k</mi><mo separator="true">,</mo><mi>i</mi></mrow></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>η</mi><msub><mi>λ</mi><mi>i</mi></msub><msup><mo stretchy="false">)</mo><mi>k</mi></msup><mtext> </mtext><msub><mi>y</mi><mrow><mn>0</mn><mo separator="true">,</mo><mi>i</mi></mrow></msub></mrow></math></span></span>

~~~dg-raw
%%viz quad live | Shape the quadratic form with a, b, c, then run gradient descent from any start. Push η past 2/λ₁ to watch it blow up.
~~~

- **Stability**: need <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mn>1</mn><mo>−</mo><mi>η</mi><msub><mi>λ</mi><mi>i</mi></msub><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span> for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span>, i.e. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>&lt;</mo><mn>2</mn><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span>.
- **Speed**: the slowest direction has factor near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>−</mo><mi>η</mi><msub><mi>λ</mi><mi>min</mi><mo>⁡</mo></msub></mrow></math></span>. With the best fixed step the error shrinks by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>κ</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>κ</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> per step. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo>=</mo><mn>1</mn><msup><mn>0</mn><mn>4</mn></msup></mrow></math></span> that is about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.9998</mn></mrow></math></span>, so reducing error 1000 times takes about 35,000 steps.

```
A = np.diag([1.0, 100])
for eta in (0.019, 0.021):
    x = np.array([1.0, 1.0])
    for _ in range(100):
        x = x - eta * A @ x
    print(eta, np.round(x, 4))
```

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>=</mo><mn>0.019</mn></mrow></math></span> the first coordinate is still about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.98</mn><msup><mn>1</mn><mn>100</mn></msup><mo>≈</mo><mn>0.15</mn></mrow></math></span> after 100 steps. With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>=</mo><mn>0.021</mn></mrow></math></span> the second has exploded to about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.</mn><msup><mn>1</mn><mn>100</mn></msup><mo>≈</mo><mn>14</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span>.

## Newton's method

The step <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>=</mo><mo>−</mo><msup><mi>H</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> divides each direction by its own curvature. On a quadratic it lands on the minimum in one step:

```
A = np.diag([1.0, 100]); x = np.array([1.0, 1.0])
print(x - np.linalg.solve(A, A @ x))  # [0. 0.]
```

For a large network <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi></mrow></math></span> is far too big to build, so practice uses cheaper fixes.

| Fix for ill-conditioning | Effect |
| --- | --- |
| Standardising features | makes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> closer to a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi></mrow></math></span> |
| Momentum | steps needed scale like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi>κ</mi></msqrt></mrow></math></span> instead of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi></mrow></math></span> |
| Adam, preconditioning | divides each coordinate by an estimate of its curvature |

<div class="take">
<div class="lab">Key takeaway</div>

the largest Hessian eigenvalue caps the learning rate at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span>; the condition number sets how many steps training needs.

</div>

## Where this breaks in practice

- Real losses are only locally quadratic, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span> is a local limit. Training often sits right at it (the "edge of stability").
- The Hessian is never formed for big models; Hessian-vector products from autodiff are enough to estimate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span> by power iteration.

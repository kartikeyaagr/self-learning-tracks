---
id: "07.07"
course: "ml-maths-course"
module: "07"
slug: "singular-values-as-matrix-stretch"
title: "Singular Values as Matrix Stretch"
keyIdeas: ["Every matrix maps the unit circle (sphere) to an ellipse (ellipsoid); the semi-axis lengths are the singular values ⟦2237⟧.", "⟦2238⟧ is the spectral norm; the smallest singular value is the least stretch.", "Singular values exist for every matrix, any shape, unlike eigenvalues."]
question: "For ⟦2233⟧, compute ⟦2234⟧, ⟦2235⟧ and ⟦2236⟧. Which is the largest stretch?"
answer: "Step 1, ⟦2247⟧, length 5. Step 2, ⟦2248⟧, length 5. Step 3, ⟦2249⟧, length ⟦2250⟧. Step 4, the third is largest; in fact ⟦2251⟧, the maximum over all unit vectors."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Singular value decomposition"]
tools: ["NumPy"]
topic: {"id": "07.02", "title": "Singular value decomposition"}
---

## The big picture

Eigenvalues only make sense for square matrices, and can be complex. A cleaner question works for every matrix: how much does it stretch vectors, and in which directions? The answers are the singular values.

## The circle becomes an ellipse

~~~dg-raw
%%viz f_plane x=-7,7 y=-7,7 tk=2 pg=3.00:4.00_2.95:4.81_2.82:5.47_2.60:5.96_2.30:6.28_1.93:6.40_1.50:6.33_1.03:6.07_0.52:5.62_0.00:5.00_-0.52:4.23_-1.03:3.33_-1.50:2.33_-1.93:1.26_-2.30:0.15_-2.60:-0.96_-2.82:-2.05_-2.95:-3.07_-3.00:-4.00_-2.95:-4.81_-2.82:-5.47_-2.60:-5.96_-2.30:-6.28_-1.93:-6.40_-1.50:-6.33_-1.03:-6.07_-0.52:-5.62_0.00:-5.00_0.52:-4.23_1.03:-3.33_1.50:-2.33_1.93:-1.26_2.30:-0.15_2.60:0.96_2.82:2.05_2.95:3.07,a;1.00:0.00_0.98:0.17_0.94:0.34_0.87:0.50_0.77:0.64_0.64:0.77_0.50:0.87_0.34:0.94_0.17:0.98_0.00:1.00_-0.17:0.98_-0.34:0.94_-0.50:0.87_-0.64:0.77_-0.77:0.64_-0.87:0.50_-0.94:0.34_-0.98:0.17_-1.00:0.00_-0.98:-0.17_-0.94:-0.34_-0.87:-0.50_-0.77:-0.64_-0.64:-0.77_-0.50:-0.87_-0.34:-0.94_-0.17:-0.98_0.00:-1.00_0.17:-0.98_0.34:-0.94_0.50:-0.87_0.64:-0.77_0.77:-0.64_0.87:-0.50_0.94:-0.34_0.98:-0.17,m v=0.71,0.71,v₁,k;2.12,6.36,Av₁,f;-2.12,0.71,Av₂,b | A = [[3, 0], [4, 5]] maps the unit circle (grey) to the green ellipse. The longest semi-axis is Av₁ with length σ₁ ≈ 6.71; the shortest is Av₂ with length σ₂ ≈ 2.24.
~~~

A linear map sends the unit circle to an ellipse. Its semi-axes have lengths <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>2</mn></msub></mrow></math></span>, the **singular values**. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> dimensions the sphere becomes an ellipsoid with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> semi-axes.

```
A = np.array([[3.0, 0], [4, 5]])
th = np.linspace(0, 2 * np.pi, 10001)
stretch = np.linalg.norm(A @ np.stack([np.cos(th), np.sin(th)]), axis=0)
print(round(stretch.max(), 3), round(stretch.min(), 3))  # 6.708 2.236
print(np.round(np.linalg.svd(A, compute_uv=False), 3))  # [6.708 2.236]
```

Checking 10,001 directions, the largest and smallest stretches match the singular values from `np.linalg.svd`.

## What they tell you

| Quantity | From singular values |
| --- | --- |
| spectral norm <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><msub><mi mathvariant="normal">∥</mi><mn>2</mn></msub></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub></mrow></math></span> |
| area or volume scaling <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">∣</mo><mi>det</mi><mo>⁡</mo><mi>A</mi><mo stretchy="false">∣</mo></mrow></math></span> (square <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub><msub><mi>σ</mi><mn>2</mn></msub><mo>⋯</mo><msub><mi>σ</mi><mi>n</mi></msub></mrow></math></span> |
| rank | number of non-zero <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span> |
| condition number | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub><mi mathvariant="normal">/</mi><msub><mi>σ</mi><mi>n</mi></msub></mrow></math></span> |

Here <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub><msub><mi>σ</mi><mn>2</mn></msub><mo>=</mo><mn>6.708</mn><mo>×</mo><mn>2.236</mn><mo>=</mo><mn>15</mn><mo>=</mo><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

singular values are the semi-axes of the ellipse a matrix makes from the unit circle; the largest is the most the matrix can stretch anything.

</div>

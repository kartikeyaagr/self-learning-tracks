---
id: "07.10"
course: "ml-maths-course"
module: "07"
slug: "best-low-rank-approximation"
title: "Best Low-Rank Approximation"
keyIdeas: ["Keeping the top ⟦1199⟧ pieces, ⟦2282⟧, gives the best rank-⟦1199⟧ approximation (Eckart-Young).", "The errors are ⟦2283⟧ and ⟦2284⟧.", "How fast singular values decay tells you how compressible a matrix is."]
question: "A matrix has singular values ⟦2269⟧. Give the spectral and Frobenius errors of its best rank-2 approximation."
answer: "Step 1, spectral error is the first dropped value, ⟦2291⟧. Step 2, Frobenius error is ⟦2292⟧. Step 3, relative to ⟦2293⟧, that is about 9%."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Singular value decomposition"]
tools: ["NumPy"]
topic: {"id": "07.02", "title": "Singular value decomposition"}
---

## The big picture

If a matrix is "mostly" a few patterns, you can store just those patterns. The SVD finds them, and a theorem says no other choice of the same rank does better.

## Eckart-Young

Among all matrices of rank at most <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>A</mi><mi>k</mi></msub></mrow></math></span> is closest to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> in both the spectral and Frobenius norms:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><mo>−</mo><msub><mi>A</mi><mi>k</mi></msub><msub><mi mathvariant="normal">∥</mi><mn>2</mn></msub><mo>=</mo><msub><mi>σ</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">∥</mi><mi>A</mi><mo>−</mo><msub><mi>A</mi><mi>k</mi></msub><msub><mi mathvariant="normal">∥</mi><mi>F</mi></msub><mo>=</mo><msqrt><mstyle scriptlevel="0" displaystyle="false"><msub><mo>∑</mo><mrow><mi>i</mi><mo>&gt;</mo><mi>k</mi></mrow></msub><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup></mstyle></msqrt></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**a rank-2 "image"**

On a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>50</mn><mo>×</mo><mn>50</mn></mrow></math></span> grid, brightness <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sin</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mi>cos</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>+</mo><mn>0.5</mn><mtext> </mtext><mi>x</mi><mi>y</mi></mrow></math></span> is a sum of two rank-1 patterns, so it has rank 2.

Keeping 2 singular values reproduces it essentially exactly from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mo stretchy="false">(</mo><mn>50</mn><mo>+</mo><mn>50</mn><mo stretchy="false">)</mo><mo>=</mo><mn>200</mn></mrow></math></span> numbers instead of 2,500.

</div>

```
x = np.linspace(0, 3, 50)
M = np.outer(np.sin(x), np.cos(x)) + 0.5 * np.outer(x, x)
U, s, Vt = np.linalg.svd(M)
for k in (1, 2):
    Mk = U[:, :k] * s[:k] @ Vt[:k]
    print(k, np.isclose(np.linalg.norm(M - Mk, 2), s[k]))
```

Both lines print `True`: the measured error equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow></math></span>.

## Decay decides compressibility

~~~dg-raw
%%viz f_curve f=Math.pow(0.7,x) g=1-x/40 x=1,30 y=0,1 tk=5 fl=fast_decay gl=slow_decay | Two singular value profiles, scaled so σ₁ = 1. With fast decay a few pieces carry almost everything; with slow decay (as for random matrices) every piece matters.
~~~

Natural images, embedding tables and many trained weight matrices have fast-decaying singular values. Random noise matrices do not, which is why noise does not compress.

<div class="take">
<div class="lab">Key takeaway</div>

truncating the SVD is the best way to compress a matrix to rank <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span>, and the first dropped singular value is exactly the spectral error.

</div>

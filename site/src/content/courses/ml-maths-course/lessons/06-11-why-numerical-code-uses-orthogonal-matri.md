---
id: "06.11"
course: "ml-maths-course"
module: "06"
slug: "why-numerical-code-uses-orthogonal-matrices"
title: "Why Numerical Code Uses Orthogonal Matrices"
keyIdeas: ["Multiplying by an orthogonal ⟦571⟧ never magnifies errors: ⟦1996⟧.", "Algorithms built from orthogonal steps (Householder QR, the SVD) are backward stable.", "Forming ⟦1997⟧ squares the condition number and loses about twice as many digits."]
question: "⟦1994⟧ in float64 (about 16 digits). Roughly how many correct digits can you expect when solving least squares via QR, and via the normal equations ⟦1995⟧?"
answer: "Step 1, digits lost is about ⟦1177⟧. Step 2, QR works with ⟦1994⟧: lose about 5, keep about 11. Step 3, the normal equations work with ⟦1999⟧: lose about 10, keep about 6."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Orthogonal structure"]
tools: ["NumPy"]
topic: {"id": "06.02", "title": "Orthogonal structure"}
---

## The big picture

Rounding errors are unavoidable. What matters is whether each step of an algorithm amplifies them. Orthogonal matrices have condition number exactly 1, so algorithms made of orthogonal steps never amplify. That is why the standard library routines are built from them.

## Condition numbers are unchanged

Because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>Q</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi></mrow></math></span>, multiplying by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> stretches nothing, so
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>κ</mi><mo stretchy="false">(</mo><mi>Q</mi><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mi>κ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span></span>
An algorithm that transforms <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> by a sequence of orthogonal matrices keeps the problem exactly as well-conditioned as it started.

~~~dg-flow Orthogonal steps keep the condition number; forming A-transpose-A squares it.
A -(Householder QR)-> *R with κ(R) = κ(A)
A -(form Aᵀ A)-> κ squared
~~~

```
A = np.array([[1.0, 1], [1, 1.001], [1, 0.999]])
Q, R = np.linalg.qr(A)
print(round(np.linalg.cond(R) / np.linalg.cond(A), 6))  # 1.0
print(round(np.linalg.cond(A.T @ A) / np.linalg.cond(A) ** 2, 6))  # 1.0
```

The first line: QR keeps the condition number. The second: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi></mrow></math></span> has its square.

## Backward stability

An algorithm is **backward stable** if its answer is the exact answer to a slightly changed input. Householder QR, the SVD and the QR eigenvalue algorithm all are. Gaussian elimination without pivoting and explicit inversion are not reliably so.

<div class="ex">
<div class="lab">Worked example</div>

**digits lost**

Rule of thumb: you lose about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mrow><mi>log</mi><mo>⁡</mo></mrow><mn>10</mn></msub><mi>κ</mi></mrow></math></span> of your ~16 float64 digits.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><msup><mn>0</mn><mn>5</mn></msup></mrow></math></span>: QR leaves about 11 good digits; the normal equations about 6.

In float32 (about 7 digits) the normal equations would leave nothing.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

orthogonal steps keep the condition number, so routines like QR and SVD are stable; avoid forming <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi></mrow></math></span> when accuracy matters.

</div>

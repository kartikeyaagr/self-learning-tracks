---
id: "08.02"
course: "ml-maths-course"
module: "08"
slug: "solving-least-squares-well"
title: "Solving Least Squares Well"
keyIdeas: ["The normal equations square the condition number; solve with QR or the SVD (`np.linalg.lstsq`) instead.", "With dependent columns there are infinitely many solutions; `lstsq` returns the minimum-norm one, ⟦2307⟧.", "Squared error matches Gaussian noise and is sensitive to outliers; L1 and Huber losses are robust alternatives."]
question: "Why prefer `np.linalg.lstsq(X, y)` over `np.linalg.inv(X.T @ X) @ X.T @ y`? What does each do if one column of ⟦513⟧ duplicates another?"
answer: "Step 1, forming ⟦895⟧ squares ⟦1792⟧: ⟦1564⟧ becomes ⟦1794⟧, losing about 12 of 16 digits instead of 6. Step 2, with a duplicated column, ⟦895⟧ is singular: `inv` raises an error or, after rounding, returns huge meaningless numbers. Step 3, `lstsq` detects the reduced rank and returns the minimum-norm solution, splitting the weight equally between the two copies. Step 4, the fitted values ⟦2348⟧ are the same in every case; only the coefficients are ambiguous."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Residual minimization"]
tools: ["NumPy"]
topic: {"id": "08.01", "title": "Residual minimization"}
---

## The big picture

The normal equations are right on paper and risky in floating point. This lesson covers how to compute least squares reliably, what to do when the answer is not unique, and when squared error is the wrong loss.

## Three routes

~~~dg-flow Three ways to compute the same least-squares solution.
[Normal equations] form Aᵀ A -(Cholesky)-> fast, but condition number squared
[QR] A = QR -(solve R x = Qᵀ b)-> *standard choice, stable
[SVD] A = U S Vᵀ -(x = V S^+ Uᵀ b)-> slowest, handles rank deficiency
~~~

```
x = np.array([0.0, 1, 2, 3]); y = np.array([1.0, 2, 2, 4])
A = np.column_stack([np.ones(4), x])
Q, R = np.linalg.qr(A)
print(np.round(np.linalg.solve(R, Q.T @ y), 6))  # [0.9 0.9]
```

For very large or sparse problems, iterative solvers (LSQR, conjugate gradient) or SGD avoid factoring anything.

## When the solution is not unique

With dependent columns you can add any null-space vector to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> without changing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span>. The minimum-norm solution <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mo>+</mo></msup><mi>b</mi></mrow></math></span> (lesson 07.12) has no null-space part.

<div class="ex">
<div class="lab">Worked example</div>

**two identical columns**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> has columns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>. Every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>2</mn></mrow></math></span> fits perfectly.

The shortest is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mo>−</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span> are longer.

</div>

```
A = np.ones((3, 2)); b = np.full(3, 2.0)
sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
print(np.round(sol, 6), rank)  # [1. 1.] 1
```

Gradient descent started at zero converges to this same minimum-norm solution, because every gradient <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> lies in the row space. It is one simple model of "implicit regularisation".

## Choosing the loss

~~~dg-raw
%%viz f_curve f=x*x g=Math.abs(x) x=-3,3 y=0,9 tk=1 fl=squared:_r² gl=absolute:_∣r∣ | Penalty for a residual r. Squared error charges 9 for a miss of 3 and 100 for a miss of 10, so one outlier can dominate the fit. Absolute error grows only linearly.
~~~

| Loss on residual <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> | Behaviour | Noise it assumes |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>r</mi><mn>2</mn></msup></mrow></math></span> | closed form; outliers dominate | Gaussian |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">∣</mo><mi>r</mi><mo stretchy="false">∣</mo></mrow></math></span> | fits a median; robust | Laplace |
| Huber | quadratic near 0, linear in the tails | Gaussian core, heavy tails |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mi>i</mi></msub><msubsup><mi>r</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span> | trusts some rows more | known variances <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msub><mi>w</mi><mi>i</mi></msub></mrow></math></span> |

<div class="ex">
<div class="lab">Worked example</div>

**mean versus median**

Fit one constant <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span> to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>100</mn></mrow></math></span>.

Squared error picks the mean, 26.5; the outlier drags it far away.

Absolute error picks a median, anything from 2 to 3.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

compute least squares with QR or SVD, expect the minimum-norm answer when columns are dependent, and switch to a robust loss when outliers matter.

</div>

## Where this breaks in practice

- Unscaled features (metres next to millimetres) make <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> badly conditioned. Standardise the columns first; fitted values do not change.

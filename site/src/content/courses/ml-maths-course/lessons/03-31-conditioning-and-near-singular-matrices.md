---
id: "03.31"
course: "ml-maths-course"
module: "03"
slug: "conditioning-and-near-singular-matrices"
title: "Conditioning and Near-Singular Matrices"
keyIdeas: ["The condition number ⟦1176⟧ measures how much ⟦8⟧ can amplify relative errors.", "Rule of thumb: you lose about ⟦1177⟧ digits of accuracy.", "Least squares via ⟦895⟧ squares the condition number; QR or SVD on ⟦513⟧ avoids that."]
question: "⟦895⟧ in a regression has eigenvalues from ⟦1174⟧ down to ⟦1175⟧. What is its condition number, roughly how many digits can you trust in float64, and what are two fixes?"
answer: "Step 1, ⟦1192⟧. Step 2, float64 has about 16 digits, so about ⟦1193⟧ are trustworthy. Step 3, fixes: solve with QR or SVD on ⟦513⟧ instead of forming ⟦895⟧ (that avoids squaring ⟦1190⟧); standardise the features; or add ridge regularisation ⟦1194⟧, which lifts the smallest eigenvalue to at least ⟦559⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Matrix invertibility"]
tools: ["NumPy"]
topic: {"id": "03.04", "title": "Matrix invertibility"}
---

## The big picture

Invertible is a yes-or-no question. The useful question in practice is "how easily can it be undone?". A matrix that almost squashes a direction turns tiny rounding errors into huge errors in the answer. The condition number measures this.

## Nearly parallel lines

<div class="ex">
<div class="lab">Worked example</div>

**a small change, a big jump**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>2</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mn>1.0001</mn><mi>y</mi><mo>=</mo><mn>2.0001</mn></mrow></math></span>. Solution: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

Change the second right-hand side to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2.0002</mn></mrow></math></span>. Now <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>2</mn></mrow></math></span>. A 0.005% change in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> moved the answer by 100%.

</div>

~~~dg-raw
%%viz f_plane x=-1,3 y=-1,3 ln=1,1,2,x+y=2,a,0.12;1,1.05,2.05,second_line,f,0.88 pt=1,1,solution | Two nearly parallel lines. Nudge either one slightly and the crossing point slides a long way along them.
~~~

```
A = np.array([[1, 1], [1, 1.0001]])
print(np.round(np.linalg.solve(A, [2, 2.0001]), 6))  # [1. 1.]
print(np.round(np.linalg.solve(A, [2, 2.0002]), 6))  # [0. 2.]
print(round(np.linalg.cond(A)))  # 40002
```

## The condition number

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> has a small relative error, the relative error in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> can be up to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> times bigger:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mrow><mi mathvariant="normal">∥</mi><mi>δ</mi><mi>x</mi><mi mathvariant="normal">∥</mi></mrow><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi></mrow></mfrac><mo>≤</mo><mi>κ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mtext> </mtext><mfrac><mrow><mi mathvariant="normal">∥</mi><mi>δ</mi><mi>b</mi><mi mathvariant="normal">∥</mi></mrow><mrow><mi mathvariant="normal">∥</mi><mi>b</mi><mi mathvariant="normal">∥</mi></mrow></mfrac><mo separator="true">,</mo><mspace width="2em"/><mi>κ</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><msub><mi>σ</mi><mi>max</mi><mo>⁡</mo></msub><msub><mi>σ</mi><mi>min</mi><mo>⁡</mo></msub></mfrac></mrow></math></span></span>
<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>min</mi><mo>⁡</mo></msub></mrow></math></span> are the largest and smallest factors by which <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> stretches any direction (its singular values, module 07).

| condition number | digits left in float64 | digits left in float32 |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>2</mn></msup></mrow></math></span> | about 14 | about 5 |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>8</mn></msup></mrow></math></span> | about 8 | none |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>14</mn></msup></mrow></math></span> | about 2 | none |

## Normal equations square the problem

Least squares via <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mi>β</mi><mo>=</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span> faces <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mo stretchy="false">)</mo><mo>=</mo><mi>κ</mi><mo stretchy="false">(</mo><mi>X</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span>.

```
X = np.array([[1, 1], [1, 1.001], [1, 1.002]])
print(round(np.linalg.cond(X)), round(np.linalg.cond(X.T @ X)))  # 2452 6012014
```

QR or SVD works on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> directly and faces only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo></mrow></math></span>.

## Where this breaks in practice

- `np.linalg.inv` returns huge numbers for a nearly singular matrix without any error. Check `np.linalg.cond` when results look strange.
- **Jitter.** Gaussian-process code adds <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup><mi>I</mi></mrow></math></span> before Cholesky. If you need <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>2</mn></mrow></msup></mrow></math></span> to make it run, look for duplicate points or a length scale that is too long.

<div class="take">
<div class="lab">Key takeaway</div>

ask how well conditioned a matrix is, not just whether it is invertible. Avoid forming <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span>.

</div>

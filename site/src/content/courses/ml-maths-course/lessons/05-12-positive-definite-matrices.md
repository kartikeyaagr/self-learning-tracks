---
id: "05.12"
course: "ml-maths-course"
module: "05"
slug: "positive-definite-matrices"
title: "Positive Definite Matrices"
keyIdeas: ["Symmetric ⟦8⟧ is positive definite if ⟦1761⟧ for all ⟦597⟧, equivalently all ⟦1762⟧.", "Positive semidefinite (PSD): ⟦1763⟧, all ⟦1764⟧. Indefinite: eigenvalues of both signs.", "Covariances, Gram matrices ⟦895⟧ and Hessians at minima are PSD. Cholesky ⟦1109⟧ exists exactly for positive definite ⟦8⟧."]
question: "Classify ⟦1758⟧ and ⟦1759⟧. Give an ⟦7⟧ with ⟦1760⟧ if one exists."
answer: "Step 1, first matrix: trace 2, determinant ⟦23⟧, eigenvalues 3 and ⟦499⟧. Indefinite. Step 2, ⟦1774⟧: ⟦1775⟧. Step 3, second matrix: eigenvalues 3 and 1, positive definite."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

"Is this point a minimum?", "Is this a valid covariance?", "Will Cholesky work?" all ask the same question: is a symmetric matrix positive definite? The eigenvalues answer it at once.

## Quadratic forms

For symmetric <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi></mrow></math></span> is a **quadratic form**. In the eigenbasis it becomes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∑</mo><mi>i</mi></msub><msub><mi>λ</mi><mi>i</mi></msub><msubsup><mi>c</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span>, where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mi>i</mi></msub></mrow></math></span> are the coordinates of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>. So its sign is controlled by the eigenvalues:

- **Positive definite**: all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>i</mi></msub><mo>&gt;</mo><mn>0</mn></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>. A bowl.
- **Positive semidefinite**: all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>i</mi></msub><mo>≥</mo><mn>0</mn></mrow></math></span>. A bowl that may be flat in some directions.
- **Indefinite**: mixed signs. A saddle.

<div class="ex">
<div class="lab">Worked example</div>

**bowl or saddle?**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mi mathvariant="normal">d</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">a</mi><mi mathvariant="normal">g</mi></mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><mn>3</mn><msubsup><mi>x</mi><mn>2</mn><mn>2</mn></msubsup><mo>&gt;</mo><mn>0</mn></mrow></math></span>. Bowl.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mi mathvariant="normal">d</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">a</mi><mi mathvariant="normal">g</mi></mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mn>1</mn></mrow></math></span>. Saddle.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: eigenvalues 3 and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span>. Saddle, though every entry is positive.

</div>

```
for M in ([[2, 0], [0, 3]], [[2, 0], [0, -1]], [[1, 2], [2, 1]]):
    print(np.linalg.eigvalsh(np.array(M, float)))
```

## Where PSD matrices come from

| matrix | why it is PSD |
| --- | --- |
| Covariance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><mi mathvariant="normal">Σ</mi><mi>v</mi><mo>=</mo><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mo stretchy="false">)</mo><mo>≥</mo><mn>0</mn></mrow></math></span> |
| Gram or kernel matrix | by definition of a valid kernel |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mi>v</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi>X</mi><mi>v</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>≥</mo><mn>0</mn></mrow></math></span> |
| Hessian at a local minimum | second-order condition |

## Cholesky as a test

Positive definite matrices have <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>L</mi><msup><mi>L</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi></mrow></math></span> lower triangular. `np.linalg.cholesky` raises an error otherwise:

```
for M in ([[2.0, 1], [1, 2]], [[1.0, 2], [2, 1]]):
    try:
        np.linalg.cholesky(np.array(M)); print('positive definite')
    except np.linalg.LinAlgError:
        print('not positive definite')
```

<div class="take">
<div class="lab">Key takeaway</div>

positive definite means all eigenvalues positive: a bowl. Check with `eigvalsh` or try Cholesky.

</div>

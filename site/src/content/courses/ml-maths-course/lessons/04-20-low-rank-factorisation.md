---
id: "04.20"
course: "ml-maths-course"
module: "04"
slug: "low-rank-factorisation"
title: "Low-Rank Factorisation"
keyIdeas: ["Any rank-⟦118⟧ matrix factors as ⟦1555⟧ with ⟦207⟧ of size ⟦1556⟧ and ⟦1204⟧ of size ⟦1557⟧.", "Storage and matrix-vector cost drop from ⟦1558⟧ to ⟦1559⟧.", "LoRA, recommender systems and low-rank compression all use this."]
question: "LoRA adapts a ⟦1530⟧ weight with ⟦1551⟧, ⟦1552⟧, ⟦1553⟧. What is the maximum rank of ⟦1554⟧, and how many trainable parameters does it have versus a full update?"
answer: "Step 1, ⟦1571⟧ because ⟦41⟧ has 16 columns. Step 2, parameters: ⟦1572⟧. Step 3, a full update has ⟦1573⟧, which is 128 times more. Step 4, LoRA can't represent any update of rank above 16."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Dimension, rank and nullity"]
tools: ["NumPy"]
topic: {"id": "04.04", "title": "Dimension, rank and nullity"}
---

## The big picture

A low-rank matrix is secretly a product of two thin matrices. Storing and multiplying the thin factors is far cheaper than working with the full matrix. This one idea is behind LoRA fine-tuning, matrix-factorisation recommenders, and compressing embedding tables.

## The factorisation

Any rank-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> matrix can be written as
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><mi>U</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup><mo separator="true">,</mo><mspace width="2em"/><mi>U</mi><mtext> is </mtext><mi>m</mi><mo>×</mo><mi>r</mi><mo separator="true">,</mo><mspace width="1em"/><mi>V</mi><mtext> is </mtext><mi>n</mi><mo>×</mo><mi>r</mi></mrow></math></span></span>
equivalently as a sum of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> outer products: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><msub><mi>u</mi><mn>1</mn></msub><msubsup><mi>v</mi><mn>1</mn><mi mathvariant="normal">⊤</mi></msubsup><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>u</mi><mi>r</mi></msub><msubsup><mi>v</mi><mi>r</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span>.

## The saving

Storage drops from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mi>n</mi></mrow></math></span> numbers to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo stretchy="false">(</mo><mi>m</mi><mo>+</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span>. A matrix-vector product drops from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mi>n</mi></mrow></math></span> to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo stretchy="false">(</mo><mi>m</mi><mo>+</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span> operations: compute <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> first (only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> numbers), then multiply by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**counting the saving**

A <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1000</mn><mo>×</mo><mn>1000</mn></mrow></math></span> matrix of rank 10 has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>6</mn></msup></mrow></math></span> entries.

As <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> it needs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>10</mn><mo>×</mo><mn>2000</mn><mo>=</mo><mn>20</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span> numbers, 50 times less.

</div>

```
rng = np.random.default_rng(1)
U = rng.normal(size=(1000, 10)); V = rng.normal(size=(1000, 10))
A = U @ V.T; x = rng.normal(size=1000)
print(np.allclose(A @ x, U @ (V.T @ x)))  # True
print(A.size, U.size + V.size)  # 1000000 20000
```

## In ML

| use | shape | rank | saving |
| --- | --- | --- | --- |
| LoRA on one attention projection | 4096 by 4096 | 16 | 128 times fewer trainable parameters |
| Recommender ratings | 1M users by 100k items | 64 | about 70M numbers instead of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>11</mn></msup></mrow></math></span> |
| Truncated SVD of embeddings | 50k by 768 | 128 | about 5.9 times smaller |

<div class="ex">
<div class="lab">Worked example</div>

**a recommender**

Ratings matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi></mrow></math></span>: users by items. Model it as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mo>≈</mo><mi>U</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>=</mo><mn>2</mn></mrow></math></span>.

Each user gets 2 numbers (say "likes action", "likes romance"), each item gets 2 numbers. A predicted rating is a dot product of a user row and an item row.

</div>

## Where this breaks in practice

- **Not every update is low rank.** If the ideal change has important singular values beyond the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span>-th, a rank-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> factorisation captures only its best rank-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> part (module 07).

<div class="take">
<div class="lab">Key takeaway</div>

rank <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> means the matrix is a product of two thin factors. Store and multiply the factors.

</div>

---
id: "04.19"
course: "ml-maths-course"
module: "04"
slug: "the-rank-nullity-theorem"
title: "The Rank-Nullity Theorem"
keyIdeas: ["For an ⟦961⟧ matrix, ⟦1543⟧.", "Each input dimension is either passed on (rank) or thrown away (null space).", "It tells you how many free variables a system has without solving it."]
question: "A ⟦1542⟧ matrix has rank 4. What are the dimensions of its null space, column space and left null space?"
answer: "Step 1, rank-nullity: ⟦1549⟧. Step 2, column space: dimension equals the rank, 4. Step 3, left null space: ⟦1550⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Dimension, rank and nullity"]
tools: ["NumPy"]
topic: {"id": "04.04", "title": "Dimension, rank and nullity"}
---

## The big picture

Rank-nullity is bookkeeping for input dimensions. A matrix takes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> input dimensions; each one either makes it through to the output or is destroyed. The theorem says nothing else can happen, and it gives the count without any solving.

## The theorem

For an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>+</mo><mi>dim</mi><mo>⁡</mo><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mi>n</mi></mrow></math></span></span>
In elimination terms: rank is the number of pivot columns, nullity is the number of free columns, and together they are all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> columns.

<div class="ex">
<div class="lab">Worked example</div>

**checking rank-nullity**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>6</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>3</mn></mrow></math></span> and rank 1.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>dim</mi><mo>⁡</mo><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mn>2</mn></mrow></math></span>. In lesson 04.08 we found the null space: the plane spanned by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>3</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Two vectors, as predicted.

</div>

```
A = np.array([[1.0, 2, 3], [2, 4, 6]])
r = np.linalg.matrix_rank(A)
_, s, Vt = np.linalg.svd(A)
print(r, A.shape[1] - r, Vt[r:].shape[0])  # 1 2 2
```

## Uses

| question | answer from rank-nullity |
| --- | --- |
| How many free variables in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>? | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> |
| Can a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1024</mn><mo>×</mo><mn>4096</mn></mrow></math></span> layer be injective? | No: nullity is at least <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3072</mn></mrow></math></span> |
| Is a square <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> invertible? | Exactly when nullity is 0 |

<div class="ex">
<div class="lab">Worked example</div>

**a bottleneck autoencoder**

A linear encoder of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>32</mn><mo>×</mo><mn>784</mn></mrow></math></span> has rank at most 32.

So its null space has dimension at least <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>784</mn><mo>−</mo><mn>32</mn><mo>=</mo><mn>752</mn></mrow></math></span>. At least 752 input directions are thrown away, whatever the decoder does.

</div>

## Where this breaks in practice

- **Nonlinear layers.** The theorem is about linear maps. A ReLU network has no single null space, but each linear piece obeys rank-nullity on its own region.

<div class="take">
<div class="lab">Key takeaway</div>

rank + nullity = number of columns. What isn't passed on is thrown away.

</div>

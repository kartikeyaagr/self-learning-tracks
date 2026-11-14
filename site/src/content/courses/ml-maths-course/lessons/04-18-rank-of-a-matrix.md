---
id: "04.18"
course: "ml-maths-course"
module: "04"
slug: "rank-of-a-matrix"
title: "Rank of a Matrix"
keyIdeas: ["The rank of ⟦8⟧ is the dimension of its column space: the number of independent columns.", "Column rank always equals row rank; rank also equals the number of pivots and the number of non-zero singular values.", "⟦1529⟧: a bottleneck caps everything after it."]
question: "Find the rank of ⟦1525⟧ and of the outer product ⟦1526⟧ with ⟦1527⟧, ⟦1528⟧."
answer: "Step 1, column 3 = column 1 + column 2, and columns 1 and 2 are independent, so rank 2. Step 2, every column of ⟦1526⟧ is a multiple of ⟦280⟧ (column ⟦82⟧ is ⟦1541⟧), so rank 1."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Dimension, rank and nullity"]
tools: ["NumPy"]
topic: {"id": "04.04", "title": "Dimension, rank and nullity"}
---

## The big picture

A <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4096</mn><mo>×</mo><mn>4096</mn></mrow></math></span> matrix has 16.7 million numbers, but if every column is a mix of the same 16 columns, it is secretly much simpler. **Rank** measures this hidden simplicity. It is why LoRA can fine-tune huge models cheaply and why a narrow layer limits everything after it.

## The definition

The **rank** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the dimension of its column space: the number of independent columns. A surprising theorem says it also equals the number of independent **rows**.

<div class="ex">
<div class="lab">Worked example</div>

**rank by inspection**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: the columns point different ways. Rank 2.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>6</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: every column is a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>. Rank 1. Row 2 is twice row 1, so the rows agree.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: column 3 = column 1 + column 2. Rank 2.

</div>

```
for M in ([[1, 2], [3, 4]], [[1, 2, 3], [2, 4, 6]], [[1, 0, 1], [0, 1, 1], [1, 1, 2]]):
    print(np.linalg.matrix_rank(np.array(M)))
```

## Outer products are rank one

For column vectors <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> has entry <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></math></span> equal to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mi>i</mi></msub><msub><mi>v</mi><mi>j</mi></msub></mrow></math></span>. Every column is a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span>.

```
u = np.array([1, 2, 3]); v = np.array([1, 0, -1, 2])
M = np.outer(u, v)
print(M.shape, np.linalg.matrix_rank(M))  # (3, 4) 1
```

## Facts to remember

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>≤</mo><mi>min</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><mo separator="true">,</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span>. If equal, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> has **full rank**.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><mo stretchy="false">)</mo><mo>≤</mo><mi>min</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mi>A</mi><mo separator="true">,</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><mi>B</mi><mo stretchy="false">)</mo><mo>≤</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mi>A</mi><mo>+</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mi>B</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**a bottleneck**

A linear encoder <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>E</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>32</mn><mo>×</mo><mn>784</mn></mrow></math></span>) followed by a decoder <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>D</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>784</mn><mo>×</mo><mn>32</mn></mrow></math></span>).

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">rank</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>D</mi><mi>E</mi><mo stretchy="false">)</mo><mo>≤</mo><mn>32</mn></mrow></math></span>: the output of the whole map lives in a 32-dimensional subspace of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>784</mn></msup></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
E = rng.normal(size=(32, 784)); D = rng.normal(size=(784, 32))
print(np.linalg.matrix_rank(D @ E))  # 32
```

<div class="take">
<div class="lab">Key takeaway</div>

rank counts independent columns, which always equals independent rows. A product can't have more rank than its narrowest factor.

</div>

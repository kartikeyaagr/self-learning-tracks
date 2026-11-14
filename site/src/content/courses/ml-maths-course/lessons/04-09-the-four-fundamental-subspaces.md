---
id: "04.09"
course: "ml-maths-course"
module: "04"
slug: "the-four-fundamental-subspaces"
title: "The Four Fundamental Subspaces"
keyIdeas: ["An ⟦961⟧ matrix of rank ⟦118⟧ has four subspaces: column space and left null space in ⟦1394⟧, row space and null space in ⟦1377⟧.", "Their dimensions are ⟦118⟧, ⟦1419⟧, ⟦118⟧ and ⟦1038⟧.", "The row space is perpendicular to the null space, and the column space to the left null space."]
question: "For ⟦1407⟧, give the dimension of each of the four subspaces."
answer: "Step 1, ⟦1427⟧, ⟦1428⟧, rank ⟦1429⟧. Step 2, column space: dimension 1 (the line through ⟦1049⟧). Left null space: ⟦1136⟧. Step 3, row space: dimension 1 (the line through ⟦1366⟧). Null space: ⟦1430⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Subspaces, columns and null spaces"]
tools: ["NumPy"]
topic: {"id": "04.02", "title": "Subspaces, columns and null spaces"}
---

## The big picture

The column space and null space are two of four subspaces that every matrix carries. Together they give a complete picture of what a matrix does: which input directions it keeps, which it kills, which outputs it reaches and which it can never reach.

## The four

| Subspace | Lives in | Dimension |
| --- | --- | --- |
| Column space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span> (outputs) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> |
| Left null space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span> (outputs) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>−</mo><mi>r</mi></mrow></math></span> |
| Row space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> (inputs) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> |
| Null space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> (inputs) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>r</mi></mrow></math></span> |

The row space is the span of the rows. The left null space is the null space of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>.

## Perpendicular pairs

- Every row-space vector is perpendicular to every null-space vector: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> literally says each row dotted with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is 0. Together they fill <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span>.
- Likewise the column space and the left null space are perpendicular and fill <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span>.

## What a matrix does to an input

~~~dg-flow A splits the input space: the row-space part is mapped one-to-one, the null-space part is sent to zero.
x = x_row + x_null -(A)-> *A x_row in C(A)
x_null -(A)-> 0
~~~

<div class="ex">
<div class="lab">Worked example</div>

**splitting an input**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>6</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>. The row space is the line through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> has row-space part <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>6</mn><mn>14</mn></mfrac><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo><mo>≈</mo><mo stretchy="false">(</mo><mn>0.43</mn><mo separator="true">,</mo><mn>0.86</mn><mo separator="true">,</mo><mn>1.29</mn><mo stretchy="false">)</mo></mrow></math></span>. The rest, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.57</mn><mo separator="true">,</mo><mn>0.14</mn><mo separator="true">,</mo><mo>−</mo><mn>0.29</mn><mo stretchy="false">)</mo></mrow></math></span>, is in the null space.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> sends <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and its row-space part to the same output, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>6</mn><mo separator="true">,</mo><mn>12</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
A = np.array([[1.0, 2, 3], [2, 4, 6]]); x = np.array([1.0, 1, 1])
r = np.array([1.0, 2, 3]); xr = (x @ r) / (r @ r) * r
print(A @ x, np.round(A @ xr, 6))  # [ 6. 12.] [ 6. 12.]
print(round((x - xr) @ r, 10))  # 0.0
```

## Why it matters

Gilbert Strang's MIT course 18.06 is built around this picture. In ML, it explains what a linear layer can see (row space), what it ignores (null space), and what it can output (column space).

<div class="take">
<div class="lab">Key takeaway</div>

four subspaces, two in each space, in perpendicular pairs. A matrix maps the row space one-to-one onto the column space and kills the null space.

</div>

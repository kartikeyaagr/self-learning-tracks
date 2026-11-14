---
id: "04.02"
course: "ml-maths-course"
module: "04"
slug: "matrix-vector-products-as-column-mixes"
title: "Matrix-Vector Products as Column Mixes"
keyIdeas: ["Put ⟦1299⟧ as the columns of ⟦1204⟧ and the weights in ⟦119⟧. Then ⟦1314⟧.", "So ⟦839⟧ is a mix of the columns of ⟦8⟧, weighted by the entries of ⟦7⟧.", "The row view (dot product of each row with ⟦7⟧) gives the same numbers; the column view explains what outputs are possible."]
question: "Compute ⟦1313⟧ in both the row view and the column view."
answer: "Step 1, row view: row 1 dotted with ⟦731⟧ is ⟦1322⟧; row 2 is ⟦1323⟧. Result ⟦1324⟧. Step 2, column view: ⟦1325⟧. Step 3, same answer, two readings."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Vectors, spans and dependence"]
tools: ["NumPy"]
topic: {"id": "04.01", "title": "Vectors, spans and dependence"}
---

## The big picture

There are two ways to read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span>. The row view, "dot each row with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>", is how you compute it by hand. The column view, "mix the columns using the entries of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> as weights", is how you understand it. The column view is the single most useful idea in this module.

## The column view

Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> by its columns, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">[</mo><msub><mi>a</mi><mn>1</mn></msub><mtext> </mtext><msub><mi>a</mi><mn>2</mn></msub><mtext> </mtext><mo>⋯</mo><mtext> </mtext><msub><mi>a</mi><mi>n</mi></msub><mo stretchy="false">]</mo></mrow></math></span>. Then
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub><msub><mi>a</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><msub><mi>a</mi><mn>2</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>x</mi><mi>n</mi></msub><msub><mi>a</mi><mi>n</mi></msub></mrow></math></span></span>
Each entry of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> says how much of the matching column to use.

```
V = np.array([[1, 1], [0, 2]])
print(V @ np.array([3, 1]))  # [4 2]
print(3 * V[:, 0] + 1 * V[:, 1])  # [4 2]
```

## The row view

Entry <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span> is row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> dotted with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>. Same numbers, computed differently.

<div class="ex">
<div class="lab">Worked example</div>

**what a layer can output**

A layer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mi>W</mi><mi>x</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> takes 2 numbers in and gives 3 out.

Column view: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. The second column is twice the first, so every output is a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

However you choose the input, the output stays on one line in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span>. It can never produce <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

## Matrix times matrix

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>B</mi></mrow></math></span> applies the column view to each column of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>: column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>B</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> times column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>, a mix of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>'s columns.

```
A = np.array([[1, 1], [0, 2]]); B = np.array([[3, 0], [1, 1]])
print((A @ B)[:, 0], A @ B[:, 0])  # [4 2] [4 2]
```

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span> is a weighted mix of the columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>. That tells you at once which outputs are possible.

</div>

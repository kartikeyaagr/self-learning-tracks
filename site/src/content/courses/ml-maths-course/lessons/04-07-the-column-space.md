---
id: "04.07"
course: "ml-maths-course"
module: "04"
slug: "the-column-space"
title: "The Column Space"
keyIdeas: ["The column space ⟦1388⟧ is the span of ⟦8⟧'s columns: the set of every possible output ⟦839⟧.", "⟦816⟧ has a solution exactly when ⟦1391⟧.", "When ⟦1392⟧, least squares finds the closest point of ⟦1388⟧ to ⟦682⟧."]
question: "⟦1387⟧. Describe ⟦1388⟧ and decide whether ⟦1389⟧ and ⟦1390⟧ have solutions."
answer: "Step 1, outputs are ⟦1401⟧: the third entry is the sum of the first two. Step 2, ⟦1326⟧: ⟦1340⟧, so it is in ⟦1388⟧, with ⟦1396⟧. Step 3, ⟦1329⟧: ⟦1398⟧, not in ⟦1388⟧, no solution."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Subspaces, columns and null spaces"]
tools: ["NumPy"]
topic: {"id": "04.02", "title": "Subspaces, columns and null spaces"}
---

## The big picture

A matrix takes inputs and produces outputs. The first question is: **what outputs can it produce at all?** The answer is the column space. It decides whether a system is solvable, and what a layer's output is limited to.

## The definition

For an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, the **column space** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> is the span of its <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> columns. It lives in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span>, the output space. Since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span> is a mix of the columns, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> is exactly the set of all outputs:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><mi>A</mi><mi>x</mi><mo>:</mo><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo stretchy="false">}</mo></mrow></math></span></span>
So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> has a solution **exactly when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> is in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span>**.

<div class="ex">
<div class="lab">Worked example</div>

**a reachable and an unreachable target**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> outputs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>2</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> works with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>7</mn><mo stretchy="false">)</mo></mrow></math></span> does not, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>+</mo><mn>3</mn><mo mathvariant="normal">≠</mo><mn>7</mn></mrow></math></span>.

</div>

## The rank test in code

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>∈</mo><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> exactly when adding <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> as an extra column doesn't raise the rank:

```
A = np.array([[1, 0], [0, 1], [1, 1]])
for b in ([2, 3, 5], [2, 3, 7]):
    print(np.linalg.matrix_rank(np.column_stack([A, b])) == np.linalg.matrix_rank(A))
```

## When the target is outside

Least squares (module 08) finds the point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> of the column space closest to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span>. The leftover <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>−</mo><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> is perpendicular to the whole column space.

```
A = np.array([[1.0, 0], [0, 1], [1, 1]]); b = np.array([2.0, 3, 7])
xh, *_ = np.linalg.lstsq(A, b, rcond=None)
print(np.round(A @ xh, 4))  # [2.6667 3.6667 6.3333]
print(np.abs(A.T @ (b - A @ xh)).max() < 1e-10)  # True
```

<div class="take">
<div class="lab">Key takeaway</div>

the column space is the set of everything <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> can output. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> is solvable exactly when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> is in it.

</div>

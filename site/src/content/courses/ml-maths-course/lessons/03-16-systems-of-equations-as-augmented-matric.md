---
id: "03.16"
course: "ml-maths-course"
module: "03"
slug: "systems-of-equations-as-augmented-matrices"
title: "Systems of Equations as Augmented Matrices"
keyIdeas: ["A system of ⟦958⟧ equations in ⟦233⟧ unknowns is written ⟦816⟧: ⟦8⟧ holds the coefficients, ⟦7⟧ the unknowns, ⟦682⟧ the right-hand sides.", "The augmented matrix ⟦959⟧ puts ⟦682⟧ as an extra column. Each row is one equation.", "Every algorithm in this topic works on the rows of ⟦959⟧."]
question: "Write the augmented matrix of ⟦955⟧, ⟦956⟧, ⟦957⟧."
answer: "Step 1, list the coefficients of ⟦964⟧ in each equation, writing 0 for a missing variable. Step 2, rows: ⟦966⟧, ⟦967⟧, ⟦965⟧. Step 3, the third row has ⟦313⟧ coefficient 0 because ⟦313⟧ does not appear in ⟦957⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Solving linear systems is the workhorse of numerical computing: linear regression, Newton's method and Gaussian processes all end in "solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>". The first step is bookkeeping: strip away the variable names and keep only the numbers, in a grid you can operate on.

## From equations to a matrix

A system of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi></mrow></math></span> equations in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> unknowns
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtable rowspacing="0.25em" columnalign="right left" columnspacing="0em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><msub><mi>a</mi><mn>11</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>a</mi><mrow><mn>1</mn><mi>n</mi></mrow></msub><msub><mi>x</mi><mi>n</mi></msub></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mo>=</mo><msub><mi>b</mi><mn>1</mn></msub></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mtext> </mtext><mi><mi mathvariant="normal">⋮</mi><mpadded height="0em" voffset="0em"><mspace mathbackground="black" width="0em" height="1.5em"></mspace></mpadded></mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><msub><mi>a</mi><mrow><mi>m</mi><mn>1</mn></mrow></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>a</mi><mrow><mi>m</mi><mi>n</mi></mrow></msub><msub><mi>x</mi><mi>n</mi></msub></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mo>=</mo><msub><mi>b</mi><mi>m</mi></msub></mrow></mstyle></mtd></mtr></mtable></math></span></span>
is written <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span> coefficient matrix, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> the vector of unknowns, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> the right-hand sides.

Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> as an extra column next to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and you get the **augmented matrix** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mi>A</mi><mo>∣</mo><mi>b</mi><mo stretchy="false">]</mo></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**equations to matrix**

The system <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>5</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>−</mo><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span> becomes

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mi>A</mi><mo>∣</mo><mi>b</mi><mo stretchy="false">]</mo><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>5</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>, with a bar before the last column.

Row 1 is the first equation, row 2 the second. Column 1 holds the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> coefficients, column 2 the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> coefficients.

</div>

## Building it in code

```
A = np.array([[2, 1], [1, -1]]); b = np.array([5, 1])
Ab = np.column_stack([A, b])
print(Ab)
print(Ab.shape)  # (2, 3)
```

## Missing variables are zeros

If a variable does not appear in an equation, its coefficient is 0. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span> in a system with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo separator="true">,</mo><mi>z</mi></mrow></math></span> is the row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo>∣</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>. Forgetting the zero shifts every later number into the wrong column.

## Checking a candidate solution

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> solves the system when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span> equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> in every row:

```
A = np.array([[1, 2, 1], [2, 4, 3], [1, 1, 0]]); b = np.array([4, 9, 1])
print(A @ np.array([-1, 2, 1]))  # [4 9 1]
```

<div class="take">
<div class="lab">Key takeaway</div>

write the system as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mi>A</mi><mo>∣</mo><mi>b</mi><mo stretchy="false">]</mo></mrow></math></span>, one row per equation, one column per variable, with zeros for missing variables.

</div>

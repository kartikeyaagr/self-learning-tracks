---
id: "03.21"
course: "ml-maths-course"
module: "03"
slug: "reduced-row-echelon-form"
title: "Reduced Row Echelon Form"
keyIdeas: ["Reduced row echelon form (RREF) also scales each pivot to 1 and clears the entries above each pivot.", "Unlike REF, the RREF of a matrix is unique.", "The solution can be read off RREF directly, without back substitution."]
question: "Reduce ⟦1020⟧ to RREF and read off the solution of the system it represents."
answer: "Step 1, the pivots are already 1. Step 2, clear above the second pivot: R1 − 2·R2 ⟦1028⟧. Step 3, RREF is ⟦1029⟧. Step 4, read off: ⟦391⟧, ⟦831⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy", "SymPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Row echelon form still needs back substitution. Going one step further, until every pivot is 1 and is the only non-zero entry in its column, gives a form where the answer can be read directly. It is the form to use by hand, especially when there are infinitely many solutions.

## The definition

A matrix is in **reduced row echelon form** when it is in row echelon form and also:

1. Every pivot equals 1.
2. Every pivot is the only non-zero entry in its column.

## Getting there (Gauss-Jordan)

After reaching REF, divide each pivot row by its pivot, then use each pivot to clear the entries above it, working from the bottom pivot upward.

<div class="ex">
<div class="lab">Worked example</div>

**a 3 by 3 system**

REF: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>.

Scale: R1 ÷ 2 and R3 ÷ 2 give <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0.5</mn><mo>∣</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo>∣</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>.

Clear column 3: R2 − R3 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo>∣</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>, R1 − 0.5·R3 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0</mn><mo>∣</mo><mn>1.5</mn><mo stretchy="false">]</mo></mrow></math></span>.

Clear column 2: R1 − 0.5·R2 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo>∣</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>y</mi><mo>=</mo><mi>z</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

</div>

## In code

SymPy computes exact RREF with fractions:

```
import sympy
M = sympy.Matrix([[2, 1, 1, 4], [4, 3, 3, 10], [8, 7, 9, 24]])
R, pivots = M.rref()
print(R.tolist(), pivots)  # [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]] (0, 1, 2)
```

NumPy has no RREF function, because floating-point RREF is fragile; numerical code uses LU, QR or SVD instead.

## Why it is unique

Whatever row operations you use, the RREF comes out the same. That makes it a good way to compare two systems: they have the same solutions exactly when their augmented matrices have the same RREF (ignoring zero rows).

<div class="take">
<div class="lab">Key takeaway</div>

RREF has pivots equal to 1 with zeros above and below. Read the solution straight off it.

</div>

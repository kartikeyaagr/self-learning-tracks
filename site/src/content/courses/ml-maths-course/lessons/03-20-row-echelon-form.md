---
id: "03.20"
course: "ml-maths-course"
module: "03"
slug: "row-echelon-form"
title: "Row Echelon Form"
keyIdeas: ["A matrix is in row echelon form when each row's first non-zero entry (its pivot) is to the right of the pivot above, and zero rows are at the bottom.", "Gaussian elimination produces row echelon form for any matrix, square or not.", "The number of pivots is the rank."]
question: "Is ⟦1016⟧ in row echelon form? Where are the pivots?"
answer: "Step 1, row 1's first non-zero entry is in column 1. Row 2's is in column 3, further right. Row 3 is all zeros and is at the bottom. Step 2, so yes, it is in row echelon form. Step 3, the pivots are the 1 in column 1 and the 4 in column 3. There are 2 pivots, so the rank is 2."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Square systems with a unique solution are the easy case. Real systems can have more equations than unknowns, fewer, or redundant ones. **Row echelon form** is the shape elimination produces in every case, and reading it tells you everything about the solutions.

## The definition

A matrix is in **row echelon form** (REF) when:

1. Any all-zero rows are at the bottom.
2. The first non-zero entry of each row, its **pivot**, is strictly to the right of the pivot in the row above.

It looks like a staircase, possibly with wide steps.

| row | col 1 | col 2 | col 3 | col 4 |
| --- | --- | --- | --- | --- |
| 1 | **1** | 2 | 0 | 3 |
| 2 | 0 | 0 | **4** | 1 |
| 3 | 0 | 0 | 0 | 0 |

## Getting there

Run elimination column by column. If a column has no non-zero entry in the remaining rows, skip it and move to the next column; that column will have no pivot.

<div class="ex">
<div class="lab">Worked example</div>

**a column with no pivot**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>. R2 − 2·R1 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>.

Column 2 has nothing left below row 1, so the next pivot is in column 3.

REF: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>, pivots in columns 1 and 3.

</div>

## Rank

The number of pivots is the **rank** of the matrix: how many independent equations there really are.

```
M = np.array([[1, 2, 0, 3], [0, 0, 4, 1], [0, 0, 0, 0]])
print(np.linalg.matrix_rank(M))  # 2
print(np.linalg.matrix_rank(np.array([[1, 2, 1], [2, 4, 2]])))  # 1
```

The second matrix has a row that is twice the other, so only one row is independent.

## Where this breaks in practice

- **REF is not unique.** Different row operations give different staircases. The pivot columns and the rank are always the same, though.

<div class="take">
<div class="lab">Key takeaway</div>

row echelon form is a staircase of pivots. The number of pivots is the rank.

</div>

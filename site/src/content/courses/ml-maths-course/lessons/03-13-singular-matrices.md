---
id: "03.13"
course: "ml-maths-course"
module: "03"
slug: "singular-matrices"
title: "Singular Matrices"
keyIdeas: ["A square matrix is singular when ⟦890⟧: it squashes space into a lower dimension.", "Equivalent: the columns are linearly dependent, ⟦889⟧ has a non-zero solution, and ⟦8⟧ has no inverse.", "Different inputs land on the same output, so a singular map cannot be undone."]
question: "For ⟦888⟧, find a non-zero ⟦7⟧ with ⟦889⟧, and two different inputs with the same output."
answer: "Step 1, the second column is twice the first, so ⟦897⟧. That means ⟦891⟧ gives ⟦889⟧. Step 2, add it to any input: ⟦111⟧ and ⟦893⟧. Step 3, ⟦898⟧ and ⟦899⟧: the same output."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Determinants and singularity"]
tools: ["NumPy"]
topic: {"id": "03.02", "title": "Determinants and singularity"}
---

## The big picture

Some matrices destroy information. They flatten a whole direction to zero, so you can no longer tell which input produced a given output. These are the **singular** matrices, and recognising them is essential: a singular matrix in a solver means the problem has no unique answer.

## Squashed flat

When <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>0</mn></mrow></math></span>, the unit square is squashed into a line segment (or a point): area zero.

~~~dg-raw
%%viz f_plane x=-1,4 y=-1,7 tk=1 pg=0:0_1:0_1:1_0:1,m sg=0,0,3,6,a v=1,2,Ae₁;2,4,Ae₂,f | A = [[1, 2], [2, 4]] sends the whole unit square onto the segment from (0, 0) to (3, 6). Its area is 0.
~~~

## Four ways to say the same thing

For a square matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, these are equivalent:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>0</mn></mrow></math></span>.
- The columns are **linearly dependent**: one can be built from the others.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> has a solution with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> has no inverse.

## The crushed direction

<div class="ex">
<div class="lab">Worked example</div>

**seeing the crushed direction**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>. Try <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mtext> </mtext><mn>4</mn><mo>−</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> have the same image <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo></mrow></math></span>. Given <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo></mrow></math></span>, you cannot know which input it came from.

</div>

```
A = np.array([[1, 2], [2, 4]])
print(A @ np.array([2, -1]))  # [0 0]
print(A @ np.array([1, 1]), A @ np.array([3, 0]))  # [3 6] [3 6]
print(np.linalg.matrix_rank(A))  # 1
```

## In ML

- **Duplicate features.** If one feature column is an exact copy (or a multiple) of another, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> is singular and ordinary least squares has no unique answer.
- **Bottlenecks.** A layer that maps 512 dimensions to 32 and back cannot be undone: the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>512</mn><mo>×</mo><mn>512</mn></mrow></math></span> product of the two maps is singular.

<div class="take">
<div class="lab">Key takeaway</div>

singular means some direction is crushed to zero. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> has no unique solution and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> has no inverse.

</div>

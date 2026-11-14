---
id: "03.17"
course: "ml-maths-course"
module: "03"
slug: "elementary-row-operations"
title: "Elementary Row Operations"
keyIdeas: ["Three row operations never change the solutions: swap two rows, multiply a row by a non-zero number, add a multiple of one row to another.", "Each can be undone, so no solutions are gained or lost.", "Each row operation is the same as multiplying ⟦959⟧ on the left by an invertible \"elementary\" matrix."]
question: "Apply R2 ← R2 − 2·R1 to ⟦968⟧. What does the new row 2 say?"
answer: "Step 1, 2·R1 is ⟦974⟧. Step 2, R2 − 2·R1 = ⟦975⟧. Step 3, the new row 2 says ⟦976⟧, so ⟦831⟧. Row 1 is unchanged: ⟦977⟧, so ⟦391⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Solving a system by hand means rewriting equations into simpler ones with the same solutions. Only three kinds of rewrite are allowed, and they are enough to solve any linear system.

## The three operations

1. **Swap** two rows (write the equations in a different order).
2. **Scale** a row by a non-zero number (multiply both sides of an equation).
3. **Replace** a row by itself plus a multiple of another row (add equations).

Each turns true equations into true equations, and each can be reversed: swap back, divide by the same number, subtract the same multiple. So the set of solutions never changes.

<div class="ex">
<div class="lab">Worked example</div>

**each operation on a small system**

Start: R1 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo>∣</mo><mn>4</mn><mo stretchy="false">]</mo></mrow></math></span>, R2 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>2</mn><mo separator="true">,</mo><mn>5</mn><mo>∣</mo><mn>9</mn><mo stretchy="false">]</mo></mrow></math></span>.

Replace R2 by R2 − 2·R1: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo>∣</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>. Now R2 says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

Replace R1 by R1 − 2·R2: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo>∣</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></math></span>. Now R1 says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span>.

</div>

```
M = np.array([[1.0, 2, 4], [2, 5, 9]])
M[1] = M[1] - 2 * M[0]
M[0] = M[0] - 2 * M[1]
print(M)
```

## Row operations are matrix products

"R2 ← R2 − 2·R1" is the same as multiplying on the left by
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>E</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>2</mn></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>

```
E = np.array([[1, 0], [-2, 1]])
M = np.array([[1, 2, 4], [2, 5, 9]])
print(E @ M)
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>E</mi></mrow></math></span> is invertible (its determinant is 1), which is the matrix version of "this step can be undone". This view is what turns elimination into the LU factorisation (lesson 03.26).

## Where this breaks in practice

- **Scaling by zero.** Multiplying a row by 0 destroys an equation and is not allowed.
- **Operating on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> but not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span>.** Every operation must be applied to the whole augmented row, including the right-hand side.

<div class="take">
<div class="lab">Key takeaway</div>

swap, scale by non-zero, and add multiples of rows. Each is reversible, so solutions are kept.

</div>

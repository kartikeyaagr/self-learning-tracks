---
id: "03.22"
course: "ml-maths-course"
module: "03"
slug: "pivots-and-free-variables"
title: "Pivots and Free Variables"
keyIdeas: ["Variables whose columns have a pivot are basic; the others are free.", "Free variables can take any value, and the basic variables are then determined.", "A consistent system with at least one free variable has infinitely many solutions."]
question: "Solve ⟦1030⟧, ⟦1031⟧. Which variable is free? Give two solutions."
answer: "Step 1, R2 − R1 gives ⟦1033⟧. Pivots are in the ⟦7⟧ and ⟦509⟧ columns, so ⟦313⟧ is free. Step 2, set ⟦1034⟧: ⟦1035⟧, ⟦1039⟧. Step 3, solutions ⟦1040⟧. With ⟦777⟧: ⟦1041⟧. With ⟦778⟧: ⟦1015⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy", "SymPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

When a system has fewer independent equations than unknowns, some variables are not pinned down. They are free to take any value, and they describe a whole line or plane of solutions. Knowing which ones are free is how you write every solution at once.

## Basic and free

After reducing to echelon form:

- A column with a pivot belongs to a **basic** variable.
- A column without a pivot belongs to a **free** variable.

Set each free variable to a parameter (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi></mrow></math></span>, ...) and solve for the basic variables in terms of them.

<div class="ex">
<div class="lab">Worked example</div>

**one free variable**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>+</mo><mi>z</mi><mo>=</mo><mn>3</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mn>2</mn><mi>y</mi><mo>+</mo><mn>3</mn><mi>z</mi><mo>=</mo><mn>6</mn></mrow></math></span>.

R2 − R1: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>+</mo><mn>2</mn><mi>z</mi><mo>=</mo><mn>3</mn></mrow></math></span>. Pivots in columns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> is free.

Set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><mi>t</mi></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>3</mn><mo>−</mo><mn>2</mn><mi>t</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>3</mn><mo>−</mo><mo stretchy="false">(</mo><mn>3</mn><mo>−</mo><mn>2</mn><mi>t</mi><mo stretchy="false">)</mo><mo>−</mo><mi>t</mi><mo>=</mo><mi>t</mi></mrow></math></span>.

All solutions: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>t</mi><mo separator="true">,</mo><mtext> </mtext><mn>3</mn><mo>−</mo><mn>2</mn><mi>t</mi><mo separator="true">,</mo><mtext> </mtext><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span>, a line in 3D.

</div>

```
for t in (0, 1, 2):
    x = np.array([t, 3 - 2 * t, t])
    print(np.array([[1, 1, 1], [1, 2, 3]]) @ x)
```

Every value of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo></mrow></math></span>.

## Counting

For a consistent system with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> unknowns and rank <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span>:

- number of basic variables = <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span>,
- number of free variables = <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>r</mi></mrow></math></span>,
- the solutions form a flat set of dimension <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>r</mi></mrow></math></span>.

```
A = np.array([[1, 1, 1], [1, 2, 3]])
print(A.shape[1] - np.linalg.matrix_rank(A))  # 1
```

## Where you will see this

Free variables are exactly what module 04 calls the null space: the directions you can move in without changing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span>. Over-parameterised models have many such directions, which is why many different weight settings fit the same training data.

<div class="take">
<div class="lab">Key takeaway</div>

pivot columns give basic variables, the rest are free. Each free variable adds one dimension to the set of solutions.

</div>

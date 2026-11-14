---
id: "03.18"
course: "ml-maths-course"
module: "03"
slug: "gaussian-elimination"
title: "Gaussian Elimination"
keyIdeas: ["Gaussian elimination works column by column, using a pivot to make every entry below it zero.", "The result is upper triangular (a staircase), solved from the bottom up.", "Forward elimination costs about ⟦933⟧ operations for an ⟦908⟧ system."]
question: "Eliminate below the first pivot in ⟦978⟧. What are the multipliers?"
answer: "Step 1, the pivot is 2. Step 2, row 2 multiplier ⟦989⟧: R2 − 2·R1 = ⟦990⟧. Step 3, row 3 multiplier ⟦991⟧: R3 − 4·R1 = ⟦992⟧. Step 4, the first column is now ⟦993⟧, and the next step uses the 1 in row 2 as the pivot."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Elimination is the school method of "use one equation to remove a variable from the others", done in a fixed order so a computer can run it. It is what `np.linalg.solve` does inside, and it is the basis of LU factorisation.

## The algorithm

Work column by column, left to right:

1. In the current column, pick a **pivot**: a non-zero entry in a row not yet used.
2. For each row below, subtract (entry / pivot) times the pivot row. The number entry / pivot is the **multiplier**. This makes the entry below the pivot zero.
3. Move to the next column and the next row.

~~~dg-flow Elimination: forward to a staircase, then back substitution.
[A | b] -(zero below pivots)-> *Upper triangular [U | c] -(solve bottom-up)-> x
~~~

~~~dg-raw
%%viz rowred live | Step through elimination one row operation at a time.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a full 2 by 2 solve**

R1 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo>∣</mo><mn>5</mn><mo stretchy="false">]</mo></mrow></math></span>, R2 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo>∣</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>. Pivot 2, multiplier <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow></math></span>.

R2 − ½R1 <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>1.5</mn><mo>∣</mo><mo>−</mo><mn>1.5</mn><mo stretchy="false">]</mo></mrow></math></span>.

Bottom row: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1.5</mn><mi>y</mi><mo>=</mo><mo>−</mo><mn>1.5</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn><mo>=</mo><mn>5</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span>.

</div>

## In code

```
def eliminate(A, b):
    A = A.astype(float).copy(); b = b.astype(float).copy(); n = len(b)
    for j in range(n):
        for i in range(j + 1, n):
            m = A[i, j] / A[j, j]
            A[i] -= m * A[j]; b[i] -= m * b[j]
    return A, b

U, c = eliminate(np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]]), np.array([4, 10, 24]))
print(U)
print(c)  # [4. 2. 2.]
```

The result is upper triangular: all zeros below the diagonal.

## Cost

Clearing column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span> touches about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mi>j</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span> entries. Summed over all columns that is about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><msup><mi>n</mi><mn>3</mn></msup><mn>3</mn></mfrac></mrow></math></span> multiply-adds, or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>2</mn><mn>3</mn></mfrac><msup><mi>n</mi><mn>3</mn></msup></mrow></math></span> arithmetic operations. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>10</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span> that is about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>6.7</mn><mo>×</mo><mn>1</mn><msup><mn>0</mn><mn>11</mn></msup></mrow></math></span>, a few seconds on a laptop.

<div class="take">
<div class="lab">Key takeaway</div>

use each pivot to zero the entries below it, column by column, until the system is a staircase.

</div>

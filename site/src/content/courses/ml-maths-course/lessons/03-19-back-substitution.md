---
id: "03.19"
course: "ml-maths-course"
module: "03"
slug: "back-substitution"
title: "Back Substitution"
keyIdeas: ["Once the system is upper triangular, solve the last equation first and work upward.", "Each step substitutes the already-known values into one equation with one new unknown.", "Back substitution costs about ⟦583⟧ operations, much less than elimination."]
question: "Solve ⟦994⟧, ⟦995⟧, ⟦996⟧ by back substitution."
answer: "Step 1, bottom row: ⟦996⟧, so ⟦1012⟧. Step 2, middle row: ⟦1013⟧, so ⟦831⟧. Step 3, top row: ⟦1014⟧, so ⟦335⟧. The solution is ⟦1015⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Elimination leaves a staircase. The last row has a single unknown, the row above has two, and so on. Solving from the bottom up turns a hard problem into a sequence of one-unknown problems.

## The method

For an upper-triangular system <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mi>x</mi><mo>=</mo><mi>c</mi></mrow></math></span>:

1. The last row says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mrow><mi>n</mi><mi>n</mi></mrow></msub><msub><mi>x</mi><mi>n</mi></msub><mo>=</mo><msub><mi>c</mi><mi>n</mi></msub></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>n</mi></msub><mo>=</mo><msub><mi>c</mi><mi>n</mi></msub><mi mathvariant="normal">/</mi><msub><mi>u</mi><mrow><mi>n</mi><mi>n</mi></mrow></msub></mrow></math></span>.
2. Row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub><msub><mi>x</mi><mi>i</mi></msub><mo>+</mo><mo stretchy="false">(</mo><mtext>known terms</mtext><mo stretchy="false">)</mo><mo>=</mo><msub><mi>c</mi><mi>i</mi></msub></mrow></math></span>. Move the known terms over and divide by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></math></span>.
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>x</mi><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>c</mi><mi>i</mi></msub><mo>−</mo><munder><mo>∑</mo><mrow><mi>j</mi><mo>&gt;</mo><mi>i</mi></mrow></munder><msub><mi>u</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mi>x</mi><mi>j</mi></msub></mrow><msub><mi>u</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mfrac></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**three unknowns**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mn>2</mn><mi mathvariant="normal">/</mi><mn>2</mn><mo>=</mo><mn>1</mn></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo>−</mo><mn>1</mn><mo>⋅</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>1</mn><mo>=</mo><mn>1</mn></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo>−</mo><mn>1</mn><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn><mo>=</mo><mn>1</mn></mrow></math></span>.

</div>

## In code

```
def back_sub(U, c):
    n = len(c); x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

U = np.array([[2.0, 1, 1], [0, 1, 1], [0, 0, 2]])
print(back_sub(U, np.array([4.0, 2, 2])))  # [1. 1. 1.]
```

## Forward substitution

A lower-triangular system <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mi>x</mi><mo>=</mo><mi>c</mi></mrow></math></span> is solved the same way from the top down. LU solves (lesson 03.26) use one forward and one back substitution.

## Cost

Row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> needs about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>i</mi></mrow></math></span> multiplications, so the total is about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mn>2</mn></mrow></math></span> multiply-adds. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>10</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span> that is fast even when elimination is slow, which is why factoring once and reusing the factors pays off.

## Where this breaks in practice

- **Zero on the diagonal.** If some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub><mo>=</mo><mn>0</mn></mrow></math></span>, the division fails. That means the system is singular (lessons 03.22 to 03.24).

<div class="take">
<div class="lab">Key takeaway</div>

solve the bottom equation first, then substitute upward one row at a time.

</div>

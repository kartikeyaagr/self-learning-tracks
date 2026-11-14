---
id: "03.14"
course: "ml-maths-course"
module: "03"
slug: "determinant-rules"
title: "Determinant Rules"
keyIdeas: ["⟦904⟧ and ⟦905⟧.", "⟦906⟧, and ⟦907⟧ for an ⟦908⟧ matrix.", "For a triangular matrix, ⟦856⟧ is the product of the diagonal."]
question: "⟦900⟧, ⟦901⟧. Compute ⟦902⟧ and ⟦903⟧ without multiplying matrices."
answer: "Step 1, both are triangular: ⟦922⟧, ⟦923⟧. Step 2, ⟦924⟧. Step 3, ⟦925⟧, not ⟦926⟧, because each of the 2 dimensions is scaled by 3."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Determinants and singularity"]
tools: ["NumPy"]
topic: {"id": "03.02", "title": "Determinants and singularity"}
---

## The big picture

You rarely compute a determinant from scratch. Instead you use a few rules that follow from "the determinant is a volume factor". They make proofs short and computations cheap.

## The rules

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span>. Doing one map then another multiplies the volume factors.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>det</mi><mo>⁡</mo><mi>A</mi></mrow></math></span>. Undoing a map divides the volume back.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mo>=</mo><mi>det</mi><mo>⁡</mo><mi>A</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>c</mi><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>c</mi><mi>n</mi></msup><mi>det</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> for an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix. Scaling every axis by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span> scales volume by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>c</mi><mi>n</mi></msup></mrow></math></span>.
- Triangular <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> (zeros above or below the diagonal): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><msub><mo>∏</mo><mi>i</mi></msub><msub><mi>a</mi><mrow><mi>i</mi><mi>i</mi></mrow></msub></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><msub><mo>∏</mo><mi>i</mi></msub><msub><mi>λ</mi><mi>i</mi></msub></mrow></math></span>, the product of the eigenvalues (module 05).

<div class="ex">
<div class="lab">Worked example</div>

**checking the rules with numbers**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> is triangular, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>6</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> is triangular too, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>B</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mn>6</mn></mrow></math></span>. And <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>3</mn><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mn>9</mn><mo>⋅</mo><mn>6</mn><mo>=</mo><mn>54</mn></mrow></math></span>.

</div>

```
A = np.array([[2.0, 1], [0, 3]]); B = np.array([[1.0, 0], [4, 1]])
print(round(np.linalg.det(A @ B), 6), round(np.linalg.det(3 * A), 6))  # 6.0 54.0
print(round(np.linalg.det(np.linalg.inv(A)), 6))  # 0.166667
```

## A useful consequence

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is invertible, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>⋅</mo><mi>det</mi><mo>⁡</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><mi>det</mi><mo>⁡</mo><mi>I</mi><mo>=</mo><mn>1</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> can't be 0. That is one half of "invertible if and only if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>".

## Where this breaks in practice

- **<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><mi>B</mi><mo stretchy="false">)</mo><mo mathvariant="normal">≠</mo><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>+</mo><mi>det</mi><mo>⁡</mo><mi>B</mi></mrow></math></span>.** There is no simple rule for sums. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>I</mi><mo>+</mo><mi>I</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mn>2</mn><mi>n</mi></msup></mrow></math></span>, not 2.

<div class="take">
<div class="lab">Key takeaway</div>

determinants multiply under products, invert under inverses, and scale by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>c</mi><mi>n</mi></msup></mrow></math></span>. Triangular ones are just the diagonal product.

</div>

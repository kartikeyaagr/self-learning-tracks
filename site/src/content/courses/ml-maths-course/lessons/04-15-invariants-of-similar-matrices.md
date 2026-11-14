---
id: "04.15"
course: "ml-maths-course"
module: "04"
slug: "invariants-of-similar-matrices"
title: "Invariants of Similar Matrices"
keyIdeas: ["Matrices ⟦8⟧ and ⟦1486⟧ are called similar: they describe the same map in different bases.", "Similar matrices share determinant, trace, rank and eigenvalues.", "Quantities that don't change with the basis describe the map itself, not the coordinates."]
question: "Check that ⟦1483⟧ and ⟦1498⟧ have the same determinant, trace and rank."
answer: "Step 1, determinants: ⟦1503⟧ and ⟦1504⟧. Step 2, traces (sum of the diagonal): ⟦1505⟧ and ⟦1322⟧. Step 3, ranks: both 2 (non-zero determinant). All match, as they must for similar matrices."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

If a number changes when you merely relabel the axes, it is about the labels, not the map. Numbers that stay the same under every change of basis describe the map itself. Four of them come up constantly.

## Similar matrices

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>A</mi><mi>B</mi></mrow></math></span> (for invertible <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>) are **similar**. They are the same map written in two coordinate systems.

## What they share

- **Determinant:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>A</mi><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>det</mi><mo>⁡</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>det</mi><mo>⁡</mo><mi>A</mi><mi>det</mi><mo>⁡</mo><mi>B</mi><mo>=</mo><mi>det</mi><mo>⁡</mo><mi>A</mi></mrow></math></span>. The volume factor of a map doesn't depend on the axes.
- **Trace**, the sum of the diagonal: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>A</mi><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mi>A</mi></mrow></math></span>, using <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>X</mi><mi>Y</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>Y</mi><mi>X</mi><mo stretchy="false">)</mo></mrow></math></span>.
- **Rank:** multiplying by invertible matrices doesn't change rank.
- **Eigenvalues** (module 05).

```
A = np.array([[2.0, 1], [1, 2]]); B = np.array([[1.0, 1], [1, -1]])
Ap = np.linalg.inv(B) @ A @ B
print(round(np.linalg.det(A), 6), round(np.linalg.det(Ap), 6), np.trace(A), round(np.trace(Ap), 6))  # 3.0 3.0 4.0 4.0
```

<div class="ex">
<div class="lab">Worked example</div>

**a random basis**

Take any invertible <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>, say <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>A</mi><mi>B</mi></mrow></math></span> looks nothing like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, but its determinant is still 3 and its trace still 4.

</div>

```
A = np.array([[2.0, 1], [1, 2]]); B = np.array([[1.0, 2], [0, 1]])
Ap = np.linalg.inv(B) @ A @ B
print(Ap)
print(round(np.linalg.det(Ap), 6), round(np.trace(Ap), 6))  # 3.0 4.0
```

## What is not shared

Individual entries, symmetry, and whether the matrix looks diagonal all depend on the basis. A symmetric matrix written in a non-orthonormal basis is usually not symmetric.

<div class="take">
<div class="lab">Key takeaway</div>

determinant, trace, rank and eigenvalues don't depend on the basis. They describe the map, not the coordinates.

</div>

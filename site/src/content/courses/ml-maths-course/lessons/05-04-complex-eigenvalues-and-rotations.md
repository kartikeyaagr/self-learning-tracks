---
id: "05.04"
course: "ml-maths-course"
module: "05"
slug: "complex-eigenvalues-and-rotations"
title: "Complex Eigenvalues and Rotations"
keyIdeas: ["Some real matrices have no real direction they only stretch; their eigenvalues are complex.", "Complex eigenvalues of a real matrix come in conjugate pairs ⟦1634⟧ and signal rotation.", "⟦1635⟧ still controls growth. Symmetric matrices always have real eigenvalues."]
question: "Find the eigenvalues of ⟦1633⟧. What does the matrix do geometrically?"
answer: "Step 1, trace 0, determinant ⟦1647⟧, so ⟦1648⟧ and ⟦1649⟧. Step 2, complex eigenvalues mean rotation; ⟦1650⟧ means lengths double. Step 3, the matrix is a 90 degree rotation followed by scaling by 2."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
---

## The big picture

A rotation turns every vector, so it has no real eigenvectors. Its eigenvalues become complex numbers. You don't need complex arithmetic for ML often, but you need to recognise what complex eigenvalues mean: rotation, and oscillation when the matrix is applied repeatedly.

## A rotation has no real eigenvector

~~~dg-raw
%%viz f_plane x=-2,2 y=-2,2 v=1,0.5,v;-0.5,1,Rv,f | Rotation by 90 degrees turns every vector, so Rv never points along v. No real direction is only stretched.
~~~

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> has trace 0 and determinant 1, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>λ</mi><mn>2</mn></msup><mo>+</mo><mn>1</mn><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mo>±</mo><mi>i</mi></mrow></math></span>, where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi><mo>=</mo><msqrt><mrow><mo>−</mo><mn>1</mn></mrow></msqrt></mrow></math></span>.

```
R = np.array([[0, -1], [1, 0]])
print(np.linalg.eigvals(R))  # [0.+1.j 0.-1.j]
```

## Reading complex eigenvalues

For a real <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>2</mn></mrow></math></span> matrix with eigenvalues <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>±</mo><mi>b</mi><mi>i</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>=</mo><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi><mo>=</mo><msqrt><mrow><msup><mi>a</mi><mn>2</mn></msup><mo>+</mo><msup><mi>b</mi><mn>2</mn></msup></mrow></msqrt></mrow></math></span> is the scaling per step.
- The angle <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi><mo>=</mo><mi>arctan</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>b</mi><mi mathvariant="normal">/</mi><mi>a</mi><mo stretchy="false">)</mo></mrow></math></span> is the rotation per step.

<div class="ex">
<div class="lab">Worked example</div>

**a spiral**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mn>0.9</mn><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mn>30</mn><mi mathvariant="normal">°</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mi>sin</mi><mo>⁡</mo><mn>30</mn><mi mathvariant="normal">°</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>sin</mi><mo>⁡</mo><mn>30</mn><mi mathvariant="normal">°</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mn>30</mn><mi mathvariant="normal">°</mi></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> has eigenvalues <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.9</mn><msup><mi>e</mi><mrow><mo>±</mo><mi>i</mi><mi>π</mi><mi mathvariant="normal">/</mi><mn>6</mn></mrow></msup></mrow></math></span>, with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi><mo>=</mo><mn>0.9</mn></mrow></math></span>.

Applied repeatedly, it turns each point by 30 degrees and shrinks it by 10%: a spiral into the origin.

</div>

```
t = np.pi / 6
A = 0.9 * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
print(np.round(np.abs(np.linalg.eigvals(A)), 6))  # [0.9 0.9]
```

## Symmetric matrices are safe

**Symmetric** matrices (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><mi>A</mi></mrow></math></span>) always have real eigenvalues and perpendicular eigenvectors. Covariances, Hessians and kernel matrices are symmetric, which is why most ML eigenproblems stay real.

## In ML

Recurrent networks and state-space models have transition matrices with complex eigenvalues; they produce oscillating responses. Some models (like S4 and LRU) parameterise those eigenvalues directly, with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span> for stability.

<div class="take">
<div class="lab">Key takeaway</div>

complex eigenvalues mean rotation; their size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>λ</mi><mi mathvariant="normal">∣</mi></mrow></math></span> still sets growth or decay.

</div>

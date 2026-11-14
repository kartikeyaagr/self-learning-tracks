---
id: "05.08"
course: "ml-maths-course"
module: "05"
slug: "powers-and-functions-of-a-matrix"
title: "Powers and Functions of a Matrix"
keyIdeas: ["⟦1700⟧: raise each eigenvalue to the power ⟦1199⟧.", "More generally ⟦1701⟧, for example ⟦1702⟧ or ⟦1703⟧.", "Linear recurrences (Fibonacci, linear RNNs) are solved in closed form this way."]
question: "Use ⟦1690⟧ to find ⟦1699⟧ for ⟦1483⟧, entry ⟦111⟧."
answer: "Step 1, ⟦1716⟧ with ⟦571⟧ columns ⟦1717⟧, ⟦1718⟧, ⟦1697⟧. Step 2, ⟦1719⟧. Step 3, entry ⟦855⟧ is ⟦1720⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

Computing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mn>100</mn></msup></mrow></math></span> by repeated multiplication is slow and hides what is going on. In the eigenbasis it is trivial: raise each eigenvalue to the 100th power. The same trick computes matrix exponentials and square roots, used in differential equations, state-space models and Gaussian sampling.

## Powers

The <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>V</mi></mrow></math></span> in the middle cancel:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mn>2</mn></msup><mo>=</mo><mi>V</mi><mi mathvariant="normal">Λ</mi><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>V</mi><mi mathvariant="normal">Λ</mi><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><mi>V</mi><msup><mi mathvariant="normal">Λ</mi><mn>2</mn></msup><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo separator="true">,</mo><mspace width="2em"/><msup><mi>A</mi><mi>k</mi></msup><mo>=</mo><mi>V</mi><msup><mi mathvariant="normal">Λ</mi><mi>k</mi></msup><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span></span>

```
A = np.array([[2.0, 1], [1, 2]])
vals, Q = np.linalg.eigh(A)
A10 = Q @ np.diag(vals ** 10) @ Q.T
print(np.allclose(A10, np.linalg.matrix_power(A, 10)))  # True
print(np.round(A10[0]))  # [29525. 29524.]
```

## Functions

For any function <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> defined on the eigenvalues:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mi>V</mi><mi>f</mi><mo stretchy="false">(</mo><mi mathvariant="normal">Λ</mi><mo stretchy="false">)</mo><msup><mi>V</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>e</mi><mi>A</mi></msup></mrow></math></span>: solves <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>˙</mo></mover><mo>=</mo><mi>A</mi><mi>x</mi></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>e</mi><mrow><mi>A</mi><mi>t</mi></mrow></msup><mi>x</mi><mo stretchy="false">(</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. State-space models such as S4 and Mamba discretise exactly this.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup></mrow></math></span>: for a covariance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">Σ</mi><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup><mi>z</mi></mrow></math></span> with standard normal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> has covariance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span>.

```
S = np.array([[2.0, 1], [1, 2]])
vals, Q = np.linalg.eigh(S)
R = Q @ np.diag(np.sqrt(vals)) @ Q.T
print(np.allclose(R @ R, S))  # True
```

## Fibonacci in closed form

<div class="ex">
<div class="lab">Worked example</div>

**a linear recurrence**

Fibonacci: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>F</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>F</mi><mi>k</mi></msub></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>F</mi><mi>k</mi></msub></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msub><mi>F</mi><mrow><mi>k</mi><mo>−</mo><mn>1</mn></mrow></msub></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

The matrix has eigenvalues <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>φ</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo>+</mo><msqrt><mn>5</mn></msqrt><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn><mo>≈</mo><mn>1.618</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>φ</mi><mo>≈</mo><mo>−</mo><mn>0.618</mn></mrow></math></span>.

Diagonalising gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>F</mi><mi>k</mi></msub><mo>=</mo><mo stretchy="false">(</mo><msup><mi>φ</mi><mi>k</mi></msup><mo>−</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>φ</mi><msup><mo stretchy="false">)</mo><mi>k</mi></msup><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><msqrt><mn>5</mn></msqrt></mrow></math></span>. The second term dies out, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>F</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mi mathvariant="normal">/</mi><msub><mi>F</mi><mi>k</mi></msub><mo>→</mo><mi>φ</mi></mrow></math></span>.

</div>

```
phi = (1 + 5 ** 0.5) / 2
fib = lambda k: round((phi ** k - (-1 / phi) ** k) / 5 ** 0.5)
print([fib(k) for k in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

<div class="take">
<div class="lab">Key takeaway</div>

in the eigenbasis, any function of a matrix is that function applied to each eigenvalue.

</div>

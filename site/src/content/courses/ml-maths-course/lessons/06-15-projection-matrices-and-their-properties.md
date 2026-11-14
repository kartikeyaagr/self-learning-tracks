---
id: "06.15"
course: "ml-maths-course"
module: "06"
slug: "projection-matrices-and-their-properties"
title: "Projection Matrices and Their Properties"
keyIdeas: ["The projection onto ⟦1388⟧ is ⟦2047⟧; with orthonormal columns ⟦571⟧, simply ⟦2048⟧.", "Orthogonal projections satisfy ⟦2045⟧ and ⟦2049⟧; their eigenvalues are 0 and 1, and ⟦2050⟧.", "⟦2046⟧ projects onto the orthogonal complement."]
question: "For ⟦2043⟧, write ⟦2044⟧. Check ⟦2045⟧, find its trace, and apply ⟦2046⟧ to ⟦280⟧."
answer: "Step 1, ⟦2061⟧. Step 2, ⟦2062⟧ since ⟦2063⟧. Step 3, ⟦2064⟧, the dimension of the line. Step 4, ⟦2065⟧: the complement kills ⟦280⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Projection and orthonormalization"]
tools: ["NumPy"]
topic: {"id": "06.03", "title": "Projection and orthonormalization"}
---

## The big picture

Writing the projection as a matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> makes its properties easy to state and check. Projection matrices show up as "hat matrices" in regression, as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> in PCA, and in debiasing methods that remove a direction.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo>=</mo><mi>A</mi><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>
If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi></mrow></math></span> has orthonormal columns, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mo>=</mo><mi>I</mi></mrow></math></span> and this collapses to
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo>=</mo><mi>Q</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><msub><mi>q</mi><mi>i</mi></msub><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span></span>
a sum of rank-1 projections onto each basis direction.

## Properties

| Property | Meaning |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>P</mi><mn>2</mn></msup><mo>=</mo><mi>P</mi></mrow></math></span> | projecting twice changes nothing |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>P</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><mi>P</mi></mrow></math></span> | the projection is orthogonal (perpendicular drop) |
| eigenvalues 0 and 1 | vectors in the subspace kept, perpendicular ones removed |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mi>P</mi><mo>=</mo><mi mathvariant="normal">rank</mi><mo>⁡</mo><mi>P</mi></mrow></math></span> | the trace counts the dimensions kept |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>−</mo><mi>P</mi></mrow></math></span> | projects onto the complement |

```
A = np.array([[1.0, 0], [1, 1], [1, 2]])
P = A @ np.linalg.inv(A.T @ A) @ A.T
print(np.allclose(P @ P, P), np.allclose(P, P.T), round(np.trace(P), 6))  # True True 2.0
print(np.round(np.sort(np.linalg.eigvalsh(P)), 6) + 0)  # [0. 1. 1.]
```

<div class="ex">
<div class="lab">Worked example</div>

**removing a direction**

To remove a unit direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> from every embedding (for example a "gender direction"), apply <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>−</mo><mi>u</mi><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>.

For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi><mo>=</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mn>3</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>I</mi><mo>−</mo><mi>u</mi><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>−</mo><mn>3</mn><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>3.2</mn><mo separator="true">,</mo><mo>−</mo><mn>2.4</mn><mo stretchy="false">)</mo></mrow></math></span>.

Check: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3.2</mn><mo separator="true">,</mo><mo>−</mo><mn>2.4</mn><mo stretchy="false">)</mo><mo>⋅</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo><mo>=</mo><mn>1.92</mn><mo>−</mo><mn>1.92</mn><mo>=</mo><mn>0</mn></mrow></math></span>.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a projection matrix is symmetric and idempotent (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>P</mi><mn>2</mn></msup><mo>=</mo><mi>P</mi></mrow></math></span>), its eigenvalues are 0 and 1, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>−</mo><mi>P</mi></mrow></math></span> removes what <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> keeps.

</div>

## Where this breaks in practice

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>m</mi></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>=</mo><mn>1</mn><msup><mn>0</mn><mn>6</mn></msup></mrow></math></span> samples that is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>12</mn></msup></mrow></math></span> entries. Compute <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi><mo stretchy="false">(</mo><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> instead: two thin multiplications.
- Projecting out a "bias direction" removes only its linear component; a nonlinear classifier can often still recover the attribute.

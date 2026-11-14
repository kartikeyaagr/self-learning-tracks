---
id: "06.08"
course: "ml-maths-course"
module: "06"
slug: "orthogonal-matrices"
title: "Orthogonal Matrices"
keyIdeas: ["A square ⟦571⟧ with orthonormal columns satisfies ⟦1731⟧, so ⟦1947⟧.", "It keeps lengths and angles: ⟦1948⟧ and ⟦1949⟧.", "⟦1950⟧, every eigenvalue has ⟦1951⟧, and the condition number is 1."]
question: "⟦1945⟧. Check ⟦1731⟧, compute ⟦1946⟧, and undo it."
answer: "Step 1, the columns ⟦5⟧ and ⟦822⟧ have length 1 and dot product 0, so ⟦1731⟧. Step 2, ⟦1959⟧, still length 5. Step 3, ⟦1960⟧, and ⟦1961⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Orthogonal structure"]
tools: ["NumPy"]
topic: {"id": "06.02", "title": "Orthogonal structure"}
---

## The big picture

Some matrices move vectors without distorting them: rotations and reflections. These **orthogonal matrices** keep every length and angle, are inverted by transposing, and never magnify errors.

## Why <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mo>=</mo><mi>I</mi></mrow></math></span>

Entry <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi></mrow></math></span> is the dot product of column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> with column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span>. Orthonormal columns make that 1 on the diagonal and 0 elsewhere. So
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>Q</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>
Inverting is free.

## What they preserve

- Lengths: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>Q</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mi>x</mi><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span>.
- Dot products and angles: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>Q</mi><mi>x</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>Q</mi><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span>.
- Volume: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>Q</mi><mo>=</mo><mo>±</mo><mn>1</mn></mrow></math></span>.
- Every eigenvalue has size 1, and the condition number <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo stretchy="false">(</mo><mi>Q</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></math></span>.
- A product of orthogonal matrices is orthogonal.

```
t = np.pi / 6
Q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
x = np.array([3.0, 4])
print(np.allclose(Q.T @ Q, np.eye(2)), round(np.linalg.norm(Q @ x), 10))  # True 5.0
print(round(np.linalg.cond(Q), 10), np.round(np.abs(np.linalg.eigvals(Q)), 10))  # 1.0 [1. 1.]
```

## Tall matrices with orthonormal columns

An <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>&gt;</mo><mi>n</mi></mrow></math></span> and orthonormal columns still has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mo>=</mo><msub><mi>I</mi><mi>n</mi></msub></mrow></math></span>. But <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> is **not** the identity. It is the projection onto the column space of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> (lesson 06.15). These "thin" <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> matrices come out of QR and the SVD.

<div class="take">
<div class="lab">Key takeaway</div>

an orthogonal matrix is a rigid motion; you invert it by transposing, and it keeps lengths, angles and condition numbers.

</div>

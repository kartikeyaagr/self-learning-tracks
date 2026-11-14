---
id: "06.17"
course: "ml-maths-course"
module: "06"
slug: "qr-decomposition"
title: "QR Decomposition"
keyIdeas: ["Recording the Gram-Schmidt coefficients gives ⟦2096⟧: ⟦571⟧ has orthonormal columns and ⟦1568⟧ is upper triangular with ⟦2097⟧.", "Least squares becomes ⟦2098⟧, solved by back substitution.", "This avoids forming ⟦1997⟧, so it keeps the condition number at ⟦1182⟧."]
question: "For ⟦2077⟧, ⟦2078⟧, write ⟦2096⟧ using the Gram-Schmidt result ⟦2079⟧, ⟦2082⟧."
answer: "Step 1, ⟦2103⟧. Step 2, ⟦2104⟧. Step 3, ⟦2105⟧. Step 4, ⟦2106⟧ and ⟦2107⟧; indeed ⟦2108⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Projection and orthonormalization"]
tools: ["NumPy"]
topic: {"id": "06.03", "title": "Projection and orthonormalization"}
---

## The big picture

Gram-Schmidt does more than produce <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span>: the numbers it computes along the way form a triangular matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi></mrow></math></span>. Together <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi>R</mi></mrow></math></span> is one of the workhorse factorisations of numerical linear algebra, used for least squares and inside eigenvalue algorithms.

## Where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi></mrow></math></span> comes from

Rearranging step 2 of Gram-Schmidt:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>a</mi><mi>k</mi></msub><mo>=</mo><msub><mi>r</mi><mrow><mn>1</mn><mi>k</mi></mrow></msub><msub><mi>q</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>r</mi><mrow><mi>k</mi><mi>k</mi></mrow></msub><msub><mi>q</mi><mi>k</mi></msub><mo separator="true">,</mo><mspace width="2em"/><msub><mi>r</mi><mrow><mi>j</mi><mi>k</mi></mrow></msub><mo>=</mo><msubsup><mi>q</mi><mi>j</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>a</mi><mi>k</mi></msub><mo separator="true">,</mo><mspace width="1em"/><msub><mi>r</mi><mrow><mi>k</mi><mi>k</mi></mrow></msub><mo>=</mo><mi mathvariant="normal">∥</mi><msub><mi>v</mi><mi>k</mi></msub><mi mathvariant="normal">∥</mi></mrow></math></span></span>
Column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> uses only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>q</mi><mi>k</mi></msub></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi></mrow></math></span> is upper triangular:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi>R</mi></mrow></math></span></span>

```
A = np.array([[2.0, 1], [0, 2]])
Q, R = np.linalg.qr(A)
print(np.round(np.abs(R), 6))
print(np.allclose(Q @ R, A))  # True
```

NumPy may flip the signs of matching rows of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi></mrow></math></span> and columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span>; the product is the same.

## Least squares with QR

Substitute <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi>R</mi></mrow></math></span> into the normal equations: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>R</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mi>R</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><msup><mi>R</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>Q</mi><mo>=</mo><mi>I</mi></mrow></math></span> leaves
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>R</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span></span>
A triangular system, solved from the bottom up.

```
A = np.array([[1.0, 0], [1, 1], [1, 2]]); b = np.array([1.0, 2, 4])
Q, R = np.linalg.qr(A)
xh = np.linalg.solve(R, Q.T @ b)
print(np.round(xh, 4))  # [0.8333 1.5   ]
```

Same answer as lesson 06.14, without forming <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi></mrow></math></span>.

~~~dg-flow Least squares via QR.
A, b -(QR)-> Q, R -(Qᵀ b)-> R x = Qᵀ b -(back substitution)-> *x-hat
~~~

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>Q</mi><mi>R</mi></mrow></math></span> packages Gram-Schmidt; least squares becomes the triangular system <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span>.

</div>

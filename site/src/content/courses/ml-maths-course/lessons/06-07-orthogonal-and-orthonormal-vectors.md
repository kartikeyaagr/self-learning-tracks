---
id: "06.07"
course: "ml-maths-course"
module: "06"
slug: "orthogonal-and-orthonormal-vectors"
title: "Orthogonal and Orthonormal Vectors"
keyIdeas: ["Vectors are orthogonal when ⟦1816⟧. An orthonormal set has length-1 vectors that are pairwise orthogonal.", "Orthonormal vectors are automatically independent.", "In an orthonormal basis, coordinates are dot products: ⟦1928⟧."]
question: "Check that ⟦1925⟧ and ⟦1926⟧ are orthonormal, then write ⟦1927⟧ in this basis."
answer: "Step 1, ⟦1939⟧ and ⟦1940⟧. Step 2, ⟦1941⟧. Orthonormal. Step 3, coordinates: ⟦1942⟧ and ⟦1943⟧. Step 4, check: ⟦1944⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Orthogonal structure"]
tools: ["NumPy"]
topic: {"id": "06.02", "title": "Orthogonal structure"}
---

## The big picture

In module 04 finding coordinates in a basis meant solving a linear system. If the basis vectors are perpendicular and of length 1, that work disappears: each coordinate is one dot product. This is why so much of ML (PCA, Fourier features, QR) works in orthonormal bases.

## Definitions

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> are **orthogonal** if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><mn>0</mn></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>q</mi><mi>k</mi></msub></mrow></math></span> are **orthonormal** if
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>q</mi><mi>j</mi></msub><mo>=</mo><mrow><mo fence="true">{</mo><mtable rowspacing="0.36em" columnalign="left left" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>i</mi><mo>=</mo><mi>j</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>i</mi><mo mathvariant="normal">≠</mo><mi>j</mi></mrow></mstyle></mtd></mtr></mtable></mrow></mrow></math></span></span>
Each has length 1, and every pair is perpendicular.

## Orthonormal means independent

Suppose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mn>1</mn></msub><msub><mi>q</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>c</mi><mi>k</mi></msub><msub><mi>q</mi><mi>k</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span>. Take the dot product with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mi>j</mi></msub></mrow></math></span>: every term vanishes except <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mi>j</mi></msub><msubsup><mi>q</mi><mi>j</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>q</mi><mi>j</mi></msub><mo>=</mo><msub><mi>c</mi><mi>j</mi></msub></mrow></math></span>. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>c</mi><mi>j</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span>. No vector in the set can be built from the others.

## Coordinates by dot products

~~~dg-raw
%%viz f_plane x=-2,7 y=-1,3 tk=1 v=0.6,0.8,q₁,a;-0.8,0.6,q₂,b va=0,0,1.8,2.4,3q₁,a;1.8,2.4,5,0,−4q₂~_ends_at_x,b | x = (5, 0) = 3q₁ − 4q₂. Each coordinate is one dot product: q₁·x = 3 and q₂·x = −4.
~~~

For an orthonormal basis <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>q</mi><mi>n</mi></msub></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>x</mi><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><mo stretchy="false">(</mo><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><msub><mi>q</mi><mi>i</mi></msub></mrow></math></span></span>
Dot <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> with each basis vector and you have its coordinates. No equations to solve.

```
q1 = np.array([0.6, 0.8]); q2 = np.array([-0.8, 0.6]); x = np.array([5.0, 0])
c1, c2 = q1 @ x, q2 @ x
print(round(c1, 6), round(c2, 6), np.round(c1 * q1 + c2 * q2, 6))  # 3.0 -4.0 [5. 0.]
```

## Lengths in an orthonormal basis

Pythagoras holds for the coordinates: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><msub><mo>∑</mo><mi>i</mi></msub><mo stretchy="false">(</mo><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>x</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span>. Here <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mn>3</mn><mn>2</mn></msup><mo>+</mo><msup><mn>4</mn><mn>2</mn></msup><mo>=</mo><mn>25</mn><mo>=</mo><mi mathvariant="normal">∥</mi><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>. This is why "fraction of variance explained" in PCA is just a ratio of sums of squared coordinates.

<div class="take">
<div class="lab">Key takeaway</div>

in an orthonormal basis, coordinates are dot products and lengths are Pythagoras on the coordinates.

</div>

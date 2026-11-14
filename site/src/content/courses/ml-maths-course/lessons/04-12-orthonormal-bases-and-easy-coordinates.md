---
id: "04.12"
course: "ml-maths-course"
module: "04"
slug: "orthonormal-bases-and-easy-coordinates"
title: "Orthonormal Bases and Easy Coordinates"
keyIdeas: ["An orthonormal basis has unit-length vectors that are pairwise perpendicular: ⟦1459⟧.", "Then coordinates are just dot products: ⟦1460⟧, so ⟦1461⟧.", "Orthonormal changes of basis keep lengths and angles."]
question: "For ⟦1457⟧, ⟦1458⟧, find the coordinates of ⟦1439⟧ and check the length is kept."
answer: "Step 1, ⟦1469⟧. Step 2, ⟦1470⟧. Step 3, ⟦1471⟧. Length kept."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

Finding coordinates in a general basis means solving a linear system. If the basis vectors are perpendicular and of length 1, that work disappears: each coordinate is a single dot product. That is why orthonormal bases are preferred everywhere in numerical work.

## Orthonormal

A basis is **orthonormal** when each vector has length 1 and every pair is perpendicular: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>b</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>b</mi><mi>j</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi><mo mathvariant="normal">≠</mo><mi>j</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>b</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>b</mi><mi>i</mi></msub><mo>=</mo><mn>1</mn></mrow></math></span>. In matrix form, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mi mathvariant="normal">⊤</mi></msup><mi>B</mi><mo>=</mo><mi>I</mi></mrow></math></span>.

## Coordinates by dot products

Since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mi mathvariant="normal">⊤</mi></msup><mi>B</mi><mo>=</mo><mi>I</mi></mrow></math></span>, the inverse of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>, so
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>c</mi><mo>=</mo><msup><mi>B</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo separator="true">,</mo><mspace width="2em"/><msub><mi>c</mi><mi>i</mi></msub><mo>=</mo><msubsup><mi>b</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>x</mi></mrow></math></span></span>
Each coordinate is "how much of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> points along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>b</mi><mi>i</mi></msub></mrow></math></span>".

~~~dg-raw
%%viz f_plane x=-1,4 y=-2,3 ln=1,-1,0,,m v=3,1,x,k;0.707,0.707,b₁;-0.707,0.707,b₂,f sg=3,1,2,2,m,d pt=2,2,lands_at_2.83_along_b₁,a | The rotated axes b₁ and b₂. The dashed segment drops x straight onto the b₁ direction; how far along it lands is the coordinate c₁ = b₁ᵀx.
~~~

```
B = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
x = np.array([3.0, 1.0])
c = B.T @ x
print(np.round(c, 2))  # [ 2.83 -1.41]
print(np.allclose(B @ c, x), np.isclose(c @ c, x @ x))  # True True
```

## Lengths and angles are kept

For orthonormal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>B</mi><mi>c</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><msup><mi>c</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>B</mi><mi mathvariant="normal">⊤</mi></msup><mi>B</mi><mi>c</mi><mo>=</mo><msup><mi>c</mi><mi mathvariant="normal">⊤</mi></msup><mi>c</mi></mrow></math></span>. So lengths, distances and angles are the same in the new coordinates. Rotations and reflections are exactly the orthonormal changes of basis.

## Getting one

QR factorisation (module 06) turns any basis into an orthonormal one for the same subspace. The SVD and eigendecompositions of symmetric matrices also produce them.

<div class="take">
<div class="lab">Key takeaway</div>

in an orthonormal basis, coordinates are dot products and lengths don't change. Prefer such bases whenever you can.

</div>

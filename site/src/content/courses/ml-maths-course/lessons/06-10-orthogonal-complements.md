---
id: "06.10"
course: "ml-maths-course"
module: "06"
slug: "orthogonal-complements"
title: "Orthogonal Complements"
keyIdeas: ["The orthogonal complement ⟦1978⟧ is every vector perpendicular to all of ⟦385⟧.", "Every ⟦7⟧ splits uniquely as ⟦1979⟧, and ⟦1980⟧.", "The null space is the orthogonal complement of the row space: ⟦1981⟧."]
question: "⟦385⟧ is the line through ⟦1335⟧ in ⟦26⟧. Describe ⟦1978⟧ and split ⟦1326⟧ into its two parts."
answer: "Step 1, ⟦1978⟧ is every ⟦7⟧ with ⟦1990⟧, i.e. ⟦1991⟧: the horizontal plane. Step 2, ⟦1992⟧, one part in each. Step 3, dimensions ⟦1993⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Orthogonal structure"]
tools: ["NumPy"]
topic: {"id": "06.02", "title": "Orthogonal structure"}
---

## The big picture

A subspace and "everything perpendicular to it" together fill the whole space, with no overlap. This split is behind least squares (residuals are perpendicular to features) and the four fundamental subspaces.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>S</mi><mo>⊥</mo></msup><mo>=</mo><mo stretchy="false">{</mo><mi>x</mi><mo>:</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>s</mi><mo>=</mo><mn>0</mn><mtext> for every </mtext><mi>s</mi><mo>∈</mo><mi>S</mi><mo stretchy="false">}</mo></mrow></math></span></span>
read "S perp". It is itself a subspace. To check membership, it is enough to be perpendicular to a basis of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-3,4 y=-3,4 tk=1 ln=1,-2,0,S,a,0.85;2,1,0,S⊥,f,0.85 v=1,3,x,k;2,1,s,a;-1,2,s⊥,f sg=2,1,1,3,m,d;-1,2,1,3,m,d | S is the line through (2, 1); S⊥ is the perpendicular line through (1, −2). Any x splits into s in S plus s⊥ in S⊥.
~~~

## Facts

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi><mo>∩</mo><msup><mi>S</mi><mo>⊥</mo></msup><mo>=</mo><mo stretchy="false">{</mo><mn>0</mn><mo stretchy="false">}</mo></mrow></math></span>: only zero is perpendicular to itself.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>dim</mi><mo>⁡</mo><mi>S</mi><mo>+</mo><mi>dim</mi><mo>⁡</mo><msup><mi>S</mi><mo>⊥</mo></msup><mo>=</mo><mi>n</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msup><mi>S</mi><mo>⊥</mo></msup><msup><mo stretchy="false">)</mo><mo>⊥</mo></msup><mo>=</mo><mi>S</mi></mrow></math></span>.
- Every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> splits in exactly one way as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>s</mi><mo>+</mo><msup><mi>s</mi><mo>⊥</mo></msup></mrow></math></span>.

## The fundamental subspaces pair up

For an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>:

| Subspace | Its complement |
| --- | --- |
| row space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> | null space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> |
| column space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span> | left null space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo></mrow></math></span> |

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> says exactly that <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is perpendicular to every row.

## Residuals are perpendicular to features

In least squares (module 08) the residual <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>=</mo><mi>y</mi><mo>−</mo><mi>X</mi><mover accent="true"><mi>β</mi><mo>^</mo></mover></mrow></math></span> satisfies <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>r</mi><mo>=</mo><mn>0</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>∈</mo><mi>C</mi><mo stretchy="false">(</mo><mi>X</mi><msup><mo stretchy="false">)</mo><mo>⊥</mo></msup></mrow></math></span>. With an intercept column of ones, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mn mathvariant="normal">𝟏</mn><mi mathvariant="normal">⊤</mi></msup><mi>r</mi><mo>=</mo><mn>0</mn></mrow></math></span>: residuals sum to zero.

```
rng = np.random.default_rng(0)
X = np.column_stack([np.ones(20), rng.normal(size=20)]); y = rng.normal(size=20)
beta = np.linalg.lstsq(X, y, rcond=None)[0]
r = y - X @ beta
print(np.allclose(X.T @ r, 0))  # True
```

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>S</mi><mo>⊥</mo></msup></mrow></math></span> split the space; every vector is a part in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> plus a part perpendicular to it.

</div>

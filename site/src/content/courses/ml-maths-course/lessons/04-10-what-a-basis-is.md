---
id: "04.10"
course: "ml-maths-course"
module: "04"
slug: "what-a-basis-is"
title: "What a Basis Is"
keyIdeas: ["A basis of a subspace is a list of vectors that is linearly independent and spans the subspace.", "Every basis of the same subspace has the same number of vectors: the dimension.", "The standard basis of ⟦1377⟧ is ⟦1434⟧."]
question: "Is ⟦1431⟧ a basis of ⟦662⟧? Is ⟦1432⟧? Is ⟦1433⟧?"
answer: "Step 1, ⟦111⟧ and ⟦1352⟧ are independent (lesson 04.04) and two independent vectors in ⟦662⟧ span it. Basis. Step 2, ⟦1438⟧, dependent. Not a basis. Step 3, three vectors in ⟦662⟧ are always dependent. They span ⟦662⟧ but are not a basis."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

A point's coordinates depend on the axes you measure along. A **basis** is a choice of axes: just enough vectors to reach everything, with nothing redundant. Many ML methods are nothing more than a clever choice of basis.

## The definition

A **basis** of a subspace <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> is a list of vectors that

1. is linearly independent (nothing redundant), and
2. spans <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> (nothing missing).

~~~dg-raw
%%viz f_plane x=-2,3 y=-2,3 v=1,1,b₁;1,-1,b₂,f | b₁ = (1, 1) and b₂ = (1, −1): independent, and together they reach every point of the plane. A basis of ℝ².
~~~

## The standard basis

The **standard basis** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>…</mo><mtext> </mtext><mo stretchy="false">)</mo></mrow></math></span>, and so on. In it, the coordinates of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> are just its entries: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>3</mn><msub><mi>e</mi><mn>1</mn></msub><mo>+</mo><mn>1</mn><msub><mi>e</mi><mn>2</mn></msub></mrow></math></span>.

## Dimension

A key theorem: every basis of the same subspace has the same number of vectors. That number is the **dimension**.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> has dimension <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>.
- A plane through the origin has dimension 2.

<div class="ex">
<div class="lab">Worked example</div>

**checking a basis**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span>: independent, and 2 vectors in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span>. A basis.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span>: dependent. Not a basis; it spans only a line.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span>: spans <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span> but is dependent. Too many.

</div>

```
for B in ([[1, 1], [1, -1]], [[1, 2], [2, 4]]):
    print(np.linalg.matrix_rank(np.array(B).T) == 2)
```

## A basis from a spanning set

Given vectors that span <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> but have redundancy, drop redundant ones until the rest are independent. In matrix terms, the pivot columns found by elimination form a basis of the column space.

<div class="take">
<div class="lab">Key takeaway</div>

a basis spans with no redundancy. Every basis of a space has the same size, the dimension.

</div>

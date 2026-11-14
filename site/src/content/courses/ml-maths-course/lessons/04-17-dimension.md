---
id: "04.17"
course: "ml-maths-course"
module: "04"
slug: "dimension"
title: "Dimension"
keyIdeas: ["The dimension of a subspace is the number of vectors in any basis.", "A subspace of ⟦1377⟧ has dimension between 0 and ⟦233⟧.", "Data often lies near a subspace or surface of much lower dimension than the raw feature count."]
question: "What is the dimension of (a) the plane ⟦1515⟧ in ⟦26⟧; (b) the span of ⟦1366⟧ and ⟦1516⟧; (c) ⟦1517⟧?"
answer: "Step 1, (a) one equation in 3 unknowns leaves 2 free variables, so dimension 2. Step 2, (b) the second vector is twice the first, so the span is a line: dimension 1. Step 3, (c) the zero subspace has the empty basis: dimension 0."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Dimension, rank and nullity"]
tools: ["NumPy"]
topic: {"id": "04.04", "title": "Dimension, rank and nullity"}
---

## The big picture

"How many numbers do you need to describe a point here?" is what dimension measures. A plane needs 2 coordinates even if it sits inside a space with 1,000 axes. Much of ML is about finding the low dimension hiding in high-dimensional data.

## The definition

The **dimension** of a subspace is the number of vectors in any of its bases. All bases have the same size, so this is well defined.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> has dimension <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>.
- A plane through the origin has dimension 2.
- A line through the origin has dimension 1.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>0</mn><mo stretchy="false">}</mo></mrow></math></span> has dimension 0.

## Counting with equations

In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span>, each independent linear equation removes one dimension. So the solutions of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> independent equations <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> form a subspace of dimension <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>k</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**dimensions from equations**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span>: 1 equation, dimension <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>−</mo><mn>1</mn><mo>=</mo><mn>2</mn></mrow></math></span>. A basis: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Add <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>−</mo><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span>: 2 independent equations, dimension 1. The line through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
A = np.array([[1, 1, 1], [1, -1, 0]])
print(3 - np.linalg.matrix_rank(A))  # 1
print(A @ np.array([1, 1, -2]))  # [0 0]
```

## Low dimension in data

A <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>28</mn><mo>×</mo><mn>28</mn></mrow></math></span> image is a point in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>784</mn></msup></mrow></math></span>, but images of handwritten digits occupy a tiny part of that space. PCA finds that about 50 directions explain most of the variation in MNIST, so the data lies near a subspace of dimension about 50.

## Where this breaks in practice

- **Curved data.** Data on a curved surface (a circle in the plane) has low intrinsic dimension but no low-dimensional subspace fits it. Linear methods then overestimate the dimension.

<div class="take">
<div class="lab">Key takeaway</div>

dimension is the size of any basis. Each independent equation removes one dimension.

</div>

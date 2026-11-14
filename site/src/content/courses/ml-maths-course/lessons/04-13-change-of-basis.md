---
id: "04.13"
course: "ml-maths-course"
module: "04"
slug: "change-of-basis"
title: "Change of Basis"
keyIdeas: ["Changing basis changes the coordinates, not the vector.", "Old-to-new: ⟦1443⟧. New-to-old: ⟦1450⟧.", "Between two bases ⟦1474⟧ and ⟦1475⟧: ⟦1476⟧."]
question: "A vector has coordinates ⟦836⟧ in basis ⟦1440⟧, ⟦1441⟧. Find its coordinates in the basis ⟦1472⟧, ⟦1473⟧."
answer: "Step 1, convert to standard: ⟦1480⟧. Step 2, solve ⟦1481⟧: ⟦1455⟧, then ⟦1337⟧. Step 3, coordinates ⟦836⟧ in the new basis too, by coincidence; check: ⟦1482⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

Data can be described in many coordinate systems: raw pixels, principal components, Fourier frequencies. Moving between them is a change of basis. This lesson shows the bookkeeping, which always goes through the standard coordinates.

## The two directions

With basis matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>:

- From new coordinates to standard: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>B</mi><mi>c</mi></mrow></math></span>.
- From standard to new coordinates: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo>=</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>x</mi></mrow></math></span>.

## Between two bases

To go from basis <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>1</mn></msub></mrow></math></span> to basis <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>2</mn></msub></mrow></math></span>, pass through standard coordinates:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>c</mi><mn>2</mn></msub><mo>=</mo><msubsup><mi>B</mi><mn>2</mn><mrow><mo>−</mo><mn>1</mn></mrow></msubsup><mo stretchy="false">(</mo><msub><mi>B</mi><mn>1</mn></msub><msub><mi>c</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow></math></span></span>

~~~dg-flow Changing between two bases goes through standard coordinates.
coords in B1 -(multiply by B1)-> *standard coords -(solve with B2)-> coords in B2
~~~

```
B1 = np.array([[1.0, 1], [1, -1]]); B2 = np.array([[1.0, 1], [0, 1]])
c1 = np.array([2.0, 1])
x = B1 @ c1
print(x, np.linalg.solve(B2, x))  # [3. 1.] [2. 1.]
```

<div class="ex">
<div class="lab">Worked example</div>

**standardising is a change of coordinates**

A feature in centimetres, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>180</mn></mrow></math></span>. In metres it is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.8</mn></mrow></math></span>: the basis vector got 100 times longer, so the coordinate got 100 times smaller.

Coordinates always move opposite to the basis vectors: stretch the axis, shrink the number.

</div>

## The vector does not change

It is worth repeating: a change of basis relabels the same point. Anything that depends only on the point (its length, which side of a hyperplane it is on, its distance to other points) cannot depend on the basis, provided you compute it correctly in each basis.

<div class="take">
<div class="lab">Key takeaway</div>

to change basis, convert to standard coordinates with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>1</mn></msub></mrow></math></span>, then solve with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>2</mn></msub></mrow></math></span>. The vector stays put; only its numbers change.

</div>

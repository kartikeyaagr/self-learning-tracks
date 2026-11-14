---
id: "06.02"
course: "ml-maths-course"
module: "06"
slug: "norms-and-distance"
title: "Norms and Distance"
keyIdeas: ["The length (Euclidean norm) is ⟦1822⟧; the distance between ⟦7⟧ and ⟦509⟧ is ⟦1823⟧.", "Expanding gives ⟦1824⟧.", "That identity computes a whole table of pairwise distances with one matrix multiply."]
question: "Find ⟦1821⟧, the distance between ⟦758⟧ and ⟦658⟧, and check the expansion formula."
answer: "Step 1, ⟦1838⟧, and ⟦1839⟧ too. Step 2, ⟦1840⟧, with length ⟦1841⟧. Step 3, the formula: ⟦1842⟧, and ⟦1843⟧ matches."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
---

## The big picture

The dot product of a vector with itself gives its squared length. From length you get distance, and distance is what nearest-neighbour search, clustering and kernels run on.

## Length

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>=</mo><msqrt><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></msqrt><mo>=</mo><msqrt><mrow><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><mo>⋯</mo><mo>+</mo><msubsup><mi>x</mi><mi>n</mi><mn>2</mn></msubsup></mrow></msqrt></mrow></math></span></span>
In 2D this is Pythagoras: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> has length <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mn>9</mn><mo>+</mo><mn>16</mn></mrow></msqrt><mo>=</mo><mn>5</mn></mrow></math></span>. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> dimensions it is the same formula with more terms.

A **unit vector** has length 1. Divide any non-zero vector by its length to get one: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi></mrow></math></span>. This is called normalising.

## Distance

The distance between <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> is the length of the arrow from one to the other:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>d</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>y</mi><mi mathvariant="normal">∥</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-1,5 y=-4,5 tk=1 v=3,4,x,k;4,-3,y,b sg=4,-3,3,4,f | The distance between x = (3, 4) and y = (4, −3) is the length of the orange segment, x − y = (−1, 7): √50 ≈ 7.07.
~~~

```
x = np.array([3.0, 4]); y = np.array([4.0, -3])
print(np.linalg.norm(x), round(np.linalg.norm(x - y), 4))  # 5.0 7.0711
```

## Distances from dot products

Expand the square:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>y</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>+</mo><mi mathvariant="normal">∥</mi><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>−</mo><mn>2</mn><mtext> </mtext><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span></span>
This lets you compute **all** pairwise distances between two sets of points at once. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> holds <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> points as rows and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> holds <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi></mrow></math></span>, then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><msup><mi>Y</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> is the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>m</mi></mrow></math></span> table of dot products; add the squared lengths and you have every squared distance.

```
X = np.array([[0.0, 0], [3, 4]]); Y = np.array([[0.0, 0], [6, 8], [3, 0]])
sq = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
print(np.sqrt(np.maximum(sq, 0)))
```

Row 1 is the distances from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>: 0, 10 and 3. Row 2 is from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>: 5, 5 and 4. The `np.maximum(sq, 0)` guards against tiny negative values from rounding.

<div class="ex">
<div class="lab">Worked example</div>

**why this matters**

k-nearest neighbours on 10,000 queries and 1,000,000 stored points needs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>10</mn></msup></mrow></math></span> distances.

A Python loop over pairs would take hours. One matrix multiply on a GPU takes seconds.

Vector databases, k-means and RBF kernels all start from this table.

</div>

## Other norms

| Norm | Formula | Where it appears |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow></math></span> (Euclidean) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mo>∑</mo><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup></mrow></msqrt></mrow></math></span> | distances, weight decay |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>L</mi><mn>1</mn></msub></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∑</mo><mo stretchy="false">∣</mo><msub><mi>x</mi><mi>i</mi></msub><mo stretchy="false">∣</mo></mrow></math></span> | lasso, sparse models |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>L</mi><mi mathvariant="normal">∞</mi></msub></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>max</mi><mo>⁡</mo><mo stretchy="false">∣</mo><msub><mi>x</mi><mi>i</mi></msub><mo stretchy="false">∣</mo></mrow></math></span> | adversarial robustness bounds |

Only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow></math></span> comes from a dot product; this module uses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>L</mi><mn>2</mn></msub></mrow></math></span> unless it says otherwise.

<div class="take">
<div class="lab">Key takeaway</div>

length is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></msqrt></mrow></math></span>, distance is the length of the difference, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>+</mo><mi mathvariant="normal">∥</mi><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>−</mo><mn>2</mn><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span> turns a distance table into one matrix multiply.

</div>

## Where this breaks in practice

- In float32, the expansion can give small negative numbers when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> are almost equal, because two large nearly equal numbers are subtracted (catastrophic cancellation). Clamp at 0 before the square root.

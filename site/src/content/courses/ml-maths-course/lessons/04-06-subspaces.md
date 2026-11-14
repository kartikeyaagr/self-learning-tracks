---
id: "04.06"
course: "ml-maths-course"
module: "04"
slug: "subspaces"
title: "Subspaces"
keyIdeas: ["A subspace contains 0 and is closed under adding vectors and scaling them.", "Lines and planes through the origin are subspaces; shifted lines and quadrants are not.", "Every span is a subspace, and every subspace is the span of some vectors."]
question: "Which are subspaces of ⟦662⟧: (a) ⟦1374⟧; (b) ⟦1375⟧; (c) ⟦1376⟧?"
answer: "Step 1, (a) contains ⟦685⟧; sums and multiples of points ⟦1385⟧ stay of that form. Subspace. Step 2, (b) does not contain ⟦685⟧ (that would need ⟦1058⟧). Not a subspace. Step 3, (c) contains ⟦111⟧ but not ⟦1386⟧. Not closed under scaling, so not a subspace."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Subspaces, columns and null spaces"]
tools: ["NumPy"]
topic: {"id": "04.02", "title": "Subspaces, columns and null spaces"}
---

## The big picture

The sets that come up in linear algebra (spans, solution sets of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>, ranges of layers) all share one shape: a flat set through the origin that goes on forever. That shape is called a subspace, and it is what "dimension" is measured on.

## The definition

A **subspace** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> is a set of vectors with three properties:

- It contains the zero vector.
- Adding two vectors in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> gives a vector in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>.
- Scaling a vector in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> by any number gives a vector in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>.

In pictures: a point at the origin, a line through the origin, a plane through the origin, or the whole space.

~~~dg-raw
%%viz f_plane x=-3,3 y=-3,3 ln=2,-1,0,y_=_2x_(subspace),a,0.85;2,-1,-1,y_=_2x+1_(not),f,0.2 pt=0,0,0 | y = 2x passes through the origin and is a subspace. y = 2x + 1 is the same line shifted, misses the origin, and is not.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**subspace or not?**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>t</mi><mo separator="true">,</mo><mn>2</mn><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span>: contains <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>8</mn><mo stretchy="false">)</mo></mrow></math></span> is on it; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>5</mn><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>10</mn><mo stretchy="false">)</mo></mrow></math></span> is on it. Subspace.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>t</mi><mo separator="true">,</mo><mn>2</mn><mi>t</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span>: does not contain <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. It is a shifted line, called an affine set.

The first quadrant: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> is in it, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> is not. Not a subspace.

</div>

## Spans are subspaces

A sum of two combinations of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>v</mi><mi>k</mi></msub></mrow></math></span> is another combination, and so is a multiple. So every span is a subspace. Going the other way, every subspace is the span of some list of vectors (a basis, lesson 04.10).

## The solution set of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>y</mi><mo>=</mo><mn>0</mn></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mi>c</mi><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span>. So the solutions of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> form a subspace.

```
A = np.array([[1, 2, 3]])
x = np.array([-2, 1, 0]); y = np.array([-3, 0, 1])
print(A @ x, A @ y, A @ (x + 5 * y))  # [0] [0] [0]
```

The solutions of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span> are a shifted copy, not a subspace.

<div class="take">
<div class="lab">Key takeaway</div>

a subspace is a flat set through the origin, closed under adding and scaling. Spans and solution sets of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> are subspaces.

</div>

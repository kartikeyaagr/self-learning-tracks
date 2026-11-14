---
id: "09.46"
course: "ml-maths-course"
module: "09"
slug: "double-integrals-over-rectangles"
title: "Double Integrals Over Rectangles"
keyIdeas: ["A double integral ⟦3043⟧ is the volume under the surface ⟦2449⟧ over the region ⟦1568⟧.", "It is the limit of 2D Riemann sums: cut ⟦1568⟧ into small cells, multiply each cell's area by a height, and add.", "Where ⟦90⟧ is negative the volume counts as negative, just as in one dimension."]
question: "Estimate ⟦3041⟧ over ⟦3042⟧ with a ⟦838⟧ grid of cells and midpoint heights."
answer: "Step 1, the cells are ⟦329⟧ wide and ⟦134⟧ tall, so each has area ⟦329⟧. Step 2, the midpoints are ⟦3051⟧, ⟦3052⟧, ⟦3053⟧, ⟦3054⟧, with heights ⟦3055⟧. Step 3, the sum is ⟦3056⟧, which is the exact answer here because ⟦90⟧ is linear."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Integration over two variables"]
tools: ["NumPy"]
topic: {"id": "09.07", "title": "Integration over two variables"}
---

## The big picture

Two random variables at once (height and weight, two pixel values, two weights in a network) have a **joint** density, a surface over the plane. Probabilities are volumes under that surface. This lesson builds the volume the same way lesson 09.40 built area: with small pieces.

## From strips to cells

In one dimension you cut an interval into strips. In two you cut a rectangle into cells of area <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Δ</mi><mi>A</mi><mo>=</mo><mi mathvariant="normal">Δ</mi><mi>x</mi><mtext> </mtext><mi mathvariant="normal">Δ</mi><mi>y</mi></mrow></math></span>, pick a point in each, and add height times area:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mo>∬</mo><mi>R</mi></msub><mi>f</mi><mtext> </mtext><mi>d</mi><mi>A</mi><mo>≈</mo><munder><mo>∑</mo><mrow><mi>i</mi><mo separator="true">,</mo><mi>j</mi></mrow></munder><mi>f</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo separator="true">,</mo><msub><mi>y</mi><mi>j</mi></msub><mo stretchy="false">)</mo><mtext> </mtext><mi mathvariant="normal">Δ</mi><mi>x</mi><mtext> </mtext><mi mathvariant="normal">Δ</mi><mi>y</mi></mrow></math></span></span>
As the cells shrink, the sum settles on the volume.

~~~dg-raw
%%viz riem2 | The 2D version of a Riemann sum: each column is height × cell area, and the columns add up to the volume.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**volume under a tilted plane**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>x</mi><mo>+</mo><mi>y</mi></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo><mo>×</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></math></span>. The average height is the height at the centre, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.5</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>1.5</mn></mrow></math></span>, and the area is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn></mrow></math></span>, so the volume is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn></mrow></math></span>.

A grid of cells with midpoint heights gives exactly 3 for any grid size, because the plane is linear.

</div>

```
n = 200
x = (np.arange(n) + 0.5) / n          # midpoints in [0, 1]
y = 2 * (np.arange(n) + 0.5) / n      # midpoints in [0, 2]
X, Y = np.meshgrid(x, y)
print(round(((X ** 2 + Y) * (1 / n) * (2 / n)).sum(), 4))  # 2.6667
```

The exact value of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∬</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>A</mi></mrow></math></span> over the same rectangle is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">/</mi><mn>3</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>8</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

a double integral is a volume, built from height × small area and added up.

</div>

## Where this breaks in practice

- A grid with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> points per side has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mi>d</mi></msup></mrow></math></span> cells in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions. Grids are fine in 2D and hopeless in 20D; that is why lesson 09.45 uses sampling.

---
id: "01.12"
course: "ml-maths-course"
module: "01"
slug: "cartesian-products-as-regions"
title: "Cartesian Products as Regions"
keyIdeas: ["A product of intervals is a box: ⟦245⟧ is a rectangle, ⟦241⟧ is a hypercube.", "A grid with 10 points per axis has ⟦246⟧ points, so grids explode with dimension.", "In high dimensions almost all of a cube's volume is in its corners: the inscribed ball takes a vanishing share."]
question: "How many grid points does a 10-per-axis grid on ⟦241⟧ need for ⟦242⟧ and ⟦243⟧? In ⟦243⟧, roughly what fraction of uniform random points in ⟦244⟧ fall inside the unit ball, and what does this mean for nearest-neighbour methods?"
answer: "Step 1, sizes multiply, so the grid has ⟦246⟧ points: ⟦254⟧ for ⟦242⟧ and ⟦250⟧ for ⟦243⟧. Step 2, the ball's share of the cube is about 0.0025 in 10 dimensions, so only about 1 point in 400 lands inside the ball. Step 3, most points sit in the corners, far from the centre and far from each other. Distances between random points become similar, so \"nearest\" neighbours are not much nearer than the rest, and you need enormous amounts of data to cover the space."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Set operations"]
tools: ["NumPy"]
topic: {"id": "01.02", "title": "Set operations"}
---

## The big picture

A product of intervals is a box. Boxes are how we describe "each feature in its own range", and two facts about boxes in many dimensions explain why high-dimensional data behaves so differently from what your 3-D intuition expects.

## Boxes

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">]</mo><mo>×</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo><mo>=</mo><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>:</mo><mn>0</mn><mo>≤</mo><mi>x</mi><mo>≤</mo><mn>2</mn><mo separator="true">,</mo><mtext> </mtext><mn>0</mn><mo>≤</mo><mi>y</mi><mo>≤</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span> is a 2 by 1 rectangle. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mn>3</mn></msup></mrow></math></span> is the unit cube. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mi>d</mi></msup></mrow></math></span> is called a **hypercube**.

<div class="ex">
<div class="lab">Worked example</div>

**a feature box**

Normalised pixel values live in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mn>784</mn></msup></mrow></math></span> for MNIST: each of the 784 coordinates in its own interval <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>.

</div>

~~~dg-raw
%%viz f_region kind=box a=0,2 b=0,1 x=-1,3 y=-1,2 edge=closed lab=[0,2]×[0,1] | The product [0, 2] × [0, 1] is a rectangle: x ranges over the first interval, y over the second.
~~~

## Fact 1: grids explode

Put grid points every 0.1 along each axis: 10 points per axis. The total is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mi>d</mi></msup></mrow></math></span>.

| d | grid points | share of the cube inside the inscribed ball |
| --- | --- | --- |
| 1 | 10 | 1.000 |
| 2 | 100 | 0.785 |
| 3 | 1,000 | 0.524 |
| 5 | 100,000 | 0.164 |
| 10 | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>10</mn></msup></mrow></math></span> | 0.0025 |

## Fact 2: corners take over

Put a ball inside the cube <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mi>d</mi></msup></mrow></math></span>, touching the middle of each face. In 2-D the disc covers <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>π</mi><mi mathvariant="normal">/</mi><mn>4</mn><mo>≈</mo><mn>78.5</mn><mi mathvariant="normal">%</mi></mrow></math></span> of the square. In 10-D the ball covers only 0.25% of the cube: nearly all the volume is in the corners. You can check by simulation:

```
rng = np.random.default_rng(0)
for d in [2, 10]:
    pts = rng.uniform(-1, 1, size=(200_000, d))
    share = ((pts ** 2).sum(axis=1) <= 1).mean()
    print(d, round(float(share), 2))
```

This prints about 0.79 for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>=</mo><mn>2</mn></mrow></math></span> and 0.0 (to two decimals) for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>=</mo><mn>10</mn></mrow></math></span>.

## The curse of dimensionality

Both facts together are the **curse of dimensionality**: you cannot cover a high-dimensional space with samples, and random points are almost never close to each other. Methods that rely on "nearby" examples, such as k-nearest neighbours, need either very large datasets or data that really lives on a lower-dimensional surface (which real data usually does).

<div class="take">
<div class="lab">Key takeaway</div>

products of intervals are boxes. In many dimensions grids explode and volume flees to the corners.

</div>

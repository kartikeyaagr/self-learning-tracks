---
id: "01.06"
course: "ml-maths-course"
module: "01"
slug: "planar-regions-in-set-builder-form"
title: "Planar Regions in Set-Builder Form"
keyIdeas: ["A set-builder rule on pairs ⟦116⟧ describes a region of the plane: a disc, a half-plane, a strip.", "⟦117⟧ is a ball of radius ⟦118⟧ around ⟦119⟧. Radius searches return exactly this set.", "A linear classifier predicts class 1 on a half-space ⟦120⟧."]
question: "Which of the points ⟦110⟧, ⟦111⟧, ⟦112⟧ and ⟦113⟧ are in ⟦114⟧, and which are in ⟦115⟧?"
answer: "Step 1, the disc test is ⟦130⟧. ⟦110⟧: ⟦131⟧, in. ⟦111⟧: ⟦132⟧, out. ⟦112⟧: ⟦133⟧, in. ⟦113⟧: ⟦134⟧, in (on the edge, and the ⟦135⟧ includes it). Step 2, the half-plane test is ⟦136⟧. ⟦110⟧: ⟦137⟧? No. ⟦111⟧: ⟦138⟧? No. ⟦112⟧: ⟦139⟧? Yes. ⟦113⟧: ⟦140⟧? No. So only ⟦112⟧ is in the half-plane."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["NumPy"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

When the elements are points <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>, a set-builder rule draws a shape. This is how you will see decision regions, constraint sets and neighbourhoods described in ML papers.

## Shapes from rules

<div class="ex">
<div class="lab">Worked example</div>

**four regions**

Disc: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>:</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>≤</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, every point on or inside the circle of radius 1.

Half-plane: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>:</mo><mi>y</mi><mo>≥</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, everything on or above the line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn></mrow></math></span>.

Strip: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>:</mo><mn>0</mn><mo>≤</mo><mi>x</mi><mo>≤</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, a vertical band of width 1.

Rectangle: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>:</mo><mn>0</mn><mo>≤</mo><mi>x</mi><mo>≤</mo><mn>2</mn><mo separator="true">,</mo><mtext> </mtext><mn>0</mn><mo>≤</mo><mi>y</mi><mo>≤</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, a 2 by 1 box.

</div>

In code, each row of an array is a point, and the rule becomes a mask over rows:

```
pts = np.array([[0.0, 0.5], [1.0, 1.0], [-0.6, 0.6], [0.0, -1.0]])
inside = pts[(pts ** 2).sum(axis=1) <= 1]
print(inside.tolist())  # [[0.0, 0.5], [-0.6, 0.6], [0.0, -1.0]]
```

The point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> fails because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>2</mn><mo>&gt;</mo><mn>1</mn></mrow></math></span>.

~~~dg-raw
%%viz f_region kind=disk c=0,0 r=1 x=-2,2 y=-2,2 edge=closed pts=0.5,0.5,in;1.5,1,out,f lab=x²+y²≤1 | Every point whose coordinates pass the test x² + y² ≤ 1 is shaded. (0.5, 0.5) passes; (1.5, 1) fails.
~~~

## Balls around a centre

The set of points within distance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> of a centre <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span> is a **ball**:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>:</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>c</mi><mi mathvariant="normal">∥</mi><mo>≤</mo><mi>r</mi><mo stretchy="false">}</mo></mrow></math></span></span>
Here <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>c</mi><mi mathvariant="normal">∥</mi></mrow></math></span> is the straight-line distance between <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span> (module 06 covers it properly).

```
X = np.array([[1.0, 1.0], [4.0, 5.0], [1.5, 0.5], [0.0, 3.0]])
c = np.array([1.0, 1.0])
print(X[np.linalg.norm(X - c, axis=1) <= 1.0].tolist())  # [[1.0, 1.0], [1.5, 0.5]]
```

Nearest-neighbour search and radius queries in vector databases return this set.

## Classifier regions

A linear classifier says "class 1" exactly on a set of this shape:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup><mo>:</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo>&gt;</mo><mn>0</mn><mo stretchy="false">}</mo></mrow></math></span></span>
Read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> as "a weighted sum of the features" for now; module 03 explains it. In 2-D this is a half-plane, the side of a line.

## Where this breaks in practice

- **Testing "on the edge" with `==`.** Whether a point is exactly on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn></mrow></math></span> almost never comes out True in floats. Use `np.isclose`.

<div class="take">
<div class="lab">Key takeaway</div>

a rule on coordinates draws a region. Balls and half-planes are the two shapes you will meet most.

</div>

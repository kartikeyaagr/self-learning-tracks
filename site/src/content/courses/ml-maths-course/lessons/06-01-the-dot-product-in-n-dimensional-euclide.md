---
id: "06.01"
course: "ml-maths-course"
module: "06"
slug: "the-dot-product-in-n-dimensional-euclidean-space"
title: "The Dot Product in N-Dimensional Euclidean Space"
keyIdeas: ["The dot product of ⟦1803⟧ is ⟦1804⟧: multiply matching entries, then add.", "It is linear in each input, symmetric, and ⟦1805⟧ for every ⟦597⟧.", "A dot product of 0 means the vectors are perpendicular."]
question: "Compute ⟦1800⟧, ⟦1801⟧ and ⟦1802⟧. Which pair is perpendicular?"
answer: "Step 1, ⟦1819⟧. Step 2, ⟦675⟧. Step 3, ⟦1820⟧. Step 4, the second pair has dot product 0, so ⟦758⟧ and ⟦658⟧ are perpendicular."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
---

## The big picture

How similar are two word embeddings? How far is this image from that one? Which keys should a query attend to? All of these are answered with one operation, the **dot product**. Length, distance and angle all come from it. This lesson defines it; the next few build the geometry on top.

## The definition

For two vectors <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>x</mi><mi>i</mi></msub><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub><msub><mi>y</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><msub><mi>y</mi><mn>2</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>x</mi><mi>n</mi></msub><msub><mi>y</mi><mi>n</mi></msub></mrow></math></span></span>
It is also written <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>⋅</mo><mi>y</mi></mrow></math></span> or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">⟨</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">⟩</mo></mrow></math></span>. The result is a single number.

<div class="ex">
<div class="lab">Worked example</div>

**three dot products**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo><mo>=</mo><mn>4</mn><mo>+</mo><mn>10</mn><mo>+</mo><mn>18</mn><mo>=</mo><mn>32</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mo>−</mo><mn>3</mn><mo stretchy="false">)</mo><mo>=</mo><mn>12</mn><mo>−</mo><mn>12</mn><mo>=</mo><mn>0</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mn>9</mn><mo>+</mo><mn>16</mn><mo>=</mo><mn>25</mn></mrow></math></span>.

</div>

```
x = np.array([1.0, 2, 3]); y = np.array([4.0, 5, 6])
print(x @ y, np.dot(x, y), (x * y).sum())  # 32.0 32.0 32.0
```

## Three properties

Everything geometric in this module follows from these:

| Property | Formula |
| --- | --- |
| Linear in each input | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>a</mi><mi>y</mi><mo>+</mo><mi>b</mi><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mi>a</mi><mtext> </mtext><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>+</mo><mi>b</mi><mtext> </mtext><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>z</mi></mrow></math></span> |
| Symmetric | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><msup><mi>y</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> |
| Positive | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> whenever <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span> |

The third holds because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mo>+</mo><mo>⋯</mo><mo>+</mo><msubsup><mi>x</mi><mi>n</mi><mn>2</mn></msubsup></mrow></math></span> is a sum of squares.

## Sign tells direction

~~~dg-raw
%%viz f_plane x=-3,5 y=-4,5 tk=1 v=3,4,x,k;4,1,y₁,a;4,-3,y₂,b;-2,-1,y₃,f | x = (3, 4). y₁ points roughly the same way (x·y₁ = 16 > 0), y₂ is perpendicular (x·y₂ = 0), y₃ points away (x·y₃ = −10 < 0).
~~~

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>: the vectors point roughly the same way (angle under 90 degrees).
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><mn>0</mn></mrow></math></span>: perpendicular.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>&lt;</mo><mn>0</mn></mrow></math></span>: roughly opposite (angle over 90 degrees).

## A matrix product is many dot products

Entry <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>B</mi></mrow></math></span> is the dot product of row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> with column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span> of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>. A linear layer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mi>x</mi></mrow></math></span> computes one dot product per output unit: each row of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi></mrow></math></span> is a "template" and the output says how well <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> matches it.

<div class="take">
<div class="lab">Key takeaway</div>

the dot product multiplies matching entries and adds; its sign tells you whether two vectors point together, apart, or at right angles.

</div>

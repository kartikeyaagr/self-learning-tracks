---
id: "04.03"
course: "ml-maths-course"
module: "04"
slug: "the-span-of-a-set-of-vectors"
title: "The Span of a Set of Vectors"
keyIdeas: ["The span of ⟦1299⟧ is the set of all their linear combinations: ⟦1330⟧.", "One non-zero vector spans a line through the origin; two non-parallel vectors span a plane.", "A layer ⟦1317⟧ can only output vectors in the span of ⟦1259⟧'s columns."]
question: "Is ⟦1326⟧ in the span of ⟦1327⟧ and ⟦1328⟧? Is ⟦1329⟧?"
answer: "Step 1, a combination is ⟦1336⟧. Step 2, for ⟦1326⟧: ⟦1337⟧, ⟦1338⟧, and ⟦1339⟧. Yes. Step 3, for ⟦1329⟧: the third entry would have to be ⟦1340⟧, not 7. No."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Vectors, spans and dependence"]
tools: ["NumPy"]
topic: {"id": "04.01", "title": "Vectors, spans and dependence"}
---

## The big picture

Given some vectors, which points can you reach by mixing them? The answer is a flat set through the origin called the span. It tells you which targets a system can hit and which outputs a layer can ever produce.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">span</mi><mo>⁡</mo><mo stretchy="false">(</mo><msub><mi>v</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>v</mi><mi>k</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><msub><mi>c</mi><mn>1</mn></msub><msub><mi>v</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>c</mi><mi>k</mi></msub><msub><mi>v</mi><mi>k</mi></msub><mo>:</mo><msub><mi>c</mi><mi>i</mi></msub><mo>∈</mo><mi mathvariant="normal">ℝ</mi><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mi>V</mi><mi>c</mi><mo>:</mo><mi>c</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>k</mi></msup><mo stretchy="false">}</mo></mrow></math></span></span>
Read it: "every vector you can get as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> times some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span>".

- One non-zero vector in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span> spans a line through the origin.
- Two vectors that point in different directions span a plane through the origin.
- Three vectors not all in one plane span all of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span>.

~~~dg-raw
%%viz span live | Drag u and v, then mix them with c₁ and c₂. Line the two vectors up and the span collapses from the whole plane to a line.
~~~

## Testing membership

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> is in the span exactly when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi><mi>c</mi><mo>=</mo><mi>b</mi></mrow></math></span> has a solution. In code, compare ranks, or solve by least squares and check the residual:

```
V = np.array([[1, 0], [0, 1], [1, 1]])
for b in ([2, 3, 5], [2, 3, 7]):
    c, *_ = np.linalg.lstsq(V, np.array(b, float), rcond=None)
    print(np.allclose(V @ c, b))
```

<div class="ex">
<div class="lab">Worked example</div>

**the span of a layer**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span> maps <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span> into <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span>.

Every output has the form <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>2</mn></msub><mo separator="true">,</mo><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span>: the third entry is the sum of the first two.

No input can make the output <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. The layer's outputs fill only a plane inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span>.

</div>

## Where this breaks in practice

- **Narrow layers.** A layer with 64 output units feeding a layer with 4096 can only produce outputs in a 64-dimensional span, however wide the next layer is.

<div class="take">
<div class="lab">Key takeaway</div>

the span is everything reachable by mixing. A target outside it can't be hit exactly.

</div>

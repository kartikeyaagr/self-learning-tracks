---
id: "06.03"
course: "ml-maths-course"
module: "06"
slug: "angles-and-cosine-similarity"
title: "Angles and Cosine Similarity"
keyIdeas: ["The angle between ⟦7⟧ and ⟦509⟧ satisfies ⟦1847⟧; this ratio is the cosine similarity.", "Cosine 1 means same direction, 0 perpendicular, ⟦499⟧ opposite. It ignores length.", "For unit vectors, ⟦1848⟧, so ranking by distance and by cosine agree."]
question: "Embeddings ⟦1844⟧, ⟦1845⟧, ⟦1846⟧. Give the cosine similarity and the Euclidean distance of ⟦682⟧ and of ⟦119⟧ from ⟦364⟧."
answer: "Step 1, ⟦1861⟧, ⟦1862⟧, ⟦1863⟧, so ⟦1864⟧. Distance ⟦1865⟧. Step 2, ⟦1866⟧, so ⟦1860⟧. Distance ⟦1867⟧. Step 3, ⟦682⟧ points exactly the same way as ⟦364⟧ but is twice as long; cosine ignores that, distance does not."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
---

## The big picture

Two documents about the same topic can have embeddings of very different lengths (one is long, one is short). Cosine similarity compares only the **direction**, which is usually what "similar meaning" should depend on.

## The angle formula

In 2D, the law of cosines gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span>, where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span> is the angle between the arrows. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> dimensions we use this as the definition of the angle:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mfrac><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi></mrow></mfrac></mrow></math></span></span>
This number is the **cosine similarity**.

~~~dg-raw
%%viz dot live | Drag a and b. The dot product, the angle and the projection update together.
~~~

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span> | Angle | Meaning |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn></mrow></math></span> | 0 degrees | same direction |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn></mrow></math></span> | 90 degrees | perpendicular, "unrelated" |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> | 180 degrees | opposite directions |

```
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
a = np.array([3.0, 4]); b = np.array([6.0, 8]); c = np.array([4.0, -3])
print(cos(a, b), cos(a, c))  # 1.0 0.0
print(np.linalg.norm(a - b), round(np.linalg.norm(a - c), 4))  # 5.0 7.0711
```

## Cosine versus distance

- Cosine ignores length; distance does not. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mn>2</mn><mi>a</mi></mrow></math></span> have cosine 1 but distance 5.
- If every vector is first scaled to length 1, then
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>−</mo><mn>2</mn><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mn>2</mn><mo>−</mo><mn>2</mn><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span></span>
so the closest point by distance is the most similar by cosine. Vector databases rely on this: normalise once, then use fast Euclidean search.

<div class="ex">
<div class="lab">Worked example</div>

**normalising first**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>a</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>b</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>c</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mn>0.8</mn><mo separator="true">,</mo><mo>−</mo><mn>0.6</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mover accent="true"><mi>a</mi><mo>^</mo></mover><mo>−</mo><mover accent="true"><mi>b</mi><mo>^</mo></mover><mi mathvariant="normal">∥</mi><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mo>=</mo><mn>1</mn></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mover accent="true"><mi>a</mi><mo>^</mo></mover><mo>−</mo><mover accent="true"><mi>c</mi><mo>^</mo></mover><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mn>2</mn><mo>−</mo><mn>0</mn><mo>=</mo><mn>2</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mo>=</mo><mn>0</mn></mrow></math></span>.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

cosine similarity is the dot product of the normalised vectors; it measures direction only.

</div>

## Where this breaks in practice

- Raw transformer embeddings often all share one large common direction, so cosines between unrelated inputs come out high (0.8 or more). Subtracting the mean embedding first fixes much of this.

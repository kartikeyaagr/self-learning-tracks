---
id: "03.05"
course: "ml-maths-course"
module: "03"
slug: "distance-from-a-point-to-a-hyperplane"
title: "Distance from a Point to a Hyperplane"
keyIdeas: ["The signed distance from ⟦7⟧ to ⟦679⟧ is ⟦750⟧.", "The raw score ⟦734⟧ is not a distance: doubling ⟦378⟧ and ⟦682⟧ doubles it but leaves the hyperplane unchanged.", "A linear SVM chooses the boundary that maximises the smallest distance to the training points (the margin)."]
question: "A classifier has ⟦655⟧, ⟦749⟧ and predicts positive when ⟦353⟧. Give the signed distance of ⟦656⟧, and say what happens to the score and the distance if you double ⟦378⟧ and ⟦682⟧."
answer: "Step 1, score: ⟦763⟧. Step 2, ⟦756⟧. Step 3, distance ⟦764⟧, on the positive side. Step 4, doubling gives ⟦765⟧, ⟦766⟧: score ⟦767⟧, twice as big. But ⟦768⟧, so the distance is still ⟦134⟧. The boundary line has not moved, so the raw score is not a fair confidence; the distance is."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

"How far is this point from the boundary?" is the natural measure of how confident a linear classifier should be, and it is the quantity that support vector machines optimise. The raw score looks like a distance but isn't. This lesson derives the correct formula.

## The formula

The **signed distance** from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> to the hyperplane <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>d</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>−</mo><mi>b</mi></mrow><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></mfrac></mrow></math></span></span>
Why divide by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></math></span>? Start at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and walk straight toward the plane along the unit normal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></math></span>, for a distance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span>. You land on the plane, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mtext> ⁣</mtext><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">(</mo><mi>x</mi><mo>−</mo><mi>d</mi><mtext> </mtext><mi>w</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">)</mo><mo>=</mo><mi>b</mi></mrow></math></span>. Expand: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>−</mo><mi>d</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo>=</mo><mi>b</mi></mrow></math></span>. Solve for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> and you get the formula.

~~~dg-raw
%%viz hyper | The dashed segment is the shortest path from the point x to the line. It runs parallel to the normal w.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 2D boundary**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mn>5</mn></mrow></math></span>, so the line is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mn>4</mn><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>5</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo>=</mo><mn>5</mn></mrow></math></span>.

The origin: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>−</mo><mn>5</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>5</mn><mo>=</mo><mo>−</mo><mn>1</mn></mrow></math></span>. It is 1 unit away, on the negative side.

The point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>9</mn><mo>+</mo><mn>16</mn><mo>−</mo><mn>5</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>5</mn><mo>=</mo><mn>4</mn></mrow></math></span>. Four units away on the positive side.

Check by geometry: the closest point on the line to the origin is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>b</mi><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></mfrac><mi>w</mi><mo>=</mo><mfrac><mn>5</mn><mn>25</mn></mfrac><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>0.6</mn><mo separator="true">,</mo><mn>0.8</mn><mo stretchy="false">)</mo></mrow></math></span>, which has length exactly 1.

</div>

```
w = np.array([3.0, 4.0]); b = 5.0
dist = lambda x: (w @ x - b) / np.linalg.norm(w)
print(dist(np.array([0, 0])), dist(np.array([3, 4])))  # -1.0 4.0
print((2 * w @ np.array([3, 4]) - 2 * b) / np.linalg.norm(2 * w))  # 4.0
```

The last line doubles <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span>: the distance does not change.

## Margins

For labelled points with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>y</mi><mi>i</mi></msub><mo>∈</mo><mo stretchy="false">{</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mo>+</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>y</mi><mi>i</mi></msub><mtext> </mtext><mi>d</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></math></span> is positive when the point is on its correct side. The **support vector machine** picks the line that makes the smallest of these as large as possible. Logistic regression does not divide by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></math></span>, so on perfectly separable data it keeps growing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> to push scores higher, which is why it needs regularisation there.

## Where this breaks in practice

- **Unscaled features.** Distances depend on units. With age in years and income in dollars, "distance" mostly measures income. Standardise first.

<div class="take">
<div class="lab">Key takeaway</div>

divide the score by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></math></span> before you treat it as a distance.

</div>

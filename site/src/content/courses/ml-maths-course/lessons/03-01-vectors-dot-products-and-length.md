---
id: "03.01"
course: "ml-maths-course"
module: "03"
slug: "vectors-dot-products-and-length"
title: "Vectors, Dot Products and Length"
keyIdeas: ["A vector in ⟦2⟧ is a list of ⟦3⟧ numbers. The dot product ⟦128⟧ multiplies matching entries and adds them.", "The length (norm) is ⟦659⟧.", "⟦660⟧ means ⟦378⟧ and ⟦7⟧ are perpendicular (orthogonal)."]
question: "For ⟦655⟧ and ⟦656⟧, compute ⟦128⟧, ⟦394⟧ and ⟦657⟧. Is ⟦658⟧ perpendicular to ⟦378⟧?"
answer: "Step 1, ⟦672⟧. Step 2, ⟦673⟧. Step 3, ⟦674⟧. Step 4, ⟦675⟧, so yes, they are perpendicular."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

Every model input is a vector: an image is a long list of pixel values, a sentence becomes a list of embedding numbers. Lines, planes and linear systems are all written with three vector tools: the vector itself, the dot product, and the length. This lesson is a short refresher on those three.

## Vectors

A **vector** in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> is a list of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> numbers, like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span>. You can picture a 2D or 3D vector as an arrow from the origin to that point. In higher dimensions you can't picture it, but the arithmetic is the same.

You add vectors entry by entry and scale them entry by entry: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>⋅</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>6</mn><mo separator="true">,</mo><mn>8</mn><mo stretchy="false">)</mo></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-1,5 y=-1,5 v=3,4,w;2,1,x,f | Two vectors drawn as arrows from the origin: w = (3, 4) and x = (2, 1).
~~~

## The dot product

The **dot product** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> multiplies matching entries and adds them:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>w</mi><mi>d</mi></msub><msub><mi>x</mi><mi>d</mi></msub></mrow></math></span></span>
For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>⋅</mo><mn>2</mn><mo>+</mo><mn>4</mn><mo>⋅</mo><mn>1</mn><mo>=</mo><mn>10</mn></mrow></math></span>. The little <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">⊤</mi></mrow></math></span> means "transpose"; for now, read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> as "w dot x".

## Length

The **length** (or norm) is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo>=</mo><msqrt><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>w</mi></mrow></msqrt></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mn>9</mn><mo>+</mo><mn>16</mn></mrow></msqrt><mo>=</mo><mn>5</mn></mrow></math></span>. This is Pythagoras: the arrow is the long side of a right triangle with sides 3 and 4.

## Angles and perpendicular vectors

The dot product also measures angle: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span>. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>=</mo><mn>0</mn></mrow></math></span>, so the angle is 90 degrees. Such vectors are **perpendicular**, also called **orthogonal**.

```
w = np.array([3, 4]); x = np.array([2, 1])
print(w @ x)  # 10
print(np.linalg.norm(w))  # 5.0
print(w @ np.array([4, -3]))  # 0
```

<div class="ex">
<div class="lab">Worked example</div>

**a model score is a dot product**

A linear model with weights <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> scores an input <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0.5</mn><mo stretchy="false">)</mo></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mn>2</mn><mo>−</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>2</mn></mrow></math></span>.

Each weight multiplies its own feature, and the results are added. That is all a dot product is.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

dot product = multiply matching entries and add. Length = square root of a vector dotted with itself. Zero dot product = perpendicular.

</div>

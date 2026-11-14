---
id: "06.04"
course: "ml-maths-course"
module: "06"
slug: "the-cauchy-schwarz-and-triangle-inequalities"
title: "The Cauchy-Schwarz and Triangle Inequalities"
keyIdeas: ["Cauchy-Schwarz: ⟦1871⟧, with equality only when one vector is a multiple of the other.", "It is what keeps ⟦1852⟧ between ⟦499⟧ and ⟦134⟧.", "Triangle inequality: ⟦1872⟧. A straight path is never longer than a detour."]
question: "Check Cauchy-Schwarz for ⟦1868⟧ with ⟦1869⟧ and with ⟦1870⟧. When is it an equality?"
answer: "Step 1, ⟦1882⟧, while ⟦1883⟧. So ⟦1884⟧. Step 2, ⟦1885⟧, and ⟦1886⟧. Equal. Step 3, equality happened because ⟦1887⟧ is a multiple of ⟦7⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
---

## The big picture

The angle formula divides a dot product by a product of lengths. For that to be a cosine, the ratio has to stay between <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn></mrow></math></span>. The Cauchy-Schwarz inequality guarantees it, in any number of dimensions.

## Cauchy-Schwarz

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∣</mi><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mi mathvariant="normal">∣</mi><mo>≤</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mtext> </mtext><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi></mrow></math></span></span>
Read it: "a dot product can never be bigger than the product of the lengths". Equality holds exactly when one vector is a multiple of the other (the angle is 0 or 180 degrees).

<div class="ex">
<div class="lab">Worked example</div>

**two checks**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><mn>5</mn><mo>≤</mo><msqrt><mn>50</mn></msqrt><mo>≈</mo><mn>7.07</mn></mrow></math></span>. Strict, since they point different ways.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><mn>10</mn><mo>=</mo><msqrt><mn>5</mn></msqrt><mo>⋅</mo><msqrt><mn>20</mn></msqrt></mrow></math></span>. Equal, since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>2</mn><mi>x</mi></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
ok = all(abs(x @ y) <= np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
         for x, y in rng.normal(size=(1000, 2, 50)))
print(ok)  # True
```

## The triangle inequality

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>+</mo><mi>y</mi><mi mathvariant="normal">∥</mi><mo>≤</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>+</mo><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-1,6 y=-1,5 tk=1 va=0,0,4,1,x,a;4,1,5,4,y,b;0,0,5,4,x+y,f | Walking x then y covers ‖x‖ + ‖y‖ ≈ 4.12 + 3.16 = 7.28. Going straight along x + y covers only ‖(5, 4)‖ ≈ 6.40.
~~~

It follows from Cauchy-Schwarz:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>+</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>+</mo><mn>2</mn><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>+</mo><mi mathvariant="normal">∥</mi><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>≤</mo><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>+</mo><mn>2</mn><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi><mo>+</mo><mi mathvariant="normal">∥</mi><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mo stretchy="false">(</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>+</mo><mi mathvariant="normal">∥</mi><mi>y</mi><mi mathvariant="normal">∥</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span></span>

## Why ML cares

- Cosine similarity is always a valid number in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>.
- The triangle inequality makes Euclidean distance a true **metric**. Fast nearest-neighbour structures (ball trees, some vector indexes) use it to skip whole regions: if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo stretchy="false">(</mo><mi>q</mi><mo separator="true">,</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span> is large and every point in a cluster is within <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> of its centre <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi></mrow></math></span>, no point in that cluster can be closer than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo stretchy="false">(</mo><mi>q</mi><mo separator="true">,</mo><mi>c</mi><mo stretchy="false">)</mo><mo>−</mo><mi>r</mi></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

Cauchy-Schwarz bounds a dot product by the product of the lengths; the triangle inequality follows and makes distance a metric.

</div>

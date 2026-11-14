---
id: "11.07"
course: "ml-maths-course"
module: "11"
slug: "joint-densities"
title: "Joint Densities"
keyIdeas: ["A pair of continuous variables has a joint density ⟦3650⟧ with ⟦3118⟧.", "Probabilities are volumes: ⟦3651⟧ (lesson 09.48).", "The density is probability per unit area and can be large on a small region."]
question: "⟦3602⟧ is uniform on the triangle ⟦3075⟧. What is its density?"
answer: "Step 1, the triangle has area ⟦2547⟧. Step 2, a constant density with total 1 must be ⟦132⟧ on the triangle."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two continuous variables"]
tools: ["NumPy"]
topic: {"id": "11.02", "title": "Two continuous variables"}
---

## The big picture

This is the continuous version of the joint table. Sums become integrals, but the operations are the same.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mo stretchy="false">(</mo><mi>X</mi><mo separator="true">,</mo><mi>Y</mi><mo stretchy="false">)</mo><mo>∈</mo><mi>R</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mo>∬</mo><mi>R</mi></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi><mo separator="true">,</mo><mspace width="2em"/><msub><mo>∬</mo><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></msub><mi>p</mi><mo>=</mo><mn>1</mn></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-0.2,1.3 y=-0.2,1.3 tk=1 pg=0:0_1:0_1:1,a ln=-1,1,0,y_=_x,k,0.85 | The triangle 0 ≤ y ≤ x ≤ 1. A uniform density on it has height 2, because the area is 1/2.
~~~

```
rng = np.random.default_rng(0)
u = rng.random((400000, 2))
x, y = u.max(1), u.min(1)
print(round((y <= x).mean(), 2), round(((x < 0.5) & (y < 0.5)).mean(), 3))  # 1.0 0.25
```

Taking the larger and smaller of two uniforms gives a uniform point on the triangle. The square <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>0.5</mn><msup><mo stretchy="false">]</mo><mn>2</mn></msup></mrow></math></span> cuts off a small triangle of area <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>8</mn></mrow></math></span>, so its probability is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>8</mn><mo>=</mo><mn>0.25</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

a joint density is a surface over the plane; probabilities are volumes under it.

</div>

## Where this breaks in practice

- A density value has units of 1/area. Comparing densities across differently scaled variables is meaningless.

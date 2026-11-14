---
id: "01.17"
course: "ml-maths-course"
module: "01"
slug: "distance-and-epsilon-balls"
title: "Distance and Epsilon-Balls"
keyIdeas: ["The distance between points ⟦7⟧ and ⟦313⟧ is ⟦314⟧, the straight-line length.", "The open ball ⟦315⟧ is every point closer than ⟦305⟧ to ⟦7⟧.", "On the real line, ⟦316⟧ is the interval ⟦317⟧."]
question: "Which of the points ⟦308⟧, ⟦309⟧, ⟦310⟧ are in the ball ⟦311⟧? On the real line, write ⟦312⟧ as an interval."
answer: "Step 1, compute distances to the origin. ⟦308⟧: 0.3. ⟦309⟧: ⟦325⟧. ⟦310⟧: 1. Step 2, the ball keeps distances strictly less than 0.8: the first two are in, ⟦310⟧ is out. Step 3, on the line, ⟦326⟧."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

"Small step", "nearby point", "a neighbourhood of the solution". All of these need a precise idea of closeness. The tool is the epsilon-ball: everything within a tiny distance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi></mrow></math></span> (Greek epsilon, read "epsilon") of a point.

## Distance

For points in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span>, the distance between <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>z</mi><mi mathvariant="normal">∥</mi><mo>=</mo><msqrt><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo>−</mo><msub><mi>z</mi><mn>1</mn></msub><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>+</mo><mo>⋯</mo><mo>+</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mi>d</mi></msub><mo>−</mo><msub><mi>z</mi><mi>d</mi></msub><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></msqrt></mrow></math></span></span>
the length of the straight line between them (Pythagoras in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions). On the real line it is just <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>x</mi><mo>−</mo><mi>z</mi><mi mathvariant="normal">∣</mi></mrow></math></span>.

```
x = np.array([0.0, 0.0])
z = np.array([3.0, 4.0])
print(np.linalg.norm(x - z))  # 5.0
```

## The epsilon-ball

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>B</mi><mi>ε</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><mi>z</mi><mo>:</mo><mi mathvariant="normal">∥</mi><mi>z</mi><mo>−</mo><mi>x</mi><mi mathvariant="normal">∥</mi><mo>&lt;</mo><mi>ε</mi><mo stretchy="false">}</mo></mrow></math></span></span>
Read it: "the ball of radius epsilon around x: every point closer than epsilon to x". The strict <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo></mrow></math></span> makes it an **open** ball (the edge is left out).

<div class="ex">
<div class="lab">Worked example</div>

**balls in one, two and three dimensions**

On the line: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>0.5</mn></msub><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mn>1.5</mn><mo stretchy="false">)</mo></mrow></math></span>, an open interval.

In the plane: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math></span> is the inside of the unit circle.

In 3-D: a solid sphere without its skin.

</div>

~~~dg-raw
%%viz f_region kind=disk c=0,0 r=1 x=-2,2 y=-2,2 edge=open pts=0,0,centre;0.6,0.3,inside | The open ball of radius 1 around the origin. The dashed edge is not part of it: only points at distance strictly less than 1 are inside.
~~~

## Why it matters

- **Small steps.** One gradient step of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi></mrow></math></span> moves the weights to a point in a ball around where they were.
- **Robustness.** An adversarial example is a point in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mi>ε</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> (a tiny change of the input) that the model classifies differently. Certified defences prove that every point of the ball gets the same label.
- **The next lessons.** Interior points, boundary points, open and closed sets are all defined with these balls.

<div class="take">
<div class="lab">Key takeaway</div>

an epsilon-ball is "everything within <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi></mrow></math></span>". It is the precise version of "nearby".

</div>

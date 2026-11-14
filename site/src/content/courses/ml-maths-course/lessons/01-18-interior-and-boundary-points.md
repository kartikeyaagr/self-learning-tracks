---
id: "01.18"
course: "ml-maths-course"
module: "01"
slug: "interior-and-boundary-points"
title: "Interior and Boundary Points"
keyIdeas: ["⟦7⟧ is an interior point of ⟦8⟧ if some small ball around ⟦7⟧ lies entirely inside ⟦8⟧.", "⟦7⟧ is a boundary point if every ball around it, however small, contains points in ⟦8⟧ and points outside.", "Gradient descent is safe at interior points; at the boundary a small step can leave the allowed set."]
question: "For ⟦327⟧, classify each of ⟦328⟧, ⟦329⟧, ⟦330⟧, ⟦134⟧ and ⟦331⟧ as interior, boundary or exterior, and say which belong to ⟦8⟧."
answer: "Step 1, ⟦329⟧: the ball ⟦334⟧ fits inside, so interior, and in ⟦8⟧. Step 2, ⟦330⟧: the ball of radius ⟦340⟧ fits, so interior, in ⟦8⟧. Step 3, ⟦328⟧: every ball around 0 contains negative numbers (outside) and small positive ones (inside), so boundary, and ⟦341⟧. Step 4, ⟦134⟧: every ball contains numbers just below 1 (inside) and above 1 (outside), so boundary, but ⟦342⟧. Step 5, ⟦331⟧: the ball ⟦343⟧ misses ⟦8⟧ completely, so exterior."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

Where you stand in a set matters. If you are in the middle, you can move a little in any direction and stay inside. If you are on the edge, some directions take you out. Interior and boundary points make this precise, and they explain why constrained optimisation needs extra care.

## The definitions

Take a set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and a point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is an **interior point** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> if there is **some** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mi>ε</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> entirely inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>. You have room to move.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is a **boundary point** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> if **every** ball around <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, however tiny, contains points of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and points not in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>. You are on the edge.
- Otherwise <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is an **exterior point**: some ball around it misses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> entirely.

<div class="ex">
<div class="lab">Worked example</div>

**points of the interval [0, 1]**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0.5</mn></mrow></math></span> is interior: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.4</mn><mo separator="true">,</mo><mn>0.6</mn><mo stretchy="false">)</mo></mrow></math></span> is inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow></math></span> is a boundary point: any <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>ε</mi><mo separator="true">,</mo><mn>1</mn><mo>+</mo><mi>ε</mi><mo stretchy="false">)</mo></mrow></math></span> contains numbers above 1.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0.999</mn></mrow></math></span> is still interior. The radius must be tiny (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi><mo>=</mo><mn>0.0005</mn></mrow></math></span> works), but it exists.

</div>

~~~dg-raw
%%viz ball live | Drag p and change the radius ε. An interior point needs just one small ball that fits inside S.
~~~

A boundary point may or may not belong to the set: 1 is a boundary point of both <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span> (where it belongs) and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> (where it does not).

## Why ML cares

Gradient descent takes a small step. At an interior point a small enough step stays inside the allowed set. At a boundary point it can leave. Probabilities must stay in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>; if a parameter is already at 1 and the gradient says "increase", the step leaves the set. **Projected gradient descent** snaps the point back after each step:

```
p = np.array([0.95, 0.40, 0.02])
grad = np.array([-1.0, 0.2, 0.5])
p_new = np.clip(p - 0.1 * grad, 0.0, 1.0)
print(p_new)  # [1.   0.38 0.  ]
```

The first entry would have been 1.05 and the third <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>0.03</mn></mrow></math></span>. Clipping projects them onto the boundary of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

interior means "some ball fits inside", boundary means "every ball pokes out". Constrained optimisers spend their effort at the boundary.

</div>

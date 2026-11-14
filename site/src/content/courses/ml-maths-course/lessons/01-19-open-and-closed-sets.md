---
id: "01.19"
course: "ml-maths-course"
module: "01"
slug: "open-and-closed-sets"
title: "Open and Closed Sets"
keyIdeas: ["A set is open if every point of it is an interior point. A set is closed if it contains all its boundary points.", "Some sets are neither: ⟦33⟧ contains the boundary point 0 but not 1. ⟦19⟧ and ⟦55⟧ are both.", "Strict inequalities (⟦321⟧) give open sets; non-strict ones (⟦135⟧) give closed sets."]
question: "Classify as open, closed, both or neither: ⟦5⟧, ⟦4⟧, ⟦33⟧, ⟦344⟧, ⟦120⟧, ⟦19⟧."
answer: "Step 1, ⟦5⟧: every point has room, so open. Step 2, ⟦4⟧: contains both boundary points 0 and 1, so closed. Step 3, ⟦33⟧: 0 is a boundary point in the set, so not open; 1 is a boundary point not in the set, so not closed. Neither. Step 4, the disc with ⟦135⟧ contains its circle edge, so closed. Step 5, the strict half-space ⟦353⟧ leaves out its boundary line, so open. Step 6, ⟦19⟧ has no boundary points at all, so it is both open and closed."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

Whether a set includes its edge decides whether a best point can exist on it. A loss that improves as you approach the edge of an open set never reaches its best value. So "is this set open or closed?" is the first question to ask about a constraint.

## The definitions

- **Open**: every point of the set is an interior point. No point of the set sits on its edge.
- **Closed**: the set contains all of its boundary points. It includes its whole edge.

<div class="ex">
<div class="lab">Worked example</div>

**four sets**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> is open. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span> is closed.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> is neither: it contains the boundary point 0 but not the boundary point 1.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> is both: it has no boundary at all. So is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∅</mi></mrow></math></span>. "Not open" does not mean "closed".

</div>

~~~dg-raw
%%viz f_region kind=disk c=0,0 r=1 x=-2,2 y=-2,2 edge=open lab=open | An open disk: the dashed edge is left out, so every point inside has room around it.
~~~

~~~dg-raw
%%viz f_region kind=disk c=0,0 r=1 x=-2,2 y=-2,2 edge=closed lab=closed | A closed disk: the solid edge is included.
~~~

## A quick rule

Sets defined by **strict** inequalities with continuous functions (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&gt;</mo></mrow></math></span>) are open. Sets defined by **non-strict** ones (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≤</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≥</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo></mrow></math></span>) are closed.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>:</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>&lt;</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, the open disc. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>:</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>≤</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, the closed disc.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>:</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo>&gt;</mo><mn>0</mn><mo stretchy="false">}</mo></mrow></math></span>, an open half-space (a classifier's positive region).
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>p</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>k</mi></msup><mo>:</mo><msub><mi>p</mi><mi>i</mi></msub><mo>≥</mo><mn>0</mn><mo separator="true">,</mo><mtext> </mtext><msub><mo>∑</mo><mi>i</mi></msub><msub><mi>p</mi><mi>i</mi></msub><mo>=</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, the probability simplex, is closed.

## Why it matters

Softmax outputs are always strictly positive, so they live in the open set where every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>&gt;</mo><mn>0</mn></mrow></math></span>, never on the closed simplex's edge (where some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span>). A model trained to output a one-hot target can only approach it.

```
z = np.array([10.0, 0.0, 0.0])
p = np.exp(z) / np.exp(z).sum()
print(p.round(5))  # [9.9991e-01 5.0000e-05 5.0000e-05]
```

Even a big logit gap leaves small positive probabilities on the other classes.

<div class="take">
<div class="lab">Key takeaway</div>

open sets leave out their edge, closed sets keep it. Strict inequalities usually give open sets, non-strict ones closed sets.

</div>

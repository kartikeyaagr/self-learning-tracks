---
id: "06.13"
course: "ml-maths-course"
module: "06"
slug: "projecting-onto-a-line"
title: "Projecting onto a Line"
keyIdeas: ["The projection of ⟦682⟧ onto the line through ⟦364⟧ is ⟦2005⟧; for a unit vector ⟦280⟧ it is ⟦2006⟧.", "The residual ⟦2007⟧ is perpendicular to ⟦364⟧, and ⟦102⟧ is the closest point on the line to ⟦682⟧.", "Projecting onto the all-ones vector replaces every entry by the mean; subtracting it centres the data."]
question: "Project ⟦2003⟧ onto the line through ⟦2004⟧. Give the projection and residual, and check they are perpendicular."
answer: "Step 1, ⟦2024⟧ and ⟦2025⟧, so the coefficient is 2 and ⟦2026⟧. Step 2, ⟦2027⟧. Step 3, ⟦2028⟧. Step 4, the mean of ⟦682⟧ is 2: projecting onto ⟦1015⟧ replaced each entry by the mean, and ⟦118⟧ is ⟦682⟧ centred."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Projection and orthonormalization"]
tools: ["NumPy"]
topic: {"id": "06.03", "title": "Projection and orthonormalization"}
---

## The big picture

"Find the point in this set closest to my target" is everywhere in ML: fitting a model, removing a direction from embeddings, centring data. When the set is a line or a flat subspace through the origin, the answer is a **projection**: drop a perpendicular from the target onto the set.

## Deriving the formula

Points on the line are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mi>a</mi></mrow></math></span>. At the closest one, the leftover arrow <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>−</mo><mi>t</mi><mi>a</mi></mrow></math></span> is perpendicular to the line:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>t</mi><mi>a</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn><mspace width="1em"/><mo>⟹</mo><mspace width="1em"/><mi>t</mi><mo>=</mo><mfrac><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>a</mi></mrow></mfrac><mo separator="true">,</mo><mspace width="2em"/><mi>p</mi><mo>=</mo><mfrac><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>a</mi></mrow></mfrac><mtext> </mtext><mi>a</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-1,5 y=-1,5 tk=1 ln=2,-1,0,line_through_a,m,0.85 v=1,2,a,a;4,3,b,k va=0,0,2,4,p,f sg=4,3,2,4,b,d | b = (4, 3) projected onto the line through a = (1, 2) lands at p = (2, 4). The dashed residual (2, −1) meets the line at a right angle.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a slanted line**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi><mo>=</mo><mn>10</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>a</mi><mo>=</mo><mn>5</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>2</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>.

Residual <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>; check <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span>.

Distance from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> to the line: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">∥</mi><mo>=</mo><msqrt><mn>5</mn></msqrt><mo>≈</mo><mn>2.24</mn></mrow></math></span>.

</div>

```
proj = lambda a, b: (a @ b) / (a @ a) * a
a = np.array([1.0, 2]); b = np.array([4.0, 3])
p = proj(a, b)
print(p, b - p, a @ (b - p))  # [2. 4.] [ 2. -1.] 0.0
```

## Why it is the closest point

For any other point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> on the line, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>−</mo><mi>q</mi><mo>=</mo><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>p</mi><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mi>p</mi><mo>−</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></math></span>, and the two parts are perpendicular. By Pythagoras, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>b</mi><mo>−</mo><mi>q</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mi mathvariant="normal">∥</mi><mi>b</mi><mo>−</mo><mi>p</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>+</mo><mi mathvariant="normal">∥</mi><mi>p</mi><mo>−</mo><mi>q</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>≥</mo><mi mathvariant="normal">∥</mi><mi>b</mi><mo>−</mo><mi>p</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>.

## Centring is a projection

Projecting onto <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mrow><msup><mn mathvariant="normal">𝟏</mn><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow><mi>n</mi></mfrac><mn mathvariant="normal">𝟏</mn></mrow></math></span>: every entry replaced by the mean. "Centre the data" means "remove the component along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn></mrow></math></span>".

<div class="take">
<div class="lab">Key takeaway</div>

to project onto a line, scale <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi><mi mathvariant="normal">/</mi><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>a</mi></mrow></math></span>; the residual is perpendicular and its length is the distance to the line.

</div>

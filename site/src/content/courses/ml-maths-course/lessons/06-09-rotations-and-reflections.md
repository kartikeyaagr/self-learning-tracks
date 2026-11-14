---
id: "06.09"
course: "ml-maths-course"
module: "06"
slug: "rotations-and-reflections"
title: "Rotations and Reflections"
keyIdeas: ["Every ⟦838⟧ orthogonal matrix is a rotation (⟦867⟧) or a reflection (⟦868⟧).", "Rotation by ⟦875⟧: ⟦1965⟧.", "Householder reflection across the hyperplane perpendicular to unit ⟦280⟧: ⟦1963⟧."]
question: "Let ⟦1962⟧. Write ⟦1963⟧, apply it to ⟦758⟧, and give ⟦1964⟧."
answer: "Step 1, ⟦1974⟧, so ⟦1975⟧. Step 2, ⟦1976⟧: mirrored across the vertical axis. Step 3, ⟦1977⟧, as for every reflection."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Orthogonal structure"]
tools: ["NumPy"]
topic: {"id": "06.02", "title": "Orthogonal structure"}
---

## The big picture

Orthogonal matrices come in two kinds: rotations, which keep orientation, and reflections, which flip it like a mirror. Numerical algorithms are built from both, especially from Householder reflections.

## Rotations

~~~dg-raw
%%viz f_plane x=-5,5 y=-1,6 tk=1 v=3,4,x,k;-0.46,4.96,R₃₀x,a;-4,3,R₉₀x,f | Rotating x = (3, 4) by 30 degrees and by 90 degrees. The length stays 5 every time.
~~~

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>R</mi><mi>θ</mi></msub><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mi>sin</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>sin</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>
turns every vector counterclockwise by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><msub><mi>R</mi><mi>θ</mi></msub><mo>=</mo><msup><mrow><mi>cos</mi><mo>⁡</mo></mrow><mn>2</mn></msup><mi>θ</mi><mo>+</mo><msup><mrow><mi>sin</mi><mo>⁡</mo></mrow><mn>2</mn></msup><mi>θ</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**a 90 degree rotation**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi><mo>=</mo><mn>90</mn><mi mathvariant="normal">°</mi></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>4</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mo>−</mo><mn>4</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span>: turned by exactly 90 degrees.

</div>

## Reflections

For a unit vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span>, the **Householder matrix**
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>H</mi><mo>=</mo><mi>I</mi><mo>−</mo><mn>2</mn><mi>u</mi><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>
flips the component of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> and keeps the rest. It is symmetric, orthogonal, and its own inverse (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>H</mi><mn>2</mn></msup><mo>=</mo><mi>I</mi></mrow></math></span>).

~~~dg-raw
%%viz f_plane x=-5,5 y=-1,5 tk=1 v=3,4,x,k;-3,4,Hx,f sg=0,-1,0,5,m,d | The Householder reflection with u = (1, 0) mirrors x = (3, 4) across the vertical axis (dashed) to (−3, 4).
~~~

```
u = np.array([1.0, 0]); H = np.eye(2) - 2 * np.outer(u, u)
x = np.array([3.0, 4])
print(H @ x, round(np.linalg.det(H)))  # [-3.  4.] -1
print(np.allclose(H @ H, np.eye(2)))  # True
```

## Permutations

A permutation matrix (the identity with rows reordered) is also orthogonal. It reorders coordinates; its determinant is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>+</mo><mn>1</mn></mrow></math></span> or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> depending on whether the reordering is an even or odd number of swaps.

<div class="take">
<div class="lab">Key takeaway</div>

rotations have determinant 1, reflections <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span>; Householder reflections <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>−</mo><mn>2</mn><mi>u</mi><msup><mi>u</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> are the building block of stable QR.

</div>

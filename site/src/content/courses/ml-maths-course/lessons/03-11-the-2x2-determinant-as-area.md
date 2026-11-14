---
id: "03.11"
course: "ml-maths-course"
module: "03"
slug: "the-2x2-determinant-as-area"
title: "The 2x2 Determinant as Area"
keyIdeas: ["For ⟦858⟧, ⟦859⟧.", "⟦860⟧ is the area of the parallelogram the unit square turns into, so every area is multiplied by ⟦860⟧.", "In ⟦233⟧ dimensions, ⟦860⟧ is the volume of the image of the unit cube."]
question: "Compute ⟦856⟧ for ⟦844⟧, ⟦845⟧ and ⟦857⟧. What area does a disk of area 2 have after the third map?"
answer: "Step 1, ⟦871⟧. Step 2, ⟦872⟧: a shear keeps area. Step 3, ⟦873⟧. Step 4, every area is multiplied by 5, so the disk's image has area 10."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Determinants and singularity"]
tools: ["NumPy"]
topic: {"id": "03.02", "title": "Determinants and singularity"}
---

## The big picture

A matrix stretches some directions and squashes others. The **determinant** sums that up in one number: by how much does area (or volume) change? It appears in Gaussian densities, in change-of-variable formulas for probability, and as the test for whether a matrix can be undone.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>det</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>a</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>d</mi></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mi>a</mi><mi>d</mi><mo>−</mo><mi>b</mi><mi>c</mi></mrow></math></span></span>
The unit square (area 1) becomes the parallelogram with sides <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>b</mi><mo separator="true">,</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></math></span>, and that parallelogram has area <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>a</mi><mi>d</mi><mo>−</mo><mi>b</mi><mi>c</mi><mi mathvariant="normal">∣</mi></mrow></math></span>.

~~~dg-raw
%%viz m_det live | Change the four entries. The shaded shape is where the unit square lands; its area is |det A|.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**four matrices, four areas**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: the square becomes a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>3</mn></mrow></math></span> rectangle. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mn>6</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: a shear. Base and height are still 1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mn>1</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: a mirror. Same area, flipped. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mo>−</mo><mn>1</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: both columns point the same way, so the square is squashed onto a line. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mn>0</mn></mrow></math></span>.

</div>

```
for M in ([[2, 0], [0, 3]], [[1, 1], [0, 1]], [[0, 1], [1, 0]], [[1, 2], [2, 4]]):
    print(round(np.linalg.det(np.array(M, float)), 6))
```

## Every area scales the same way

Any shape can be filled with tiny squares, and each tiny square's area is multiplied by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span>. So every area is multiplied by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span>, whatever the shape.

## Higher dimensions

In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> dimensions, the unit cube becomes a slanted box (a parallelepiped), and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span> is its volume. The formula is longer (lesson 03.15), but the meaning is the same.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mi>a</mi><mi>d</mi><mo>−</mo><mi>b</mi><mi>c</mi></mrow></math></span> in 2D, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span> is the factor by which the map scales every area.

</div>

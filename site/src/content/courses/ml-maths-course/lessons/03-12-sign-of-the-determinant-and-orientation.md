---
id: "03.12"
course: "ml-maths-course"
module: "03"
slug: "sign-of-the-determinant-and-orientation"
title: "Sign of the Determinant and Orientation"
keyIdeas: ["A positive determinant keeps orientation; a negative one flips it, like a mirror.", "Swapping two columns (or rows) of a matrix flips the sign of its determinant.", "Rotations have determinant ⟦878⟧; reflections have ⟦499⟧."]
question: "What is ⟦874⟧ of the rotation by angle ⟦875⟧, ⟦876⟧? And of the reflection ⟦877⟧?"
answer: "Step 1, rotation: ⟦886⟧. Area and orientation are kept. Step 2, reflection: ⟦887⟧. Area is kept, orientation flipped."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Determinants and singularity"]
tools: ["NumPy"]
topic: {"id": "03.02", "title": "Determinants and singularity"}
---

## The big picture

The determinant has a sign, and the sign carries information: whether the map turns space over like a mirror. That matters for volume formulas in probability, and it is how the determinant keeps track of row swaps in algorithms.

## Orientation

Walk around the unit square anticlockwise: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>→</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>→</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>→</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. After applying <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, walk the image corners in the same order.

- If the walk is still anticlockwise, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>.
- If it is now clockwise, the map is a mirror image and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi><mo>&lt;</mo><mn>0</mn></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-2,3 y=-2,3 pg=0:0_1:0_1:1_0:1,m;0:0_1:0_1:-1_0:-1,f v=1,0,Ae₁;0,-1,Ae₂,f | The reflection [[1, 0], [0, −1]] flips the square below the axis. Same area, opposite orientation: det = −1.
~~~

## Swaps flip the sign

Swapping the two columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> swaps the roles of the two sides of the parallelogram, which reverses the walk:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>det</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>a</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>d</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mi>b</mi><mi>c</mi><mo>−</mo><mi>a</mi><mi>d</mi><mo>=</mo><mo>−</mo><mi>det</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>a</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>c</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>d</mi></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>

```
A = np.array([[3.0, 1], [1, 2]])
print(round(np.linalg.det(A), 6), round(np.linalg.det(A[:, ::-1]), 6))  # 5.0 -5.0
```

The same holds for swapping rows. Row-reduction algorithms use this to track the sign.

<div class="ex">
<div class="lab">Worked example</div>

**rotations and reflections**

Rotation by 90 degrees, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mn>0</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>1</mn></mrow></math></span>.

Mirror in the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>-axis, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mo>−</mo><mn>1</mn></mrow></math></span>.

Two mirrors in a row: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo>+</mo><mn>1</mn></mrow></math></span>. Two reflections make a rotation.

</div>

## Where this matters

In probability, a change of variables uses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi mathvariant="normal">∣</mi></mrow></math></span>, the absolute value, because densities can't be negative. In geometry code (meshes, graphics), the sign tells you if a triangle has been flipped inside out.

<div class="take">
<div class="lab">Key takeaway</div>

the sign of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> says whether the map keeps orientation (+) or mirrors it (−). Swapping two rows or columns flips the sign.

</div>

---
id: "03.34"
course: "ml-maths-course"
module: "03"
slug: "homogeneous-coordinates"
title: "Homogeneous Coordinates"
keyIdeas: ["Appending a 1 to ⟦7⟧ turns the affine map ⟦1245⟧ into one matrix product with ⟦1246⟧.", "Composing affine maps then becomes multiplying these bigger matrices.", "In ML, the same trick \"absorbs the bias\" into the weight matrix."]
question: "Write the homogeneous matrix for ⟦1221⟧, ⟦1222⟧, and apply it to ⟦733⟧."
answer: "Step 1, ⟦1261⟧. Step 2, ⟦1262⟧. Step 3, drop the final 1: ⟦1263⟧, the same as ⟦1245⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Affine transformations"]
tools: ["NumPy"]
topic: {"id": "03.05", "title": "Affine transformations"}
---

## The big picture

Affine maps are not matrix products, because of the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>+</mo><mi>b</mi></mrow></math></span>. A simple trick fixes that: add one extra coordinate that is always 1. Graphics engines, camera models and robotics use this everywhere, and ML uses it to treat the bias as just another weight.

## The trick

Append a 1 to the input: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>~</mo></mover><mo>=</mo><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Then
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>y</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>A</mi></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>b</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><msup><mn>0</mn><mi mathvariant="normal">⊤</mi></msup></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi>x</mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>
Multiply it out: the top block gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo>⋅</mo><mn>1</mn></mrow></math></span>, and the bottom row gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>1</mn></mrow></math></span>.

```
A = np.array([[2, 0], [0, 1]]); b = np.array([1, -1])
M = np.block([[A, b[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])
print(M @ np.array([2, 2, 1]))  # [5. 1. 1.]
```

## Composition is multiplication

Two affine maps <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>f</mi><mn>2</mn></msub></mrow></math></span> with homogeneous matrices <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>M</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>M</mi><mn>2</mn></msub></mrow></math></span>: doing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mn>1</mn></msub></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mn>2</mn></msub></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>M</mi><mn>2</mn></msub><msub><mi>M</mi><mn>1</mn></msub></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**shift then rotate**

Shift by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, then rotate by 90 degrees.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>M</mi><mn>1</mn></msub><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>M</mi><mn>2</mn></msub><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

The origin: shifted to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, then rotated to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. And <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>M</mi><mn>2</mn></msub><msub><mi>M</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><msub><mi>M</mi><mn>2</mn></msub><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

## Absorbing the bias

In ML, add a constant-1 feature to every input and put <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> as an extra column of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo>=</mo><msup><mi>W</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></mrow></math></span> with no separate bias:

```
W = np.array([[1.0, 2], [3, 4]]); b = np.array([0.5, -1])
x = np.array([1.0, 1])
Wp = np.column_stack([W, b]); xp = np.append(x, 1)
print(W @ x + b, Wp @ xp)  # [3.5 6. ] [3.5 6. ]
```

Some derivations (like least squares) are simpler with the bias absorbed this way.

<div class="take">
<div class="lab">Key takeaway</div>

add a coordinate that is always 1, and affine maps become matrix products.

</div>

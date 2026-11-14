---
id: "03.32"
course: "ml-maths-course"
module: "03"
slug: "low-rank-updates-and-woodbury"
title: "Low-Rank Updates and Woodbury"
keyIdeas: ["The Woodbury identity: ⟦1197⟧.", "If ⟦8⟧ is easy to invert and ⟦207⟧ is ⟦1198⟧ with small ⟦1199⟧, the only real inversion is ⟦1200⟧.", "The rank-one case is the Sherman-Morrison formula."]
question: "⟦8⟧ is a ⟦1092⟧ diagonal matrix and ⟦207⟧ is ⟦1195⟧. Roughly how expensive is inverting ⟦1196⟧ directly versus with Woodbury?"
answer: "Step 1, directly: about ⟦1217⟧ operations. Step 2, with Woodbury: inverting the diagonal ⟦8⟧ costs ⟦233⟧; the ⟦1200⟧ matrix ⟦1218⟧ costs about ⟦1219⟧ to form and ⟦1220⟧ to invert. Step 3, so about a million operations instead of a trillion."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Matrix invertibility"]
tools: ["NumPy"]
topic: {"id": "03.04", "title": "Matrix invertibility"}
---

## The big picture

Sometimes you know how to invert <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> cheaply, and then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> changes by a "thin" piece: one new data point, a few new features, a low-rank correction. The Woodbury identity lets you update the inverse without starting over.

## The identity

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><mi>U</mi><mi>C</mi><mi>V</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>−</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>U</mi><mtext> </mtext><mo stretchy="false">(</mo><msup><mi>C</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>+</mo><mi>V</mi><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>U</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mtext> </mtext><mi>V</mi><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span></span>
It looks scary, but the point is simple. Suppose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>n</mi></mrow></math></span> and easy to invert (say diagonal), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>k</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>×</mo><mi>k</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>×</mo><mi>n</mi></mrow></math></span>, with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> much smaller than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>. Then the only real inversion on the right is the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>×</mo><mi>k</mi></mrow></math></span> matrix in the middle.

```
rng = np.random.default_rng(0)
n, k = 6, 2
A = np.diag(rng.uniform(1, 2, n)); U = rng.normal(size=(n, k)); C = np.eye(k); V = U.T
Ai = np.diag(1 / np.diag(A))
wood = Ai - Ai @ U @ np.linalg.inv(np.linalg.inv(C) + V @ Ai @ U) @ V @ Ai
print(np.allclose(wood, np.linalg.inv(A + U @ C @ V)))  # True
```

## Rank one: Sherman-Morrison

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mo>=</mo><mi>u</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi><mo>=</mo><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo>=</mo><mn>1</mn></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><mi>u</mi><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>−</mo><mfrac><mrow><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>u</mi><mtext> </mtext><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow><mrow><mn>1</mn><mo>+</mo><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>u</mi></mrow></mfrac></mrow></math></span></span>
The middle "matrix" is a single number, so the update costs only a few matrix-vector products.

<div class="ex">
<div class="lab">Worked example</div>

**Sherman-Morrison with numbers**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>I</mi></mrow></math></span> (2 by 2), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi><mo>=</mo><mi>v</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>+</mo><mi>u</mi><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>u</mi><mo>=</mo><mn>2</mn></mrow></math></span>, so the inverse is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>−</mo><mfrac><mn>1</mn><mn>3</mn></mfrac><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo>=</mo><mfrac><mn>1</mn><mn>3</mn></mfrac><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

Check with the 2 by 2 formula: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mo>=</mo><mn>3</mn></mrow></math></span>, inverse <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mn>3</mn></mfrac><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mo>−</mo><mn>1</mn></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>. Same.

</div>

## Where it is used

Sparse Gaussian-process approximations, Kalman filters, online least squares (adding one data point at a time) and some second-order optimisers all use this trick.

<div class="take">
<div class="lab">Key takeaway</div>

when a matrix changes by a low-rank piece, update its inverse with Woodbury instead of inverting again.

</div>

---
id: "03.30"
course: "ml-maths-course"
module: "03"
slug: "algebra-of-inverses"
title: "Algebra of Inverses"
keyIdeas: ["⟦1160⟧: undo the last step first.", "⟦1161⟧ and ⟦1162⟧.", "You almost never need ⟦1119⟧ itself in code, only products like ⟦1120⟧."]
question: "Simplify ⟦1158⟧ in terms of ⟦1119⟧ and ⟦1159⟧."
answer: "Step 1, reverse order: ⟦1172⟧. Step 2, ⟦1161⟧. Step 3, so ⟦1173⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Matrix invertibility"]
tools: ["NumPy"]
topic: {"id": "03.04", "title": "Matrix invertibility"}
---

## The big picture

Inverses follow a handful of rules that make long derivations, such as the least-squares formula or the Gaussian posterior, manageable. The one people get wrong most often is the reversed order for products.

## The rules

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>A</mi><mi>B</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>. Undo the last step first, like taking off shoes before socks.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><mo stretchy="false">(</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>c</mi><mi>A</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><mfrac><mn>1</mn><mi>c</mi></mfrac><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo>=</mo><mi>A</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**the reverse-order rule with numbers**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mi mathvariant="normal">d</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">a</mi><mi mathvariant="normal">g</mi></mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> undoes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>B</mi></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> in general does not.

</div>

```
A = np.diag([2.0, 1]); B = np.array([[1.0, 1], [0, 1]])
inv = np.linalg.inv
print(np.allclose(inv(A @ B), inv(B) @ inv(A)))  # True
print(np.allclose(inv(A @ B), inv(A) @ inv(B)))  # False
```

## Why not compute inverses

What you usually need is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>b</mi></mrow></math></span>. Factor <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> (LU or Cholesky) and solve; it is faster and more accurate. Exceptions: tiny fixed-size matrices (as in graphics), or when the inverse's entries are what you want to read, such as a precision matrix in statistics.

<div class="ex">
<div class="lab">Worked example</div>

**reading a formula as solves**

The least-squares coefficients are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>β</mi><mo>^</mo></mover><mo>=</mo><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span>.

In code this is "solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mo stretchy="false">)</mo><mi>β</mi><mo>=</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span>", or better, `np.linalg.lstsq(X, y)`. The inverse in the formula is notation, not an instruction.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

reverse the order when inverting a product, and read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>b</mi></mrow></math></span> as "solve", not "invert then multiply".

</div>

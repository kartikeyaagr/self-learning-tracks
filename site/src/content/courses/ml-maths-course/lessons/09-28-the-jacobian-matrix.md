---
id: "09.28"
course: "ml-maths-course"
module: "09"
slug: "the-jacobian-matrix"
title: "The Jacobian Matrix"
keyIdeas: ["The Jacobian of ⟦2742⟧ is the ⟦961⟧ matrix ⟦2785⟧.", "Row ⟦274⟧ is the gradient of output ⟦274⟧; column ⟦82⟧ says how all outputs respond to input ⟦82⟧.", "For a linear map ⟦2746⟧ the Jacobian is ⟦1259⟧."]
question: "Compute the Jacobian of ⟦2784⟧ at ⟦1049⟧."
answer: "Step 1, row 1 from ⟦2795⟧: ⟦2796⟧. Step 2, row 2 from ⟦2797⟧: ⟦732⟧. Step 3, ⟦2798⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Jacobians and local approximation"]
tools: ["NumPy"]
topic: {"id": "09.04", "title": "Jacobians and local approximation"}
---

## The big picture

The derivative of a vector-valued map is a matrix: every output's sensitivity to every input. Backpropagation is built from these matrices.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>J</mi><mi>F</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi mathvariant="normal">∇</mi><msubsup><mi>F</mi><mn>1</mn><mi mathvariant="normal">⊤</mi></msubsup></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi><mi mathvariant="normal">⋮</mi><mpadded height="0em" voffset="0em"><mspace mathbackground="black" width="0em" height="1.5em"></mspace></mpadded></mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mi mathvariant="normal">∇</mi><msubsup><mi>F</mi><mi>m</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow><mo separator="true">,</mo><mspace width="2em"/><msub><mi>J</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><msub><mi>F</mi><mi>i</mi></msub></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>j</mi></msub></mrow></mfrac></mrow></math></span></span>
One row per output, one column per input.

```
F = lambda v: np.array([v[0] ** 2 * v[1], v[0] + 3 * v[1]])
x = np.array([1.0, 2]); h = 1e-6
J = np.column_stack([(F(x + h * e) - F(x - h * e)) / (2 * h) for e in np.eye(2)])
print(np.round(J, 4))
```

## Special cases

| Map | Jacobian |
| --- | --- |
| scalar <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>:</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>→</mo><mi mathvariant="normal">ℝ</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>×</mo><mi>n</mi></mrow></math></span> row, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><msup><mi>f</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> |
| linear <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi></mrow></math></span> |
| elementwise <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ϕ</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">diag</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>ϕ</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msup><mi>ϕ</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><msub><mi>x</mi><mi>n</mi></msub><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math></span> |
| softmax | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">diag</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>p</mi><mo stretchy="false">)</mo><mo>−</mo><mi>p</mi><msup><mi>p</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> |

<div class="ex">
<div class="lab">Worked example</div>

**the softmax Jacobian**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0.3</mn><mo separator="true">,</mo><mn>0.2</mn><mo stretchy="false">)</mo></mrow></math></span>: diagonal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>i</mi></msub><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msub><mi>p</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mn>0.25</mn><mo separator="true">,</mo><mn>0.21</mn><mo separator="true">,</mo><mn>0.16</mn></mrow></math></span>; off-diagonal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><msub><mi>p</mi><mi>i</mi></msub><msub><mi>p</mi><mi>j</mi></msub></mrow></math></span>.

Every row sums to 0: raising all logits together does not change the probabilities.

</div>

```
p = np.array([0.5, 0.3, 0.2])
J = np.diag(p) - np.outer(p, p)
print(np.round(J.sum(axis=1), 12) + 0)  # [0. 0. 0.]
```

<div class="take">
<div class="lab">Key takeaway</div>

the Jacobian stacks the gradients of every output; for a linear layer it is the weight matrix.

</div>

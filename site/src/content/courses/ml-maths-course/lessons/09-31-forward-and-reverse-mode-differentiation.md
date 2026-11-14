---
id: "09.31"
course: "ml-maths-course"
module: "09"
slug: "forward-and-reverse-mode-differentiation"
title: "Forward and Reverse Mode Differentiation"
keyIdeas: ["Reverse mode (backprop) multiplies the Jacobian chain from the loss end: each step is a cheap vector-Jacobian product ⟦2826⟧.", "Forward mode multiplies from the input end: each pass gives one Jacobian-vector product ⟦2827⟧.", "For many inputs and one output (training), reverse mode gets the full gradient in one pass; it pays in memory for stored activations."]
question: "A model maps ⟦1564⟧ parameters to a scalar loss. How many passes does each mode need for the full gradient?"
answer: "Step 1, forward mode gives one directional derivative per pass, so ⟦1564⟧ passes. Step 2, reverse mode gives the whole gradient in one backward pass, at a few times the cost of the forward pass. Step 3, forward mode wins when inputs are few and outputs many."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Jacobians and local approximation"]
tools: ["NumPy"]
topic: {"id": "09.04", "title": "Jacobians and local approximation"}
---

## The big picture

Matrix multiplication is associative, so the Jacobian chain can be multiplied in any order. The cost is wildly different. Choosing the right order is why training is feasible at all.

## Two orders

~~~dg-flow Reverse mode: start from the scalar loss and multiply by one Jacobian at a time.
dL/dL = 1 -(times J_K)-> dL/dh_(K-1) -(times J_(K-1))-> ... -(times J_1)-> *dL/dx
~~~

- **Reverse**: start from the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>×</mo><msub><mi>d</mi><mi>K</mi></msub></mrow></math></span> row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi mathvariant="normal">ℓ</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><msub><mi>h</mi><mi>K</mi></msub></mrow></math></span>. Row times matrix is a row, so every intermediate is a vector.
- **Forward**: start from the input side. To get all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> input derivatives you carry an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>-column matrix.

<div class="ex">
<div class="lab">Worked example</div>

**counting the work**

Three layers with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1000</mn><mo>×</mo><mn>1000</mn></mrow></math></span> Jacobians and a scalar loss.

Reverse: three row-times-matrix products, about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>×</mo><mn>1</mn><msup><mn>0</mn><mn>6</mn></msup></mrow></math></span> multiply-adds.

Forward, building the full input Jacobian: about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>1</mn><msup><mn>0</mn><mn>9</mn></msup></mrow></math></span>. Roughly 700 times more.

</div>

## Jacobians are never built

Each layer knows its vector-Jacobian product directly:

| Layer | VJP with upstream <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi></mrow></math></span> |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>W</mi><mi mathvariant="normal">⊤</mi></msup><mi>g</mi></mrow></math></span> |
| elementwise <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ϕ</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo>⊙</mo><msup><mi>ϕ</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> |
| softmax | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⊙</mo><mo stretchy="false">(</mo><mi>g</mi><mo>−</mo><mi>g</mi><mo>⋅</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></math></span> |

```
p = np.array([0.5, 0.3, 0.2]); g = np.array([1.0, -2, 0.5])
J = np.diag(p) - np.outer(p, p)
print(np.allclose(g @ J, p * (g - g @ p)))  # True
```

<div class="take">
<div class="lab">Key takeaway</div>

reverse mode multiplies vector-Jacobian products from the loss backward and gets the whole gradient in one pass.

</div>

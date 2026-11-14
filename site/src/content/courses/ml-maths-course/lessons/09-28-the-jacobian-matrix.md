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
steps: [{"kind": "explain", "title": "Many outputs", "body": "Take $F(x, y) = (x^2 y,\\; x + 3y)$: two inputs, two outputs. Each output has its own gradient. Stack them as rows and you get the **Jacobian**:\n\n$$J_{ij} = \\frac{\\partial F_i}{\\partial x_j}$$\n\nFor $F: \\mathbb{R}^n \\to \\mathbb{R}^m$, $J$ is $m \\times n$: one row per output, one column per input.", "question": {"q": "$F$ takes 3 inputs and returns 2 outputs. What shape is its Jacobian?", "options": ["3 × 2", "2 × 3", "3 × 3"], "answer": 1, "why": "Rows are outputs (2) and columns are inputs (3), so $2 \\times 3$. That shape lets $J$ multiply an input change to give an output change.", "hint": "One row per output, one column per input."}}, {"kind": "explain", "title": "Rows are gradients", "body": "Row $i$ of the Jacobian is the gradient of output $i$. For $F_1 = x^2 y$, the gradient is $(2xy, x^2)$, which is $(4, 1)$ at $(1, 2)$.\n\nYou build the matrix one output at a time: take the partials of that output with respect to each input in order, and lay them out along the row.", "question": {"q": "At $(1, 2)$, what is row 2, the gradient of $F_2 = x + 3y$?", "options": ["(3, 1)", "(1, 1)", "(1, 3)"], "answer": 2, "why": "$\\partial F_2/\\partial x = 1$ and $\\partial F_2/\\partial y = 3$, so the row is $(1, 3)$, the same at every point since $F_2$ is linear.", "hint": "The $x$ partial comes first in the row, and the coefficient on $y$ matters."}}, {"kind": "explain", "title": "Columns are inputs", "body": "Column $j$ says how **all** outputs respond to input $j$. Near a point, a small input change $\\Delta$ moves the output by about $J\\Delta$.\n\nAt $(2, 1)$, $F_1 = x^2 y$ has gradient $(2xy, x^2) = (4, 4)$ and $F_2$ has $(1, 3)$, so\n\n$$J = \\begin{pmatrix} 4 & 4 \\\\ 1 & 3 \\end{pmatrix}$$", "question": {"q": "At $(2, 1)$ you nudge $x$ by 0.1. Roughly how does $F$ change?", "options": ["(0.4, 0.1)", "(0.4, 0.4)", "(0.1, 0.3)"], "answer": 0, "why": "The change is $0.1$ times column 1, $(4, 1)$, which gives $(0.4, 0.1)$. $(0.4, 0.4)$ reads row 1 instead.", "hint": "Nudging $x$ only uses the column for $x$."}}, {"kind": "explain", "title": "Linear layers", "body": "For a linear layer $F(x) = Wx + b$, output $i$ is $\\sum_j W_{ij} x_j + b_i$. Its partial with respect to $x_j$ is just $W_{ij}$, and $b$ disappears because it is constant.\n\nSo the Jacobian of $Wx + b$ is $W$, everywhere. That is why weight matrices appear directly in the backward pass of a network.", "question": {"q": "What is the Jacobian of $Wx + b$, with $W$ of shape $3 \\times 2$?", "options": ["W⊤", "W", "W + b"], "answer": 1, "why": "$J_{ij} = W_{ij}$, already the right shape: 3 outputs by 2 inputs. $W^\\top$ would be $2 \\times 3$, and $b$ has zero derivative.", "hint": "Differentiate output $i$ with respect to input $j$. What is left?"}}, {"kind": "explain", "title": "Elementwise maps", "body": "An **elementwise** function such as ReLU applies $\\phi$ to each entry on its own: output $i$ depends only on input $i$. So every off-diagonal partial is 0, and\n\n$$J = \\text{diag}(\\phi'(x_1), \\dots, \\phi'(x_n))$$\n\nFor ReLU, $\\phi'$ is 1 for positive inputs and 0 for negative ones. At $x = (2, -1, 3)$, $J = \\text{diag}(1, 0, 1)$.", "question": {"q": "What is the Jacobian of ReLU at $x = (-2, 5)$?", "options": ["diag(−2, 5)", "[[0, 1], [0, 1]]", "diag(0, 1)"], "answer": 2, "why": "The negative input has slope 0 and the positive one slope 1, with zeros off the diagonal. $\\text{diag}(-2, 5)$ puts the inputs on the diagonal, not their slopes.", "hint": "Each output depends only on its own input, and you need ReLU's slope there, not its value."}}, {"kind": "explain", "title": "The softmax Jacobian", "body": "Softmax turns scores $z$ into probabilities $p_i = e^{z_i} / \\sum_j e^{z_j}$. Every output depends on every input, and its Jacobian is\n\n$$J = \\text{diag}(p) - pp^\\top$$\n\nFor $p = (0.5, 0.3, 0.2)$ the diagonal is $p_i(1 - p_i)$: 0.25, 0.21 and 0.16. Off the diagonal, $J_{ij} = -p_i p_j$. Every row sums to 0, because raising all scores together leaves $p$ unchanged.", "question": {"q": "For this $p$, what is $J_{12}$?", "options": ["−0.15", "0.15", "0.21"], "answer": 0, "why": "$-p_1 p_2 = -0.5 \\cdot 0.3 = -0.15$. It is negative: raising score 2 takes probability away from output 1.", "hint": "Off-diagonal entries are $-p_i p_j$. Mind the sign."}}, {"kind": "question", "title": "Recap and a check", "body": "The Jacobian of $F: \\mathbb{R}^n \\to \\mathbb{R}^m$ is $m \\times n$. Row $i$ is the gradient of output $i$, column $j$ is how the outputs respond to input $j$, and for $Wx + b$ it is $W$.", "question": {"q": "What is the Jacobian of $F(x, y) = (xy,\\; x - y)$ at $(2, 3)$?", "options": ["[[3, 1], [2, −1]]", "[[3, 2], [1, −1]]", "[[2, 3], [1, −1]]"], "answer": 1, "why": "Row 1 is $\\nabla(xy) = (y, x) = (3, 2)$, and row 2 is $\\nabla(x - y) = (1, -1)$. The first option is the transpose, with gradients as columns.", "hint": "Write the gradient of each output as a row, with the $x$ partial first."}}]
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

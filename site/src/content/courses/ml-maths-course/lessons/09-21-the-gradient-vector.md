---
id: "09.21"
course: "ml-maths-course"
module: "09"
slug: "the-gradient-vector"
title: "The Gradient Vector"
keyIdeas: ["The gradient collects all partial derivatives: ⟦2699⟧.", "It lives in the input space: one entry per input variable.", "Directional derivatives come from it: ⟦2615⟧."]
question: "For ⟦2613⟧, write ⟦2698⟧ and evaluate it at ⟦1049⟧."
answer: "Step 1, ⟦2707⟧ and ⟦2708⟧. Step 2, ⟦2709⟧. Step 3, at ⟦1049⟧: ⟦2710⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
steps: [{"kind": "explain", "title": "Collect the partials", "body": "The **gradient** puts all the partial derivatives of $f$ into one vector:\n\n$$\\nabla f(x) = \\left(\\frac{\\partial f}{\\partial x_1}, \\dots, \\frac{\\partial f}{\\partial x_n}\\right)$$\n\nThe symbol $\\nabla$ is read \"nabla\" or \"del\". For $f(x, y) = x^2 + 3xy$: $f_x = 2x + 3y$ and $f_y = 3x$, so $\\nabla f = (2x + 3y,\\; 3x)$.", "question": {"q": "What is $\\nabla f$ at $(1, 2)$?", "options": ["(8, 3)", "(3, 8)", "(2, 3)"], "answer": 0, "why": "$f_x = 2 + 6 = 8$ and $f_y = 3$. The order follows the inputs: the $x$ partial first, then the $y$ partial.", "hint": "Put $x = 1$, $y = 2$ into each partial, keeping the $3y$ term in $f_x$."}}, {"kind": "explain", "title": "One entry per input", "body": "The gradient lives in the **input space**: it has one entry for each variable the function takes in, not for anything it returns. A loss $L$ that takes a million weights and returns one number has a gradient with a million entries, the same shape as the weights.\n\nThat is why an optimiser can subtract the gradient straight from the weights.", "question": {"q": "A loss takes 5 weights and returns one number. What is $\\nabla L$?", "options": ["1 number", "5 numbers", "A 5 × 5 matrix"], "answer": 1, "why": "One partial derivative per input weight. A single number would be the loss itself, and a $5 \\times 5$ matrix would hold second derivatives.", "hint": "Count the inputs, then give each one its own partial derivative."}}, {"kind": "explain", "title": "Slope in any direction", "body": "Partial derivatives are slopes along the axes. The **directional derivative** is the slope along any unit direction $u$, and the gradient gives it with one dot product:\n\n$$D_u f = \\nabla f \\cdot u$$\n\nWith $u = (1, 0)$ you get back $f_x$. For $f = x^2 + 3xy$ at $(1, 2)$, $\\nabla f = (8, 3)$, and the direction $u = (0.6, 0.8)$ gives $4.8 + 2.4 = 7.2$.", "question": {"q": "At the same point, what is the slope in the direction $u = (0.8, -0.6)$?", "options": ["8.2", "(6.4, −1.8)", "4.6"], "answer": 2, "why": "$8 \\cdot 0.8 + 3 \\cdot (-0.6) = 6.4 - 1.8 = 4.6$. A slope is a single number, so the two products are added.", "hint": "Take the dot product of $(8, 3)$ with $u$, keeping the minus sign."}}, {"kind": "explain", "title": "Unit directions only", "body": "The formula needs $u$ to have length 1. Dot the gradient with a longer vector and the answer scales with its length: $(8, 3) \\cdot (3, 4) = 36$, five times the 7.2 that the unit vector $(0.6, 0.8)$ gives, because $(3, 4)$ has length 5.\n\nSo to get the slope \"towards\" some vector $v$, first divide $v$ by its length.", "question": {"q": "What is the slope of $f$ at $(1, 2)$ in the direction of $(1, 1)$?", "options": ["11", "≈ 7.8", "5.5"], "answer": 1, "why": "The unit vector is $(1, 1)/\\sqrt{2}$, so the slope is $11/\\sqrt{2} \\approx 7.8$. 11 skips the normalising step.", "hint": "$(1, 1)$ has length $\\sqrt{2}$, not 1, and not 2."}}, {"kind": "explain", "title": "Gradients to know", "body": "Three gradients come up constantly. They are the many-variable versions of $ax \\to a$ and $ax^2 \\to 2ax$:\n\n| $f(x)$ | $\\nabla f$ |\n| --- | --- |\n| $a^\\top x$ | $a$ |\n| $\\|x\\|^2$ | $2x$ |\n| $x^\\top A x$ ($A$ symmetric) | $2Ax$ |\n\nThe middle one follows from the partials: $\\|x\\|^2 = x_1^2 + \\dots + x_n^2$, so entry $i$ is $2x_i$.", "question": {"q": "What is $\\nabla \\|x\\|^2$ at $x = (1, -2, 3)$?", "options": ["(2, −4, 6)", "28", "(1, 4, 9)"], "answer": 0, "why": "$2x = (2, -4, 6)$. The gradient has one entry per input, so it can't be the single number 28, and $(1, 4, 9)$ squares the entries instead of differentiating.", "hint": "Differentiate $x_i^2$ with respect to $x_i$ for each entry."}}, {"kind": "explain", "title": "Checking a gradient", "body": "You can check a gradient by central differences, one coordinate at a time:\n\n```python\ng = [(f(x + h*e) - f(x - h*e)) / (2*h) for e in np.eye(n)]\n```\n\nFor $f = x^2 + 3xy$ at $(1, 2)$ this prints about $(8, 3)$. Each entry costs two evaluations of $f$, so the check is slow for big models. Engineers use it only to test an autodiff or hand-written gradient on small inputs.", "question": {"q": "A function has 1,000 inputs. How many evaluations of $f$ does this check need?", "options": ["1,000", "2,000", "2"], "answer": 1, "why": "Two per coordinate, one each side, for 1,000 coordinates. Backpropagation gets the whole gradient for about the cost of a couple of evaluations.", "hint": "Count the evaluations inside one entry of the list, then the number of entries."}}, {"kind": "question", "title": "Recap and a check", "body": "The gradient is the vector of all partial derivatives, one entry per input. Dot it with a unit vector to get the slope in that direction.", "question": {"q": "For $f(x, y) = x^2 y$, what is the slope at $(1, 3)$ in the direction $u = (0.6, 0.8)$?", "options": ["(3.6, 0.8)", "7", "4.4"], "answer": 2, "why": "$\\nabla f = (2xy, x^2) = (6, 1)$ at $(1, 3)$, and $(6, 1) \\cdot (0.6, 0.8) = 3.6 + 0.8 = 4.4$.", "hint": "Find the gradient first, then take its dot product with $u$."}}]
---

## The big picture

Put the partial derivatives in one vector and you get the **gradient**, the most important object in training. The next lessons explain what it points at.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true">(</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mn>1</mn></msub></mrow></mfrac><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>n</mi></msub></mrow></mfrac><mo fence="true">)</mo></mrow></mrow></math></span></span>
The symbol <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi></mrow></math></span> is "nabla" or "del". For a loss with a million weights, the gradient has a million entries.

```
f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
x = np.array([1.0, 2]); h = 1e-6
g = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(2)])
print(np.round(g, 4))  # [8. 3.]
```

~~~dg-raw
%%viz f_plane x=-1,10 y=-1,5 tk=1 v=8,3,∇f(1~_2)_=_(8~_3),f;8,0,∂f/∂x_=_8,m;0,3,∂f/∂y_=_3,m | The gradient at (1, 2) is the arrow built from the two partial derivatives: 8 along x and 3 along y.
~~~

## Gradients of simple functions

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>x</mi></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> symmetric) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>A</mi><mi>x</mi></mrow></math></span> |

These are the multivariable versions of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mi>x</mi><mo>→</mo><mi>a</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><msup><mi>x</mi><mn>2</mn></msup><mo>→</mo><mn>2</mn><mi>a</mi><mi>x</mi></mrow></math></span>; lesson 09.35 derives them.

<div class="take">
<div class="lab">Key takeaway</div>

the gradient is the vector of all partial derivatives, one entry per input.

</div>

---
id: "09.12"
course: "ml-maths-course"
module: "09"
slug: "introduction-to-partial-derivatives"
title: "Introduction to Partial Derivatives"
keyIdeas: ["The partial derivative ⟦2569⟧ is the slope along coordinate ⟦274⟧ with every other coordinate held fixed.", "Formally ⟦2570⟧.", "Compute it by treating the other variables as constants and differentiating as usual."]
question: "For ⟦2566⟧, find ⟦2567⟧ and ⟦2568⟧, and evaluate both at ⟦310⟧."
answer: "Step 1, treat ⟦509⟧ as a constant: ⟦2579⟧. Step 2, treat ⟦7⟧ as a constant: ⟦2582⟧. Step 3, at ⟦310⟧: ⟦2583⟧ and ⟦2584⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
steps: [{"kind": "explain", "title": "Nudge one input", "body": "Take $f(x, y) = x^2 y$ at the point $(3, 2)$, where $f = 18$. Nudge $x$ to 3.01 and keep $y$ at 2: $f$ becomes about 18.12, a change of 0.12 for a step of 0.01. That is a slope of about 12 along $x$.\n\nThis is a **partial derivative**, written $\\partial f / \\partial x$ and read \"partial f by partial x\": the slope along one coordinate while every other coordinate stays fixed. With $y$ frozen at 2, $f$ is just $2x^2$.", "question": {"q": "Now freeze $x$ at 3 instead. What is the slope along $y$ at $(3, 2)$?", "options": ["9", "12", "6"], "answer": 0, "why": "With $x = 3$ fixed, $f = 9y$, a straight line with slope 9. 12 is the slope along $x$, a different partial derivative.", "hint": "Substitute $x = 3$, so $f$ becomes a function of $y$ alone, then differentiate that."}}, {"kind": "explain", "title": "The definition", "body": "Written out, the partial derivative along coordinate $i$ is a one-variable limit:\n\n$$\\frac{\\partial f}{\\partial x_i}(x) = \\lim_{h \\to 0} \\frac{f(x + h e_i) - f(x)}{h}$$\n\nHere $e_i$ has a 1 in position $i$ and 0 everywhere else, so $x + h e_i$ moves **only** coordinate $i$, by $h$. Other notations you will meet: $f_{x_i}$ and $\\partial_i f$.", "question": {"q": "For $x = (1, 4, 2)$, what is $x + h e_2$?", "options": ["(1 + h, 4 + h, 2 + h)", "(1, 4 + h, 2)", "(1, 4h, 2)"], "answer": 1, "why": "$e_2 = (0, 1, 0)$, so $h e_2 = (0, h, 0)$ adds $h$ to the second coordinate and leaves the others alone.", "hint": "Write out $e_2$ in $\\mathbb{R}^3$, multiply it by $h$, and add entry by entry."}}, {"kind": "explain", "title": "Freeze the others", "body": "To compute a partial derivative, treat every other variable as a constant and differentiate as usual.\n\nTake $f(x, y) = x^2 y + \\sin y$. For $f_x$, $y$ is a constant: $x^2 y$ becomes $2xy$, and $\\sin y$ is a constant, so it becomes 0. That gives\n\n$$f_x = 2xy$$", "question": {"q": "For the same $f$, what is $f_y$?", "options": ["2xy + cos y", "cos y", "x² + cos y"], "answer": 2, "why": "With $x$ frozen, $x^2 y$ is a constant times $y$, giving $x^2$, and $\\sin y$ gives $\\cos y$.", "hint": "$x^2 y$ still depends on $y$: with $x$ frozen it is (some constant) $\\times\\, y$."}}, {"kind": "explain", "title": "Evaluate at a point", "body": "A partial derivative is a function. To get a number, put in the point. For $f = x^2 y + \\sin y$ you now have\n\n$$f_x = 2xy, \\qquad f_y = x^2 + \\cos y$$\n\nAt different points the two slopes can be very different. One can even be zero while the other is not: then moving along that axis doesn't change $f$ to first order, but moving along the other one does.", "question": {"q": "What are $f_x$ and $f_y$ at $(1, 0)$?", "options": ["2 and 1", "0 and 2", "0 and 1"], "answer": 1, "why": "$f_x = 2 \\cdot 1 \\cdot 0 = 0$ and $f_y = 1 + \\cos 0 = 2$. Along $x$ the surface is flat at this point; along $y$ it rises with slope 2.", "hint": "Put $x = 1$ and $y = 0$ into both formulas, and remember that $\\cos 0 = 1$."}}, {"kind": "explain", "title": "Check it numerically", "body": "You can check any partial derivative with a **central difference**: nudge one coordinate both ways by a small $h$ and divide by the gap, $2h$.\n\n```python\nf = lambda x, y: x**2 * y + np.sin(y)\nh = 1e-6\n(f(1 + h, 0) - f(1 - h, 0)) / (2 * h)   # about 0.0, which is f_x\n```\n\nThe two common slips are nudging more than one coordinate at once and dividing by $h$ instead of $2h$.", "question": {"q": "Which expression estimates $f_y$ at $(1, 0)$?", "options": ["(f(1 + h, h) − f(1 − h, −h)) / 2h", "(f(1, h) − f(1, −h)) / h", "(f(1, h) − f(1, −h)) / 2h"], "answer": 2, "why": "Only $y$ moves, from $-h$ to $+h$, a gap of $2h$. It gives about 2.0, matching $f_y = 1 + \\cos 0$.", "hint": "Hold $x$ at 1 exactly, and divide by the full distance between the two points."}}, {"kind": "explain", "title": "One per weight", "body": "A loss depends on many weights. Its partial derivative for one weight answers: if I nudge **this** weight a little, how fast does the loss change? Training needs that answer for every weight at once, so there is one partial derivative per weight.\n\nCollected into a single vector, they form the gradient, which the next lessons build on.", "question": {"q": "A loss depends on 1,000 weights. How many partial derivatives does it have?", "options": ["1", "1,000", "1,000,000"], "answer": 1, "why": "One for each weight. A million would be second derivatives, one for every pair of weights, which is a different object.", "hint": "Each partial derivative belongs to one coordinate that you nudge."}}, {"kind": "question", "title": "Recap and a check", "body": "A partial derivative is the slope along one coordinate with all the others frozen. Compute it by treating the other variables as constants, then put in the point.", "question": {"q": "For $g(x, y) = 3x + xy^2$, what are $g_x$ and $g_y$ at $(1, 2)$?", "options": ["4 and 7", "3 and 4", "7 and 4"], "answer": 2, "why": "$g_x = 3 + y^2 = 7$ and $g_y = 2xy = 4$ at $(1, 2)$.", "hint": "For $g_x$, $y^2$ is a constant multiplying $x$. For $g_y$, the $3x$ term is a constant."}}]
---

## The big picture

To improve a loss you need to know: if I nudge **this one** weight a little, how much does the loss change? That number is a partial derivative. Collect one per weight and you have the gradient, the object behind all of training.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>i</mi></msub></mrow></mfrac><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mrow><mi>lim</mi><mo>⁡</mo></mrow><mrow><mi>h</mi><mo>→</mo><mn>0</mn></mrow></munder><mfrac><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>h</mi><msub><mi>e</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mi>h</mi></mfrac></mrow></math></span></span>
<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>e</mi><mi>i</mi></msub></mrow></math></span> has a 1 in position <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> and 0 elsewhere, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>h</mi><msub><mi>e</mi><mi>i</mi></msub></mrow></math></span> nudges only coordinate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span>. The curly <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi></mrow></math></span> is read "partial". Other notations: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><msub><mi>x</mi><mi>i</mi></msub></msub></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi mathvariant="normal">∂</mi><mi>i</mi></msub><mi>f</mi></mrow></math></span>.

~~~dg-raw
%%viz part | The two slices through p, and their slopes: the partial derivatives.
~~~

## How to compute one

Freeze every other variable and differentiate as in one variable.

<div class="ex">
<div class="lab">Worked example</div>

**two partials**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mi>y</mi><mo>+</mo><mi>sin</mi><mo>⁡</mo><mi>y</mi></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>x</mi></msub></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> is a constant, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup><mi>y</mi><mo>→</mo><mn>2</mn><mi>x</mi><mi>y</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sin</mi><mo>⁡</mo><mi>y</mi><mo>→</mo><mn>0</mn></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>x</mi></msub><mo>=</mo><mn>2</mn><mi>x</mi><mi>y</mi></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>y</mi></msub></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is a constant, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mn>2</mn></msup><mi>y</mi><mo>→</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sin</mi><mo>⁡</mo><mi>y</mi><mo>→</mo><mi>cos</mi><mo>⁡</mo><mi>y</mi></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>y</mi></msub><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mi>cos</mi><mo>⁡</mo><mi>y</mi></mrow></math></span>.

</div>

```
f = lambda x, y: x ** 2 * y + np.sin(y)
h = 1e-6
print(round((f(1 + h, 0) - f(1 - h, 0)) / (2 * h), 6), round((f(1, h) - f(1, -h)) / (2 * h), 6))  # 0.0 2.0
```

<div class="take">
<div class="lab">Key takeaway</div>

a partial derivative is an ordinary derivative with every other variable frozen.

</div>

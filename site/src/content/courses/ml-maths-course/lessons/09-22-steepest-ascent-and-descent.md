---
id: "09.22"
course: "ml-maths-course"
module: "09"
slug: "steepest-ascent-and-descent"
title: "Steepest Ascent and Descent"
keyIdeas: ["⟦2711⟧, where ⟦875⟧ is the angle between ⟦280⟧ and ⟦2698⟧.", "So ⟦2698⟧ points in the direction of steepest increase, at rate ⟦2712⟧; ⟦2713⟧ is steepest decrease.", "Gradient descent ⟦2714⟧ steps in the locally best direction."]
question: "For ⟦2613⟧ at ⟦1049⟧, what is the maximum rate of increase, and in which direction is the rate zero?"
answer: "Step 1, ⟦2721⟧, so the maximum rate is ⟦2722⟧. Step 2, the rate is zero for directions perpendicular to ⟦2710⟧, such as ⟦2723⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
steps: [{"kind": "explain", "title": "Which way is uphill", "body": "Stand at $(1, 2)$ on $f(x, y) = x^2 + 3xy$, where $\\nabla f = (8, 3)$. The slope in a unit direction $u$ is $\\nabla f \\cdot u$. Since a dot product is lengths times the cosine of the angle,\n\n$$D_u f = \\|\\nabla f\\|\\,\\|u\\|\\cos\\theta = \\|\\nabla f\\|\\cos\\theta$$\n\nwhere $\\theta$ is the angle between $u$ and $\\nabla f$. Only the angle can change, so the slope is largest when $\\cos\\theta = 1$.", "question": {"q": "Which unit direction gives the largest slope at $(1, 2)$?", "options": ["(1, 0)", "(8, 3)/√73", "(0.6, 0.8)"], "answer": 1, "why": "It points along $\\nabla f$, so $\\theta = 0$, and the slope is $\\sqrt{73} \\approx 8.54$. $(1, 0)$ gives 8 and $(0.6, 0.8)$ gives 7.2.", "hint": "Which option makes the angle with $\\nabla f = (8, 3)$ zero?"}}, {"kind": "explain", "title": "Steepest ascent", "body": "At $\\theta = 0$, the slope is $\\|\\nabla f\\|$. So the gradient does two jobs: its **direction** is the direction of steepest increase, and its **length** is how steep that increase is.\n\nAt $(1, 2)$, $\\|(8, 3)\\| = \\sqrt{64 + 9} = \\sqrt{73} \\approx 8.54$. No direction climbs faster.", "question": {"q": "$\\nabla f = (3, 4)$ at some point. What is the largest possible rate of increase there?", "options": ["7", "25", "5"], "answer": 2, "why": "$\\|(3, 4)\\| = 5$. 25 is the squared length, and 7 just adds the entries.", "hint": "The steepest rate is the length of the gradient."}}, {"kind": "explain", "title": "Along the contour", "body": "At $\\theta = 90°$, $\\cos\\theta = 0$ and the slope is zero. Walking perpendicular to the gradient keeps you at the same height, along a **contour** line of $f$.\n\nSo the gradient is always perpendicular to the contour through the point. In 2D, a perpendicular to $(a, b)$ is $(b, -a)$.", "question": {"q": "$\\nabla f = (8, 3)$. Along which direction is the slope zero?", "options": ["(−8, −3)", "(3, −8)", "(8, 3)"], "answer": 1, "why": "$(8, 3) \\cdot (3, -8) = 24 - 24 = 0$, so it runs along the contour. $(-8, -3)$ is the opposite of the gradient: steepest downhill, not level.", "hint": "Find the option whose dot product with $(8, 3)$ is zero."}}, {"kind": "explain", "title": "Steepest descent", "body": "At $\\theta = 180°$, $\\cos\\theta = -1$, and the slope is $-\\|\\nabla f\\|$, the fastest possible decrease. That direction is $-\\nabla f$.\n\nThis is the whole reason gradient descent steps along minus the gradient: among all directions, it lowers the function fastest, at least locally.", "question": {"q": "$\\nabla f = (3, 4)$. What is the slope in the unit direction $-\\nabla f / \\|\\nabla f\\|$?", "options": ["−5", "−25", "0"], "answer": 0, "why": "$\\cos\\theta = -1$, so the slope is $-\\|\\nabla f\\| = -5$, the steepest possible drop.", "hint": "Use $\\|\\nabla f\\|\\cos\\theta$ with $\\theta = 180°$."}}, {"kind": "explain", "title": "A gradient descent step", "body": "**Gradient descent** repeats one update:\n\n$$x_{k+1} = x_k - \\eta\\,\\nabla f(x_k)$$\n\n$\\eta$ (\"eta\") is the **learning rate**, the step size. From $(1, 2)$ with $\\nabla f = (8, 3)$ and $\\eta = 0.1$, the step is $-0.1 \\cdot (8, 3) = (-0.8, -0.3)$.", "question": {"q": "Where does that step land?", "options": ["(1.8, 2.3)", "(0.2, 1.7)", "(0.2, 2.3)"], "answer": 1, "why": "$(1 - 0.8, 2 - 0.3) = (0.2, 1.7)$. Adding the gradient would climb uphill instead.", "hint": "Subtract $\\eta \\nabla f$ from both coordinates."}}, {"kind": "explain", "title": "How much the loss drops", "body": "To first order, a step $-\\eta\\nabla f$ changes $f$ by $\\nabla f \\cdot (-\\eta\\nabla f) = -\\eta\\|\\nabla f\\|^2$. For the step above that predicts a drop of $0.1 \\cdot 73 = 7.3$.\n\nThe real drop is smaller: $f(1, 2) = 7$ and $f(0.2, 1.7) = 1.06$, a drop of 5.94. The gradient only describes $f$ near the point, so large steps can disappoint or even overshoot.", "question": {"q": "$\\nabla f = (3, 4)$ and $\\eta = 0.01$. What drop does the first-order estimate predict?", "options": ["0.05", "2.5", "0.25"], "answer": 2, "why": "$\\eta\\|\\nabla f\\|^2 = 0.01 \\cdot 25 = 0.25$. 0.05 uses the length instead of the squared length.", "hint": "The estimate is $\\eta$ times the squared length of the gradient."}}, {"kind": "question", "title": "Recap and a check", "body": "The slope in direction $u$ is $\\|\\nabla f\\|\\cos\\theta$: largest along $\\nabla f$, zero along the contour, and most negative along $-\\nabla f$. Gradient descent steps along $-\\nabla f$.", "question": {"q": "At the current weights $\\nabla L = (0, -2)$. What does one gradient descent step do?", "options": ["Increases the second weight", "Decreases the second weight", "Changes only the first weight"], "answer": 0, "why": "$-\\eta\\nabla L = (0, 2\\eta)$, so the second weight goes up and the first stays put. Since the loss falls as the second weight rises, going up is downhill.", "hint": "Subtract $\\eta(0, -2)$. What does minus a negative number do?"}}]
---

## The big picture

Why does gradient descent step along minus the gradient? Because, of all directions, it lowers the function fastest. This lesson proves that in one line.

## The cosine argument

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>D</mi><mi>u</mi></msub><mi>f</mi><mo>=</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo>⋅</mo><mi>u</mi><mo>=</mo><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mi mathvariant="normal">∥</mi><mi>cos</mi><mo>⁡</mo><mi>θ</mi></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi><mo>=</mo><mn>0</mn></mrow></math></span>: rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mi mathvariant="normal">∥</mi></mrow></math></span>, the largest. **Steepest uphill.**
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi><mo>=</mo><mn>90</mn><mi mathvariant="normal">°</mi></mrow></math></span>: rate 0. You walk along the contour.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi><mo>=</mo><mn>180</mn><mi mathvariant="normal">°</mi></mrow></math></span>: rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mi mathvariant="normal">∥</mi></mrow></math></span>. **Steepest downhill.**

~~~dg-raw
%%viz grad live | Turn the direction u and compare ∇f · u with ‖∇f‖. Switch on the field to see ∇f everywhere.
~~~

```
g = np.array([8.0, 3])
for u in ([1, 0], [0, 1], [0.6, 0.8], np.array([3, -8]) / np.sqrt(73)):
    print(np.round(g @ np.array(u), 4))
print(round(np.linalg.norm(g), 4))  # 8.544
```

## Gradient descent

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>x</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>x</mi><mi>k</mi></msub><mo>−</mo><mi>η</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>k</mi></msub><mo stretchy="false">)</mo></mrow></math></span></span>
To first order the loss drops by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>, the most any step of length <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mi mathvariant="normal">∥</mi></mrow></math></span> can achieve.

<div class="take">
<div class="lab">Key takeaway</div>

the gradient points steepest uphill with slope <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><mi mathvariant="normal">∥</mi></mrow></math></span>; minus the gradient is the best local descent direction.

</div>

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

---
id: "09.37"
course: "ml-maths-course"
module: "09"
slug: "the-least-squares-gradient-for-matrices"
title: "The Least-Squares Gradient for Matrices"
keyIdeas: ["For ⟦2920⟧, ⟦2921⟧.", "Setting it to zero gives ⟦2922⟧: one least-squares problem per column of ⟦512⟧.", "Gradient descent on this loss is sensitive to feature scaling, because the Hessian is ⟦895⟧."]
question: "Derive ⟦2867⟧ for ⟦2920⟧ and check its shape."
answer: "Step 1, ⟦2926⟧, ⟦2927⟧. Step 2, ⟦2928⟧ and ⟦2929⟧. Step 3, ⟦2930⟧, so ⟦2931⟧. Step 4, shape: ⟦2932⟧, same as ⟦1259⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Matrix calculus"]
tools: ["NumPy"]
topic: {"id": "09.05", "title": "Matrix calculus"}
---

## The big picture

Multi-output linear regression is the simplest model with a matrix of weights. Its gradient shows the differential method end to end, and gradient descent on it shows why feature scaling matters.

## The derivation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>L</mi><mo>=</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><mi mathvariant="normal">∥</mi><mi>X</mi><mi>W</mi><mo>−</mo><mi>Y</mi><msubsup><mi mathvariant="normal">∥</mi><mi>F</mi><mn>2</mn></msubsup><mo>=</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>R</mi><mi mathvariant="normal">⊤</mi></msup><mi>R</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><mi>R</mi><mo>=</mo><mi>X</mi><mi>W</mi><mo>−</mo><mi>Y</mi></mrow></math></span></span>
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>d</mi><mi>L</mi><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>R</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mtext> </mtext><mi>d</mi><mi>W</mi><mo stretchy="false">)</mo><mspace width="1em"/><mo>⟹</mo><mspace width="1em"/><msub><mi mathvariant="normal">∇</mi><mi>W</mi></msub><mi>L</mi><mo>=</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>X</mi><mi>W</mi><mo>−</mo><mi>Y</mi><mo stretchy="false">)</mo></mrow></math></span></span>

```
rng = np.random.default_rng(3)
X = rng.normal(size=(4, 3)); W = rng.normal(size=(3, 2)); T = rng.normal(size=(4, 2))
L = lambda W: 0.5 * ((X @ W - T) ** 2).sum()
analytic = X.T @ (X @ W - T)
h = 1e-6; num = np.zeros_like(W)
for i in range(3):
    for j in range(2):
        E = np.zeros_like(W); E[i, j] = h
        num[i, j] = (L(W + E) - L(W - E)) / (2 * h)
print(analytic.shape == W.shape, np.allclose(analytic, num))  # True True
```

## Gradient descent on it

~~~dg-raw
%%viz gdls live | Gradient descent on a least-squares loss over (intercept, slope). Rescale or centre the feature and watch the path change.
~~~

The Hessian is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span>. Unscaled or uncentred features make it badly conditioned, and gradient descent zig-zags (lesson 07.06). Standardising features rounds the contours.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi mathvariant="normal">∇</mi><mi>W</mi></msub><mfrac><mn>1</mn><mn>2</mn></mfrac><mi mathvariant="normal">∥</mi><mi>X</mi><mi>W</mi><mo>−</mo><mi>Y</mi><msubsup><mi mathvariant="normal">∥</mi><mi>F</mi><mn>2</mn></msubsup><mo>=</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>X</mi><mi>W</mi><mo>−</mo><mi>Y</mi><mo stretchy="false">)</mo></mrow></math></span>; its Hessian <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> is why feature scaling speeds up training.

</div>

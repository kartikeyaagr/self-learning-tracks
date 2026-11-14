---
id: "09.07"
course: "ml-maths-course"
module: "09"
slug: "domains-and-where-losses-break"
title: "Domains and Where Losses Break"
keyIdeas: ["The domain of a function is where it is defined: ⟦303⟧ needs ⟦2511⟧, ⟦2512⟧ needs ⟦2513⟧, ⟦2514⟧ needs ⟦2515⟧.", "Leaving the domain produces `-inf` or `nan`, which then spreads through every later computation and gradient.", "Libraries stay inside the domain by clamping or by working in log space (logits and `log_softmax`)."]
question: "A model outputs probability ⟦83⟧ for the true class. What is the cross-entropy ⟦287⟧, and how do frameworks avoid it?"
answer: "Step 1, ⟦2517⟧: the loss is infinite and its gradient is undefined. Step 2, one fix is clamping: ⟦2518⟧. Step 3, the standard fix is to compute from logits with `log_softmax`, which never forms a probability that could round to zero."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

Most NaNs in training come from a function evaluated outside its domain. Knowing the domains of the common pieces tells you where to look.

## Restricted domains in ML

| Expression | Needs | Appears in |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>p</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> | cross-entropy, log-likelihood |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi>v</mi></msqrt></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi><mo>≥</mo><mn>0</mn></mrow></math></span> | standard deviations, norms |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mi>s</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span> | normalisation layers |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo><mi mathvariant="normal">Σ</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> positive definite | Gaussian likelihoods |

```
p = np.array([0.5, 1e-300, 0.0])
with np.errstate(divide='ignore'):
    print(np.log(p))  # [  -0.69314718 -690.7755279           -inf]
eps = 1e-12
print(np.round(np.log(np.maximum(p, eps)), 2))  # [ -0.69 -27.63 -27.63]
```

## Working in log space

Softmax probabilities can round to exactly 0 in float32. `log_softmax` computes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>z</mi><mi>i</mi></msub><mo>−</mo><mi>log</mi><mo>⁡</mo><msub><mo>∑</mo><mi>k</mi></msub><msup><mi>e</mi><msub><mi>z</mi><mi>k</mi></msub></msup></mrow></math></span> directly, subtracting the largest logit first so nothing overflows:

```
z = np.array([1000.0, 0.0])
lse = z.max() + np.log(np.exp(z - z.max()).sum())
print(np.round(z - lse, 4))  # [    0. -1000.]
```

<div class="take">
<div class="lab">Key takeaway</div>

NaNs usually mean a function left its domain; clamp, add an epsilon, or stay in log space.

</div>

## Where this breaks in practice

- A single `inf` in a batch turns into `nan` after the next multiplication by zero, and then every weight becomes `nan` after one update. Check for non-finite losses before the optimiser step.

---
id: "09.34"
course: "ml-maths-course"
module: "09"
slug: "layout-conventions-and-shape-checks"
title: "Layout Conventions and Shape Checks"
keyIdeas: ["Use the convention code uses: for scalar ⟦491⟧ and matrix ⟦1259⟧, ⟦2867⟧ has the same shape as ⟦1259⟧, with entry ⟦2868⟧.", "Then every derivation has a free check: the answer's shape must match the variable's shape.", "Other texts use \"numerator layout\", which gives transposed answers; translate before comparing with code."]
question: "⟦491⟧ is a scalar, ⟦1259⟧ is ⟦2860⟧, ⟦513⟧ is ⟦2861⟧ and ⟦2862⟧ is ⟦2863⟧. Which of ⟦2864⟧, ⟦2865⟧, ⟦2866⟧ can be ⟦2867⟧?"
answer: "Step 1, ⟦2867⟧ must be ⟦2860⟧. Step 2, ⟦2864⟧ needs ⟦2878⟧: generally invalid. Step 3, ⟦2865⟧ is ⟦2879⟧: valid. Step 4, ⟦2866⟧ is ⟦908⟧: wrong shape."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Matrix calculus"]
tools: ["NumPy", "PyTorch"]
topic: {"id": "09.05", "title": "Matrix calculus"}
---

## The big picture

Matrix calculus looks hard mostly because books lay results out differently. Fix one convention, the one PyTorch and JAX use, and shapes become a powerful error check.

## The same-shape convention

| Variable | Its gradient |
| --- | --- |
| scalar <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> | scalar |
| vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> | vector in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> |
| matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mrow><mi>d</mi><mo>×</mo><mi>k</mi></mrow></msup></mrow></math></span> | matrix in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mrow><mi>d</mi><mo>×</mo><mi>k</mi></mrow></msup></mrow></math></span> |

```
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3)); W = rng.normal(size=(3, 5)); G = rng.normal(size=(4, 5))
print((X.T @ G).shape == W.shape, (G @ W.T).shape == X.shape)  # True True
```

## Shapes find the answer

Often only one product of the available matrices has the right shape. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mi>X</mi><mi>W</mi></mrow></math></span> with upstream <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>G</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi mathvariant="normal">∇</mi><mi>W</mi></msub><mi>L</mi></mrow></math></span> must be <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>×</mo><mi>k</mi></mrow></math></span> and the only candidate is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>G</mi></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi mathvariant="normal">∇</mi><mi>X</mi></msub><mi>L</mi></mrow></math></span> must be <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>d</mi></mrow></math></span> and the only candidate is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>G</mi><msup><mi>W</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>. Lesson 09.38 derives both properly.

<div class="take">
<div class="lab">Key takeaway</div>

give every gradient the shape of its variable, and check shapes at every step.

</div>

## Where this breaks in practice

- Broadcasting hides sums. Any dimension copied in the forward pass must be summed over in the backward pass.

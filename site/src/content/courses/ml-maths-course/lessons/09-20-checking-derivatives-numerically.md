---
id: "09.20"
course: "ml-maths-course"
module: "09"
slug: "checking-derivatives-numerically"
title: "Checking Derivatives Numerically"
keyIdeas: ["Forward differences ⟦2684⟧ have error proportional to ⟦547⟧; central differences ⟦2685⟧ to ⟦2686⟧.", "Too small an ⟦547⟧ lets rounding dominate: error about ⟦2687⟧.", "Gradient checks use central differences in float64, away from kinks."]
question: "A forward-difference check with ⟦2683⟧ in float64 shows relative errors around ⟦1368⟧. Is backprop wrong?"
answer: "Step 1, forward differences have leftover error ⟦2695⟧, about ⟦1368⟧: exactly this size for a correct gradient. Step 2, switch to central differences: error about ⟦2696⟧ plus rounding ⟦2697⟧. Step 3, a correct gradient then shows errors near ⟦2694⟧ or less; ⟦949⟧ would mean a real bug."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy", "PyTorch"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

Hand-derived gradients are easy to get wrong. You can always estimate a derivative by nudging the input, and comparing the two catches most bugs. The trick is choosing the nudge.

## Two formulas

| Formula | Error | Good <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span> in float64 |
| --- | --- | --- |
| forward <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>h</mi><mo stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mi>h</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∝</mo><mi>h</mi></mrow></math></span> | about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>8</mn></mrow></msup></mrow></math></span> |
| central <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>h</mi><mo stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>h</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn><mi>h</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∝</mo><msup><mi>h</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow></math></span> to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow></math></span> |

## The trade-off

Large <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>: the formula's own error dominates. Small <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>h</mi><mo stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> subtracts nearly equal numbers and rounding error (about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>16</mn></mrow></msup><mi>f</mi></mrow></math></span>) is divided by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>.

```
f = np.exp; x = 1.0; true = np.exp(1.0)
for h in (1e-2, 1e-5, 1e-8, 1e-12):
    fwd = (f(x + h) - f(x)) / h
    cen = (f(x + h) - f(x - h)) / (2 * h)
    print(h, f"{abs(fwd - true) / true:.0e}", f"{abs(cen - true) / true:.0e}")
```

~~~dg-raw
%%viz f_bars v=2,5,9,4 l=h=10⁻²,h=10⁻⁵,h=10⁻⁸,h=10⁻¹² hi=none yl=correct_digits,_forward_difference | Correct digits of a forward difference for the derivative of eˣ at 1. Accuracy improves as h shrinks until about 10⁻⁸, then rounding takes over and it gets worse.
~~~

`torch.autograd.gradcheck` uses central differences in float64.

<div class="take">
<div class="lab">Key takeaway</div>

check gradients with central differences, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>≈</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow></math></span>, in float64; relative errors near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup></mrow></math></span> mean the gradient is right.

</div>

## Where this breaks in practice

- Checks through dropout or other randomness fail unless the seed is fixed for every evaluation.
- In float32, small gradients lose precision or underflow; mixed-precision training scales the loss up for this reason.

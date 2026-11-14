---
id: "09.11"
course: "ml-maths-course"
module: "09"
slug: "continuity-in-several-variables"
title: "Continuity in Several Variables"
keyIdeas: ["⟦90⟧ is continuous at ⟦364⟧ if ⟦2556⟧.", "Sums, products, quotients (where the denominator is non-zero) and compositions of continuous functions are continuous.", "Most ML building blocks are continuous; ReLU is continuous but not differentiable at 0, and step functions are not continuous."]
question: "Is ⟦2553⟧ with ⟦2554⟧ continuous at the origin? Is ⟦2555⟧ continuous in ⟦378⟧?"
answer: "Step 1, the limit at the origin does not exist (lesson 09.10), so ⟦90⟧ is not continuous there. Step 2, ⟦2565⟧ is linear and ReLU is continuous, and a composition of continuous functions is continuous. So yes."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

Continuity means no jumps. In several variables it is defined exactly as in one, and the same building-block rules let you check it without epsilons.

## Definition and rules

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mtext> continuous at </mtext><mi>a</mi><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext><munder><mrow><mi>lim</mi><mo>⁡</mo></mrow><mrow><mi>x</mi><mo>→</mo><mi>a</mi></mrow></munder><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>f</mi><mo stretchy="false">(</mo><mi>a</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Continuous building blocks: polynomials, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>e</mi><mi>x</mi></msup></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>x</mi></mrow></math></span> (for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sin</mi><mo>⁡</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>max</mi><mo>⁡</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>x</mi><mi mathvariant="normal">∣</mi></mrow></math></span>, ReLU, sigmoid, softmax. Combine them by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>+</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>×</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>÷</mo></mrow></math></span> (non-zero denominators) and composition and the result is continuous.

## Why ML cares

| Function | Continuous? | Differentiable? |
| --- | --- | --- |
| sigmoid, tanh, softmax | yes | yes |
| ReLU, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">∣</mo><mi>x</mi><mo stretchy="false">∣</mo></mrow></math></span>, max-pooling | yes | not at the kinks |
| step function, argmax, accuracy | no | no |

~~~dg-raw
%%viz f_curve f=Math.max(0,x) g=(x>0?1:0) x=-2,2 y=-0.5,2 tk=1 fl=ReLU:_continuous gl=step:_jumps_at_0 | ReLU (green) has a corner at 0 but no jump. The step function (orange) jumps from 0 to 1, so it is not continuous and has no useful derivative.
~~~

Accuracy is a step function of the weights: tiny weight changes either do nothing or flip a prediction. That is why we train on a continuous surrogate like cross-entropy instead.

```
acc = lambda w: float((np.array([1.0, -1, 2]) * w > 0).mean())
print([acc(w) for w in (-0.01, 0.0, 0.01)])  # [0.3333333333333333, 0.0, 0.6666666666666666]
```

<div class="take">
<div class="lab">Key takeaway</div>

continuity means no jumps; build continuous functions from continuous pieces, and train on continuous losses rather than accuracy.

</div>

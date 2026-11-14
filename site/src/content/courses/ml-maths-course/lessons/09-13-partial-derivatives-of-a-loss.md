---
id: "09.13"
course: "ml-maths-course"
module: "09"
slug: "partial-derivatives-of-a-loss"
title: "Partial Derivatives of a Loss"
keyIdeas: ["For a model's loss, each partial derivative says how the loss responds to one weight.", "Their signs say which way to move each weight; their sizes say how sensitive the loss is.", "For ⟦2589⟧: ⟦2590⟧ and ⟦2591⟧."]
question: "With ⟦391⟧, ⟦2585⟧, ⟦2586⟧, ⟦730⟧, compute ⟦491⟧, ⟦2587⟧ and ⟦2588⟧. Which way should each move?"
answer: "Step 1, prediction 2, error ⟦23⟧, ⟦2597⟧. Step 2, ⟦2598⟧. Step 3, ⟦2599⟧. Step 4, both negative, so increasing either lowers the loss; ⟦378⟧ matters twice as much because its input is 2."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

This lesson applies partial derivatives to the smallest possible model. Every idea in backpropagation is already visible here.

## The model

Prediction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span>, loss <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>=</mo><mo stretchy="false">(</mo><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>−</mo><mi>y</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span>. The data <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> is fixed; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> vary.

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>L</mi></mrow><mrow><mi mathvariant="normal">∂</mi><mi>w</mi></mrow></mfrac><mo>=</mo><mn>2</mn><mo stretchy="false">(</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mo>⋅</mo><mi>x</mi><mo separator="true">,</mo><mspace width="2em"/><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>L</mi></mrow><mrow><mi mathvariant="normal">∂</mi><mi>b</mi></mrow></mfrac><mo>=</mo><mn>2</mn><mo stretchy="false">(</mo><mi>w</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Both have the form **error times something**. The error is shared; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>'s derivative is scaled by its input <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=(x*2-5)**2 x=0,4 y=0,26 tk=1 pts=1,w_=_1:_slope_−12 mark=min ml=w_=_2.5 | The loss as a function of w alone, with b = 0. At w = 1 the slope is −12, so increasing w lowers the loss; the minimum is at w = 2.5, where 2w = 5.
~~~

```
L = lambda w, b: (w * 2 + b - 5) ** 2
h = 1e-6
print(round((L(1 + h, 0) - L(1 - h, 0)) / (2 * h), 4))  # -12.0
print(round((L(1, h) - L(1, -h)) / (2 * h), 4))  # -6.0
```

## A step downhill

Move each weight against its partial derivative, scaled by a learning rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>=</mo><mn>0.05</mn></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>w</mi><mo>←</mo><mn>1</mn><mo>+</mo><mn>0.05</mn><mo>⋅</mo><mn>12</mn><mo>=</mo><mn>1.6</mn><mo separator="true">,</mo><mspace width="2em"/><mi>b</mi><mo>←</mo><mn>0</mn><mo>+</mo><mn>0.05</mn><mo>⋅</mo><mn>6</mn><mo>=</mo><mn>0.3</mn></mrow></math></span></span>
New prediction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.6</mn><mo>⋅</mo><mn>2</mn><mo>+</mo><mn>0.3</mn><mo>=</mo><mn>3.5</mn></mrow></math></span>, new loss <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2.25</mn></mrow></math></span>: down from 9.

<div class="take">
<div class="lab">Key takeaway</div>

each partial derivative of a loss is error times the input feeding that weight; moving against it lowers the loss.

</div>

---
id: "09.01"
course: "ml-maths-course"
module: "09"
slug: "functions-of-several-variables"
title: "Functions of Several Variables"
keyIdeas: ["A function ⟦2447⟧ takes a vector of ⟦233⟧ inputs and returns one number.", "A loss ⟦2448⟧ is such a function, with ⟦233⟧ equal to the number of weights.", "With two inputs you can draw it as a surface ⟦2449⟧; with more you cannot, so you study slices and level sets."]
question: "A model ⟦2443⟧ has squared loss ⟦2444⟧ on one example ⟦1868⟧, ⟦2445⟧. Write ⟦491⟧ as a function of ⟦2446⟧ and evaluate it at ⟦111⟧ and ⟦310⟧."
answer: "Step 1, ⟦2457⟧, so ⟦2458⟧. Step 2, at ⟦111⟧: ⟦2459⟧. Step 3, at ⟦310⟧: ⟦2460⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

Single-variable calculus studies <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>. Machine learning needs functions of thousands or millions of inputs: a loss takes every weight and returns one number. This module extends slopes, tangent lines and areas to that setting. This lesson fixes the notation.

## Notation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo>:</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>→</mo><mi mathvariant="normal">ℝ</mi><mo separator="true">,</mo><mspace width="2em"/><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>x</mi><mi>n</mi></msub><mo stretchy="false">)</mo><mo>↦</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Read: "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> takes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> real numbers and returns one". Examples:

| Function | Inputs | Output |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></math></span> | a point in the plane | squared distance from the origin |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><mi mathvariant="normal">∥</mi><mi>X</mi><mi>w</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> weights | mean squared error |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> for an image | 784 pixel values | log-likelihood |

## Losses as functions of the weights

In training, the **data is fixed** and the **weights vary**. So for a loss, the inputs of the function are the weights, even though the formula is written in terms of data.

```
L = lambda w1, w2: (w1 * 1 + w2 * 2 - 3) ** 2
print(L(1, 1), L(1, 0), L(3, 0))  # 0 4 0
```

Several weight settings can give zero loss: here every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msub><mi>w</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>w</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span> on the line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>1</mn></msub><mo>+</mo><mn>2</mn><msub><mi>w</mi><mn>2</mn></msub><mo>=</mo><mn>3</mn></mrow></math></span>.

## Vector inputs, vector outputs

Some maps return vectors too: a layer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>=</mo><mi>ϕ</mi><mo stretchy="false">(</mo><mi>W</mi><mi>x</mi><mo>+</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>→</mo><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span>. Those come in lesson 09.25. Until then, outputs are single numbers.

<div class="take">
<div class="lab">Key takeaway</div>

a function of several variables takes a vector and returns a number; a loss is one, with the weights as its inputs.

</div>

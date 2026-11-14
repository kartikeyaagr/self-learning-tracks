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

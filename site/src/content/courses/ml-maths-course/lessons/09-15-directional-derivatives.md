---
id: "09.15"
course: "ml-maths-course"
module: "09"
slug: "directional-derivatives"
title: "Directional Derivatives"
keyIdeas: ["The directional derivative along a unit vector ⟦280⟧ is ⟦2614⟧.", "For differentiable ⟦90⟧, ⟦2615⟧: a weighted mix of the partials.", "Partial derivatives are the special cases ⟦2616⟧."]
question: "For ⟦2613⟧ at ⟦1049⟧, find the rate of change in the direction of ⟦758⟧."
answer: "Step 1, ⟦2620⟧, ⟦2621⟧. Step 2, unit direction ⟦2043⟧. Step 3, ⟦2622⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

Partial derivatives only look along the axes. Optimisers move in arbitrary directions, so we need the rate of change along any direction. For smooth functions it is a simple dot product.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>D</mi><mi>u</mi></msub><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mrow><mi>lim</mi><mo>⁡</mo></mrow><mrow><mi>t</mi><mo>→</mo><mn>0</mn></mrow></munder><mfrac><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>t</mi><mi>u</mi><mo stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow><mi>t</mi></mfrac><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">∥</mi><mi>u</mi><mi mathvariant="normal">∥</mi><mo>=</mo><mn>1</mn></mrow></math></span></span>
It is the slope of the slice <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>↦</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>t</mi><mi>u</mi><mo stretchy="false">)</mo></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow></math></span>.

## The formula

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is differentiable,
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>D</mi><mi>u</mi></msub><mi>f</mi><mo>=</mo><msub><mi>f</mi><msub><mi>x</mi><mn>1</mn></msub></msub><msub><mi>u</mi><mn>1</mn></msub><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>f</mi><msub><mi>x</mi><mi>n</mi></msub></msub><msub><mi>u</mi><mi>n</mi></msub><mo>=</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo>⋅</mo><mi>u</mi></mrow></math></span></span>

```
f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
x = np.array([1.0, 2]); u = np.array([0.6, 0.8]); t = 1e-6
print(round((f(x + t * u) - f(x - t * u)) / (2 * t), 4), np.array([8.0, 3]) @ u)  # 7.2 7.2
```

~~~dg-raw
%%viz f_plane x=-1,4 y=-1,4 tk=1 v=0.6,0.8,u,a;1,0,e₁,m;0,1,e₂,m | Three unit directions from the same point. The slope along e₁ is f_x = 8, along e₂ is f_y = 3, and along u = (0.6, 0.8) it is 0.6 · 8 + 0.8 · 3 = 7.2.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

the slope in any direction is the dot product of the gradient with that unit direction.

</div>

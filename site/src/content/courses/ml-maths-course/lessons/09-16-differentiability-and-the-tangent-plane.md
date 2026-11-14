---
id: "09.16"
course: "ml-maths-course"
module: "09"
slug: "differentiability-and-the-tangent-plane"
title: "Differentiability and the Tangent Plane"
keyIdeas: ["⟦90⟧ is differentiable at ⟦7⟧ if ⟦2623⟧: near ⟦7⟧ it looks like a plane.", "Having all partial derivatives is weaker; partials only see the axes.", "If the partials exist and are continuous near ⟦7⟧, then ⟦90⟧ is differentiable there."]
question: "⟦2553⟧ with ⟦2554⟧. Do its partials exist at the origin? Is it differentiable there?"
answer: "Step 1, on both axes ⟦2499⟧, so ⟦2635⟧. Step 2, but on the diagonal ⟦2636⟧, so ⟦90⟧ is not even continuous at the origin. Step 3, a function that is not continuous cannot be differentiable. Partials exist; differentiability fails."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

"Differentiable" in several variables means more than "has partial derivatives". It means the function looks flat (like a tangent plane) when you zoom in, from every direction at once. That is the property gradient methods rely on.

## Definition

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is differentiable at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> if there is a vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> with
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>d</mi><mo stretchy="false">)</mo><mo>=</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>+</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mo>+</mo><mi>o</mi><mo stretchy="false">(</mo><mi mathvariant="normal">∥</mi><mi>d</mi><mi mathvariant="normal">∥</mi><mo stretchy="false">)</mo></mrow></math></span></span>
<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>o</mi><mo stretchy="false">(</mo><mi mathvariant="normal">∥</mi><mi>d</mi><mi mathvariant="normal">∥</mi><mo stretchy="false">)</mo></mrow></math></span> means an error that shrinks faster than the step itself.

## The tangent plane

For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span>, the tangent plane is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>z</mi><mo>=</mo><mi>f</mi><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo><mo>+</mo><msub><mi>f</mi><mi>x</mi></msub><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>a</mi><mo stretchy="false">)</mo><mo>+</mo><msub><mi>f</mi><mi>y</mi></msub><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mi>y</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**a tangent plane**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>=</mo><mn>5</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>x</mi></msub><mo>=</mo><mn>2</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>f</mi><mi>y</mi></msub><mo>=</mo><mn>4</mn></mrow></math></span>.

Tangent plane: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><mn>5</mn><mo>+</mo><mn>2</mn><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>+</mo><mn>4</mn><mo stretchy="false">(</mo><mi>y</mi><mo>−</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1.1</mn><mo separator="true">,</mo><mn>2.1</mn><mo stretchy="false">)</mo></mrow></math></span>: plane gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>5</mn><mo>+</mo><mn>0.2</mn><mo>+</mo><mn>0.4</mn><mo>=</mo><mn>5.6</mn></mrow></math></span>; true value <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.21</mn><mo>+</mo><mn>4.41</mn><mo>=</mo><mn>5.62</mn></mrow></math></span>.

</div>

```
f = lambda x, y: x ** 2 + y ** 2
plane = lambda x, y: 5 + 2 * (x - 1) + 4 * (y - 2)
for d in (0.1, 0.01, 0.001):
    print(d, f"{abs(f(1 + d, 2 + d) - plane(1 + d, 2 + d)):.1e}")
```

The error shrinks like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>d</mi><mn>2</mn></msup></mrow></math></span>: ten times smaller step, a hundred times smaller error.

## Kinks in ML

ReLU at 0, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>x</mi><mi mathvariant="normal">∣</mi></mrow></math></span> at 0, and max at ties are not differentiable. Frameworks return a **subgradient** there (ReLU uses slope 0). With real-valued inputs you almost never land exactly on a kink.

<div class="take">
<div class="lab">Key takeaway</div>

differentiable means "locally a plane", with error smaller than the step; partials alone do not guarantee it.

</div>

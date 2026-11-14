---
id: "09.24"
course: "ml-maths-course"
module: "09"
slug: "gradient-fields"
title: "Gradient Fields"
keyIdeas: ["A vector field attaches an arrow to every point; ⟦2698⟧ and ⟦2713⟧ are vector fields.", "Following ⟦2713⟧ from any start leads downhill toward a critical point.", "Diffusion models learn the score ⟦2732⟧, a field pointing toward more likely data."]
question: "For ⟦2451⟧, what is ⟦2713⟧ at ⟦310⟧, ⟦2730⟧ and ⟦2731⟧? Where do all arrows point?"
answer: "Step 1, ⟦2736⟧. Step 2, ⟦2737⟧, ⟦2738⟧, ⟦733⟧. Step 3, every arrow points at the origin, the minimum, and longer arrows sit further away."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
---

## The big picture

Instead of one gradient at one point, picture the gradient everywhere at once: a field of arrows. Gradient descent follows the arrows of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span>, like water running downhill.

## Vector fields

A **vector field** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo>:</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>→</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> attaches an arrow <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> to each point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, like a weather map of wind.

~~~dg-raw
%%viz f_plane x=-2.5,2.5 y=-2.5,2.5 tk=1 va=1,0,0.6,0,,f;0,2,0,1.2,,f;-1,-1,-0.6,-0.6,,f;2,1,1.2,0.6,,f;-2,1,-1.2,0.6,,f;1,-2,0.6,-1.2,,f;-1.5,-0.5,-0.9,-0.3,,f;0.5,1.5,0.3,0.9,,f | The field −∇f for the bowl f = x² + y², drawn at a few points (arrows scaled to 20%). Every arrow points to the minimum at the origin, and the arrows shrink as they approach it.
~~~

```
grad = lambda p: 2 * p
x = np.array([2.0, 1.0])
for _ in range(5):
    x = x - 0.25 * grad(x)
print(np.round(x, 4))  # [0.0625 0.0312]
```

Each step halves the distance to the origin.

~~~dg-flow Gradient descent as following a vector field.
x_0 -(minus η grad f)-> x_1 -(minus η grad f)-> x_2 -(...)-> *near a critical point where grad f = 0
~~~

## Score fields

Diffusion and score-based generative models learn <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi mathvariant="normal">∇</mi><mi>x</mi></msub><mi>log</mi><mo>⁡</mo><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, the gradient of the log-density of images. Following it moves noise toward realistic images.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> is a field of downhill arrows; gradient descent follows it in small steps.

</div>

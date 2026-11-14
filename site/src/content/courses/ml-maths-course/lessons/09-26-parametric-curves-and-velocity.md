---
id: "09.26"
course: "ml-maths-course"
module: "09"
slug: "parametric-curves-and-velocity"
title: "Parametric Curves and Velocity"
keyIdeas: ["A curve ⟦2754⟧ maps a time ⟦166⟧ to a point; its velocity is ⟦2755⟧.", "The velocity is tangent to the curve; its length is the speed.", "Training trajectories are curves ⟦2756⟧ in weight space, with velocity ⟦2757⟧."]
question: "For ⟦2753⟧, find the velocity at ⟦777⟧ and the speed."
answer: "Step 1, ⟦2762⟧. Step 2, at ⟦777⟧: ⟦5⟧. Step 3, speed ⟦2763⟧ for every ⟦166⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
---

## The big picture

A moving point traces a curve. Its derivative is a velocity vector that points along the curve. The weights during training move along such a curve, and the chain rule along it explains why the loss decreases.

## Velocity

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>r</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><msup><mi>r</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><msup><mi>y</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-1.6,1.6 y=-1.6,1.6 tk=1 pg=1.00:0.00_0.99:0.16_0.95:0.31_0.89:0.45_0.81:0.59_0.71:0.71_0.59:0.81_0.45:0.89_0.31:0.95_0.16:0.99_0.00:1.00_-0.16:0.99_-0.31:0.95_-0.45:0.89_-0.59:0.81_-0.71:0.71_-0.81:0.59_-0.89:0.45_-0.95:0.31_-0.99:0.16_-1.00:0.00_-0.99:-0.16_-0.95:-0.31_-0.89:-0.45_-0.81:-0.59_-0.71:-0.71_-0.59:-0.81_-0.45:-0.89_-0.31:-0.95_-0.16:-0.99_0.00:-1.00_0.16:-0.99_0.31:-0.95_0.45:-0.89_0.59:-0.81_0.71:-0.71_0.81:-0.59_0.89:-0.45_0.95:-0.31_0.99:-0.16,m v=0.71,0.71,,k va=0.71,0.71,0,1.41,r′(π/4),f | The unit circle r(t) = (cos t, sin t). The black arrow is the position r(π/4); the orange arrow is the velocity (−0.71, 0.71). It is tangent to the circle, at right angles to the position.
~~~

```
r = lambda t: np.array([np.cos(t), np.sin(t)])
t, h = np.pi / 4, 1e-6
v = (r(t + h) - r(t - h)) / (2 * h)
print(np.round(v, 4), round(v @ r(t), 8))  # [-0.7071  0.7071] 0.0
```

## The loss along a training path

Along any curve, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>d</mi><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mi>f</mi><mo stretchy="false">(</mo><mi>r</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo>⋅</mo><msup><mi>r</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></math></span> (lesson 09.18). For the gradient-flow curve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>r</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mi>d</mi><mrow><mi>d</mi><mi>t</mi></mrow></mfrac><mi>f</mi><mo>=</mo><mo>−</mo><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>≤</mo><mn>0</mn></mrow></math></span></span>
The loss can only go down along it.

<div class="take">
<div class="lab">Key takeaway</div>

the derivative of a curve is its velocity, tangent to the curve; along gradient flow the loss falls at rate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi mathvariant="normal">∇</mi><mi>f</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>.

</div>

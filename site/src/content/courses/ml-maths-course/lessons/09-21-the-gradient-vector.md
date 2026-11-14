---
id: "09.21"
course: "ml-maths-course"
module: "09"
slug: "the-gradient-vector"
title: "The Gradient Vector"
keyIdeas: ["The gradient collects all partial derivatives: ⟦2699⟧.", "It lives in the input space: one entry per input variable.", "Directional derivatives come from it: ⟦2615⟧."]
question: "For ⟦2613⟧, write ⟦2698⟧ and evaluate it at ⟦1049⟧."
answer: "Step 1, ⟦2707⟧ and ⟦2708⟧. Step 2, ⟦2709⟧. Step 3, at ⟦1049⟧: ⟦2710⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
---

## The big picture

Put the partial derivatives in one vector and you get the **gradient**, the most important object in training. The next lessons explain what it points at.

## Definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true">(</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mn>1</mn></msub></mrow></mfrac><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>n</mi></msub></mrow></mfrac><mo fence="true">)</mo></mrow></mrow></math></span></span>
The symbol <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi></mrow></math></span> is "nabla" or "del". For a loss with a million weights, the gradient has a million entries.

```
f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
x = np.array([1.0, 2]); h = 1e-6
g = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(2)])
print(np.round(g, 4))  # [8. 3.]
```

~~~dg-raw
%%viz f_plane x=-1,10 y=-1,5 tk=1 v=8,3,∇f(1~_2)_=_(8~_3),f;8,0,∂f/∂x_=_8,m;0,3,∂f/∂y_=_3,m | The gradient at (1, 2) is the arrow built from the two partial derivatives: 8 along x and 3 along y.
~~~

## Gradients of simple functions

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>x</mi></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi></mrow></math></span> (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> symmetric) | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>A</mi><mi>x</mi></mrow></math></span> |

These are the multivariable versions of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mi>x</mi><mo>→</mo><mi>a</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><msup><mi>x</mi><mn>2</mn></msup><mo>→</mo><mn>2</mn><mi>a</mi><mi>x</mi></mrow></math></span>; lesson 09.35 derives them.

<div class="take">
<div class="lab">Key takeaway</div>

the gradient is the vector of all partial derivatives, one entry per input.

</div>

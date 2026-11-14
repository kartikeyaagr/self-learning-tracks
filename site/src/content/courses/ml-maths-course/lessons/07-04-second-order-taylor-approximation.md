---
id: "07.04"
course: "ml-maths-course"
module: "07"
slug: "second-order-taylor-approximation"
title: "Second-Order Taylor Approximation"
keyIdeas: ["Near ⟦7⟧, a smooth function satisfies ⟦2172⟧, with ⟦1785⟧ the Hessian.", "It is the multivariable version of ⟦2173⟧.", "So every smooth loss looks like a quadratic close to any point."]
question: "For ⟦2170⟧ at ⟦685⟧, write the second-order approximation and evaluate it at ⟦2171⟧. Compare with the true value."
answer: "Step 1, ⟦2181⟧, ⟦2182⟧, ⟦2183⟧. Step 2, the approximation is ⟦2184⟧. Step 3, at ⟦2185⟧: ⟦2186⟧. Step 4, true: ⟦2187⟧. Very close."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Curvature and quadratic objectives"]
tools: ["NumPy"]
topic: {"id": "07.01", "title": "Curvature and quadratic objectives"}
---

## The big picture

The quadratic forms of the previous lessons are not a special case. Zoom in close enough on any smooth loss and it looks quadratic. That is why the eigenvalues of the Hessian govern training near a minimum.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>d</mi><mo stretchy="false">)</mo><mo>≈</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>+</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mo>+</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><msup><mi>d</mi><mi mathvariant="normal">⊤</mi></msup><mi>H</mi><mi>d</mi></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, the **gradient**, is the vector of first partial derivatives (module 09).
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi><mo>=</mo><msup><mi mathvariant="normal">∇</mi><mn>2</mn></msup><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, the **Hessian**, is the symmetric matrix of second partial derivatives, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>H</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><msup><mi mathvariant="normal">∂</mi><mn>2</mn></msup><mi>f</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>i</mi></msub><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>j</mi></msub></mrow></math></span>.
The error shrinks like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>d</mi><msup><mi mathvariant="normal">∥</mi><mn>3</mn></msup></mrow></math></span> as the step gets small.

```
f = lambda v: np.exp(v[0]) + v[1] ** 2
d = np.array([0.1, 0.2])
approx = 1 + d[0] + 0.5 * (d[0] ** 2 + 2 * d[1] ** 2)
print(round(approx, 5), round(f(d), 5))  # 1.145 1.14517
```

~~~dg-raw
%%viz f_curve f=Math.exp(x) g=1+x+x*x/2 x=-2,2 y=0,7.5 tk=1 fl=eˣ gl=1_+_x_+_x²/2 | In one variable: eˣ (green) and its second-order approximation at 0 (orange). They agree closely near 0 and drift apart further out.
~~~

## Quadratic objectives

The standard quadratic objective is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mn>2</mn></mfrac></mstyle><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>−</mo><msup><mi>b</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">∇</mi><mi>f</mi><mo>=</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo separator="true">,</mo><mspace width="2em"/><msup><mi mathvariant="normal">∇</mi><mn>2</mn></msup><mi>f</mi><mo>=</mo><mi>A</mi></mrow></math></span></span>
For positive definite <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> its minimum is where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><msup><mi>x</mi><mo>⋆</mo></msup><mo>=</mo><mi>b</mi></mrow></math></span>. So **minimising a convex quadratic and solving a linear system are the same problem**; the conjugate gradient method uses this.

<div class="take">
<div class="lab">Key takeaway</div>

near any point a smooth loss is approximately a quadratic, with the gradient as its slope and the Hessian as its curvature.

</div>

---
id: "09.51"
course: "ml-maths-course"
module: "09"
slug: "joint-densities-and-the-gaussian-integral"
title: "Joint Densities and the Gaussian Integral"
keyIdeas: ["A joint density ⟦3112⟧ gives probabilities as volumes: ⟦3113⟧.", "A marginal integrates out the other variable: ⟦3114⟧.", "Squaring the Gaussian integral and switching to polar gives ⟦3017⟧."]
question: "⟦513⟧ and ⟦512⟧ are independent standard normals. Compute ⟦3111⟧ exactly."
answer: "Step 1, the joint density is ⟦3126⟧. Step 2, in polar: ⟦3127⟧. Step 3, that is ⟦3128⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Integration over two variables"]
tools: ["NumPy", "SciPy"]
topic: {"id": "09.07", "title": "Integration over two variables"}
---

## The big picture

This lesson puts the module's integration tools to work on probability, and proves the constant in the normal density.

## Four operations

| Operation | Integral |
| --- | --- |
| Probability of a region | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∬</mo><mi>R</mi></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span> |
| Marginal of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo>∫</mo><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span> |
| Expectation | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>g</mi><mo stretchy="false">(</mo><mi>X</mi><mo separator="true">,</mo><mi>Y</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo><mo>=</mo><mo>∬</mo><mi>g</mi><mtext> </mtext><mi>p</mi><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span> |
| Normalisation check | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∬</mo><mi>p</mi><mo>=</mo><mn>1</mn></mrow></math></span> |

<div class="ex">
<div class="lab">Worked example</div>

**a joint density on the unit square**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>x</mi><mo>+</mo><mi>y</mi></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mn>2</mn></msup></mrow></math></span>. It integrates to 1 (lesson 09.46 gave 3 over a rectangle twice as tall; over <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mn>2</mn></msup></mrow></math></span> the same steps give 1).

Marginal of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><mi>x</mi><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow></math></span>, which integrates to 1.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>+</mo><mi>Y</mi><mo>≤</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: the triangle below the anti-diagonal gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><msubsup><mo>∫</mo><mn>0</mn><mrow><mn>1</mn><mo>−</mo><mi>x</mi></mrow></msubsup><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mfrac><mn>1</mn><mn>3</mn></mfrac></mrow></math></span>.

</div>

## The Gaussian integral

Call <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>=</mo><msubsup><mo>∫</mo><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow><mi mathvariant="normal">∞</mi></msubsup><msup><mi>e</mi><mrow><mo>−</mo><msup><mi>x</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span>. There is no antiderivative, but squaring it gives a 2D integral that polar coordinates can do:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>I</mi><mn>2</mn></msup><mo>=</mo><mo>∬</mo><msup><mi>e</mi><mrow><mo>−</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>2</mn></msup><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup><mtext> </mtext><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mrow><mn>2</mn><mi>π</mi></mrow></msubsup><mtext> ⁣</mtext><msubsup><mo>∫</mo><mn>0</mn><mi mathvariant="normal">∞</mi></msubsup><msup><mi>e</mi><mrow><mo>−</mo><msup><mi>r</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup><mtext> </mtext><mi>r</mi><mtext> </mtext><mi>d</mi><mi>r</mi><mtext> </mtext><mi>d</mi><mi>θ</mi><mo>=</mo><mn>2</mn><mi>π</mi></mrow></math></span></span>

~~~dg-flow The Gaussian integral by squaring it.
I = integral of e^(-x²/2) dx -(square, use x and y)-> I² = double integral of e^(-(x²+y²)/2) -(polar, dA = r dr dtheta)-> 2 π times integral of r e^(-r²/2) dr -(= 2 π)-> *I = √(2 π)
~~~

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x/2) x=-3.5,3.5 y=0,1 tk=1 | The curve e^(−x²/2). The area under it is √(2π) ≈ 2.507, which is why the normal density divides by √(2π).
~~~

```
rng = np.random.default_rng(0)
xy = rng.normal(size=(1_000_000, 2))
print(round(((xy ** 2).sum(1) <= 1).mean(), 3), round(1 - np.exp(-0.5), 3))  # 0.393 0.393
```

<div class="take">
<div class="lab">Key takeaway</div>

probabilities, marginals and expectations of joint densities are all double integrals; polar coordinates crack the Gaussian.

</div>

## Where this breaks in practice

- In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions the unit ball fills a vanishing fraction of the cube around it (about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2.5</mn><mo>×</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>=</mo><mn>10</mn></mrow></math></span>). Rejection sampling from the box becomes hopeless.

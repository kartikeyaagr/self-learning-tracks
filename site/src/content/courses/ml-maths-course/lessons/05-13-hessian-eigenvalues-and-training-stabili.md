---
id: "05.13"
course: "ml-maths-course"
module: "05"
slug: "hessian-eigenvalues-and-training-stability"
title: "Hessian Eigenvalues and Training Stability"
keyIdeas: ["Near a minimum, gradient descent multiplies the error along eigen-direction ⟦274⟧ of the Hessian by ⟦1777⟧ each step.", "Stability needs ⟦1778⟧; the flattest direction then moves at rate ⟦1779⟧.", "The condition number ⟦1780⟧ controls how slow plain gradient descent is."]
question: "The Hessian at a point has eigenvalues ⟦1776⟧. Is it a minimum? What step size is safe, and why is training slow along one direction?"
answer: "Step 1, all eigenvalues positive, so (with zero gradient) a strict local minimum. Step 2, stability needs ⟦1786⟧ for all ⟦274⟧, so ⟦1796⟧. Step 3, along ⟦1797⟧ the factor is ⟦1798⟧: halving the error takes about ⟦1799⟧ steps. Step 4, the condition number 5,000 is the cause; momentum, Adam or preconditioning help."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

Everything in this module comes together here. Near a minimum, a loss looks like a quadratic bowl described by the Hessian. Its eigenvalues tell you the largest safe learning rate and how slowly the flattest direction will converge.

## Gradient descent on a bowl

Near a minimum <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mo>∗</mo></msup></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≈</mo><mi>f</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo>∗</mo></msup><mo stretchy="false">)</mo><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><msup><mi>x</mi><mo>∗</mo></msup><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mi>H</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><msup><mi>x</mi><mo>∗</mo></msup><mo stretchy="false">)</mo></mrow></math></span>. The gradient is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><msup><mi>x</mi><mo>∗</mo></msup><mo stretchy="false">)</mo></mrow></math></span>, so one step gives
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>x</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>−</mo><msup><mi>x</mi><mo>∗</mo></msup><mo>=</mo><mo stretchy="false">(</mo><mi>I</mi><mo>−</mo><mi>η</mi><mi>H</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo>−</mo><msup><mi>x</mi><mo>∗</mo></msup><mo stretchy="false">)</mo></mrow></math></span></span>
In the eigenbasis of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi></mrow></math></span>, each coordinate of the error is multiplied by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>η</mi><msub><mi>λ</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></math></span> per step.

~~~dg-raw
%%viz f_bars v=-1,0.92,0.9996 l=λ_=_50,λ_=_2,λ_=_0.01 hi=none yl=error_factor_per_step_(η_=_0.04) | With the largest safe-ish step η = 0.04: the steep direction flips sign each step (factor −1, on the edge of stability), and the flat direction barely moves (0.9996).
~~~

## The safe step size

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mn>1</mn><mo>−</mo><mi>η</mi><msub><mi>λ</mi><mi>i</mi></msub><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span> for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mi>η</mi><mo>&lt;</mo><mn>2</mn><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span>. Above that, the steepest direction oscillates with growing size and training diverges.

```
H = np.diag([50.0, 2, 0.01]); x = np.ones(3)
for eta in (0.03, 0.05):
    z = x.copy()
    for _ in range(100):
        z = z - eta * H @ z
    print(eta, np.round(np.abs(z), 3))
```

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>=</mo><mn>0.03</mn></mrow></math></span> the first two coordinates converge and the third barely moves; with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>=</mo><mn>0.05</mn></mrow></math></span> the first blows up.

## Slow directions

The flattest direction shrinks by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>−</mo><mi>η</mi><msub><mi>λ</mi><mi>min</mi><mo>⁡</mo></msub><mo>≤</mo><mn>1</mn><mo>−</mo><mn>2</mn><msub><mi>λ</mi><mi>min</mi><mo>⁡</mo></msub><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span> per step. The ratio <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi><mo>=</mo><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>min</mi><mo>⁡</mo></msub></mrow></math></span>, the condition number, sets the number of steps needed: roughly proportional to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>κ</mi></mrow></math></span>.

## In deep learning

- Hessians of deep nets have a few large eigenvalues, a bulk near zero, and negative ones away from minima.
- "Edge of stability" research finds the top eigenvalue tends to hover near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">/</mi><mi>η</mi></mrow></math></span> during training.
- The Hessian of a model with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>6</mn></msup></mrow></math></span> parameters has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>12</mn></msup></mrow></math></span> entries. The Lanczos method estimates its top eigenvalues from Hessian-vector products alone.

<div class="take">
<div class="lab">Key takeaway</div>

the largest Hessian eigenvalue caps the learning rate at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span>; the smallest then sets how slowly training finishes.

</div>

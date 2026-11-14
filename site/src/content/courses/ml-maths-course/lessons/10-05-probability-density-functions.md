---
id: "10.05"
course: "ml-maths-course"
module: "10"
slug: "probability-density-functions"
title: "Probability Density Functions"
keyIdeas: ["A density ⟦3272⟧ has total area 1, and ⟦3273⟧.", "A density can be larger than 1, because only areas must be at most 1.", "Log-likelihoods of continuous models can be positive for the same reason."]
question: "⟦513⟧ has density ⟦3270⟧ on ⟦4⟧. Find ⟦119⟧ and ⟦3271⟧."
answer: "Step 1, ⟦3280⟧, so ⟦3281⟧. Step 2, ⟦3282⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous random variables"]
tools: ["NumPy", "SciPy"]
topic: {"id": "10.02", "title": "Continuous random variables"}
---

## The big picture

The density is the object every continuous model writes down: a Gaussian likelihood, a VAE decoder, a diffusion model's noise. This lesson pins down what it means and what it doesn't.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≥</mo><mn>0</mn><mo separator="true">,</mo><mspace width="2em"/><msubsup><mo>∫</mo><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow><mi mathvariant="normal">∞</mi></msubsup><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mn>1</mn><mo separator="true">,</mo><mspace width="2em"/><mi>P</mi><mo stretchy="false">(</mo><mi>a</mi><mo>≤</mo><mi>X</mi><mo>≤</mo><mi>b</mi><mo stretchy="false">)</mo><mo>=</mo><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**the uniform distribution**

Uniform on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span> there, 0 elsewhere. Area: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>0.5</mn><mo>=</mo><mn>1</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mn>0.5</mn><mo>≤</mo><mi>X</mi><mo>≤</mo><mn>1.2</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.7</mn><mo>×</mo><mn>0.5</mn><mo>=</mo><mn>0.35</mn></mrow></math></span>.

</div>

**A density can be bigger than 1.** Uniform on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>0.1</mn><mo stretchy="false">]</mo></mrow></math></span> has density 10. The interval is short, so the area is still <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.1</mn><mo>×</mo><mn>10</mn><mo>=</mo><mn>1</mn></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=3*x*x x=0,1 y=0,3 tk=0.25 dom=0.5,1,cc | The density p(x) = 3x² on [0, 1]. It reaches 3, yet the total area is 1. The shaded area, P(X > 0.5), is 0.875.
~~~

```
from scipy import stats
print(stats.uniform(0, 0.1).pdf(0.05))  # 10.0
print(round(stats.norm(0, 0.01).logpdf(0), 3))  # 3.686
```

A Gaussian with standard deviation 0.01 has density about 40 at its centre, so its log-density is positive.

<div class="take">
<div class="lab">Key takeaway</div>

a density is probability per unit length; integrate it to get probabilities.

</div>

## Where this breaks in practice

- Comparing log-likelihoods of a continuous model and a discrete one is meaningless; one is a density, the other a probability.

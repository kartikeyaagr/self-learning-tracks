---
id: "10.25"
course: "ml-maths-course"
module: "10"
slug: "the-bernoulli-distribution"
title: "The Bernoulli Distribution"
keyIdeas: ["Bernoulli(⟦102⟧) is one yes-or-no trial: ⟦3472⟧, ⟦3473⟧.", "The PMF can be written ⟦3474⟧; mean ⟦102⟧, variance ⟦3475⟧.", "Binary cross-entropy is its negative log-likelihood."]
question: "An email is spam with probability 0.3. Give the mean and variance of the indicator ⟦513⟧."
answer: "Step 1, mean ⟦3482⟧. Step 2, variance ⟦3483⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["NumPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

Most discrete distributions are built from one coin flip. A click, a correct prediction, a spam label: each is a Bernoulli variable.

## The distribution

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>p</mi><mi>x</mi></msup><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><msup><mo stretchy="false">)</mo><mrow><mn>1</mn><mo>−</mo><mi>x</mi></mrow></msup><mo separator="true">,</mo><mspace width="1em"/><mi>x</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span></span>
For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow></math></span> this gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span>; for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span> it gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>−</mo><mi>p</mi></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=x*(1-x) x=0,1 y=0,0.3 tk=0.25 pts=0.5,max_0.25 | The variance p(1 − p) of a Bernoulli. It is largest at p = 0.5 and zero when the outcome is certain.
~~~

## Binary cross-entropy

A model predicts <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>p</mi><mo>^</mo></mover></mrow></math></span>; the label is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>. The negative log-likelihood is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo>−</mo><mi>log</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><msup><mover accent="true"><mi>p</mi><mo>^</mo></mover><mi>y</mi></msup><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><msup><mo stretchy="false">)</mo><mrow><mn>1</mn><mo>−</mo><mi>y</mi></mrow></msup><mo fence="true">)</mo></mrow><mo>=</mo><mo>−</mo><mi>y</mi><mi>log</mi><mo>⁡</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>−</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo stretchy="false">)</mo></mrow></math></span></span>
which is exactly binary cross-entropy.

```
p_hat, y = 0.8, 1
print(round(-(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)), 4))  # 0.2231
```

<div class="take">
<div class="lab">Key takeaway</div>

a Bernoulli is one biased coin; its log-likelihood is the binary cross-entropy loss.

</div>

## Where this breaks in practice

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mn>0</mn></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow></math></span>. Clip predictions or compute the loss from logits (`BCEWithLogitsLoss`).

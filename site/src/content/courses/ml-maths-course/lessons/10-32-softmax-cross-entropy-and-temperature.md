---
id: "10.32"
course: "ml-maths-course"
module: "10"
slug: "softmax-cross-entropy-and-temperature"
title: "Softmax, Cross-Entropy and Temperature"
keyIdeas: ["Cross-entropy ⟦3527⟧ is the negative log-likelihood of a categorical model.", "Training a classifier with cross-entropy is maximum likelihood (lesson 12.11).", "Temperature: ⟦3528⟧ sharpens for ⟦3529⟧ and flattens for ⟦3530⟧."]
question: "Logits ⟦3526⟧, true class 0. Compute the cross-entropy."
answer: "Step 1, ⟦3533⟧. Step 2, ⟦3534⟧. Step 3, loss ⟦3535⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["NumPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

This is the loss used to train nearly every classifier and language model. Its probability meaning explains its behaviour.

## Cross-entropy is a log-likelihood

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo>−</mo><mi>log</mi><mo>⁡</mo><mi>P</mi><mo stretchy="false">(</mo><mi>Y</mi><mo>=</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mi>log</mi><mo>⁡</mo><msub><mi>π</mi><mi>y</mi></msub><mo>=</mo><mo>−</mo><msub><mi>z</mi><mi>y</mi></msub><mo>+</mo><mi>log</mi><mo>⁡</mo><munder><mo>∑</mo><mi>k</mi></munder><msup><mi>e</mi><msub><mi>z</mi><mi>k</mi></msub></msup></mrow></math></span></span>

## Temperature

Sampling uses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">softmax</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>z</mi><mi mathvariant="normal">/</mi><mi>T</mi><mo stretchy="false">)</mo></mrow></math></span>.

~~~dg-raw
%%viz f_bars v=0.87,0.12,0.02,0.67,0.24,0.09,0.51,0.31,0.19 l=T=0.5,·,·,T=1,·,·,T=2,·,· hi=none yl=probability_of_each_token | Softmax of logits (2, 1, 0) at three temperatures. Low T piles probability on the top token; high T spreads it out.
~~~

```
z = np.array([2.0, 1, 0])
sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
print(round(-np.log(sm(z)[0]), 3))  # 0.408
print(np.round(sm(z / 0.5), 2), np.round(sm(z / 2), 2))  # [0.87 0.12 0.02] [0.51 0.31 0.19]
```

<div class="take">
<div class="lab">Key takeaway</div>

softmax plus cross-entropy is maximum likelihood for a categorical model; temperature reshapes it.

</div>

## Where this breaks in practice

- Compute softmax after subtracting the largest logit, or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>e</mi><mi>z</mi></msup></mrow></math></span> overflows.

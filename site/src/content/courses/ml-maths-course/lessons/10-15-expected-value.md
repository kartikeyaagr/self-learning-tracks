---
id: "10.15"
course: "ml-maths-course"
module: "10"
slug: "expected-value"
title: "Expected Value"
keyIdeas: ["⟦3396⟧ for discrete ⟦513⟧ and ⟦3397⟧ for continuous ⟦513⟧.", "The mean is the balance point of the distribution; it need not be a possible value.", "The sample average of many independent draws approaches ⟦3012⟧."]
question: "⟦3393⟧ with probability 0.1 and ⟦3394⟧ otherwise. Find ⟦3395⟧."
answer: "Step 1, ⟦3400⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

A distribution is a whole curve, but you usually want a few numbers that summarise it. The first is where it is centred. Training loss is a mean; so is accuracy.

## A weighted average

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>X</mi><mo stretchy="false">]</mo><mo>=</mo><munder><mo>∑</mo><mi>x</mi></munder><mi>x</mi><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo stretchy="false">)</mo><mspace width="1em"/><mtext>or</mtext><mspace width="1em"/><mo>∫</mo><mi>x</mi><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span></span>
Each value is weighted by how likely it is. Picture the distribution as weights on a ruler: the mean is where the ruler balances.

~~~dg-raw
%%viz f_bars v=1,1,1,1,1,1 l=1,2,3,4,5,6 hi=none yl=chance_in_sixths | A fair die puts equal weight on 1 to 6. It balances at 3.5, a value you can never roll.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a skewed distribution**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mn>10</mn></mrow></math></span> with probability 0.1, else 0. Mean <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mn>1</mn></mrow></math></span>. Nine times out of ten you get 0, yet the mean is 1: one large value pulls it.

</div>

```
rng = np.random.default_rng(0)
x = rng.integers(1, 7, 1_000_000)
print(round(x.mean(), 2), sum(range(1, 7)) / 6)  # 3.5 3.5
```

<div class="take">
<div class="lab">Key takeaway</div>

the mean is the probability-weighted average, the balance point of the distribution.

</div>

## Where this breaks in practice

- Some distributions have no mean. The Cauchy distribution's sample averages never settle down.

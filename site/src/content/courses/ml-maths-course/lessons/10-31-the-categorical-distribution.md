---
id: "10.31"
course: "ml-maths-course"
module: "10"
slug: "the-categorical-distribution"
title: "The Categorical Distribution"
keyIdeas: ["Categorical(⟦21⟧) is one draw from ⟦3520⟧ classes with probabilities ⟦3521⟧ that add up to 1.", "A softmax output is the parameter of a categorical distribution over labels.", "One-hot encoding writes a draw as a vector with a single 1."]
question: "A model outputs ⟦3519⟧. What is the probability it assigns to the true class 2 (the second one)?"
answer: "Step 1, it is ⟦3523⟧. Step 2, with one-hot ⟦3524⟧, this is ⟦3525⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["NumPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

The Bernoulli with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>K</mi></mrow></math></span> outcomes instead of 2. Every classifier and every language model predicts a categorical distribution.

## The PMF

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>Y</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>π</mi><mi>k</mi></msub><mo separator="true">,</mo><mspace width="2em"/><mtext>or with one-hot </mtext><mi>y</mi><mo>:</mo><mspace width="1em"/><mi>P</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mo>∏</mo><mi>k</mi></munder><msubsup><mi>π</mi><mi>k</mi><msub><mi>y</mi><mi>k</mi></msub></msubsup></mrow></math></span></span>

~~~dg-raw
%%viz f_bars v=0.7,0.2,0.1 l=cat,dog,bird hi=none yl=π_k | A categorical distribution over three labels. The bars add up to 1.
~~~

```
rng = np.random.default_rng(0)
draws = rng.choice(3, size=100000, p=[0.7, 0.2, 0.1])
print(np.round(np.bincount(draws) / draws.size, 2))  # [0.7 0.2 0.1]
```

<div class="take">
<div class="lab">Key takeaway</div>

a categorical is one draw from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>K</mi></mrow></math></span> labels; softmax outputs are its parameters.

</div>

## Where this breaks in practice

- Categorical labels have no order or numeric value. A mean of class ids is meaningless.

---
id: "10.02"
course: "ml-maths-course"
module: "10"
slug: "bayes-theorem"
title: "Bayes' Theorem"
keyIdeas: ["Bayes: ⟦3235⟧. The posterior is likelihood times prior, rescaled to add up to 1.", "The denominator comes from total probability: ⟦3236⟧.", "When the event is rare, the base rate dominates: a very accurate test can still be wrong most of the time."]
question: "A screening test has sensitivity 99% and specificity 95%. The condition affects 1% of the people tested. A person tests positive: what is the probability they have the condition?"
answer: "Step 1, ⟦3250⟧, ⟦3251⟧, ⟦3252⟧. Step 2, ⟦3253⟧. Step 3, ⟦3254⟧. Only about one positive in six is real."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Probability rules and Bayesian updating"]
tools: ["NumPy"]
topic: {"id": "10.01", "title": "Probability rules and Bayesian updating"}
---

## The big picture

Bayes' theorem flips a conditional probability around. You know how likely the evidence is under each explanation; you want how likely each explanation is given the evidence. It explains why a "99% accurate" test can mislead, and it is how a spam filter or a Bayesian model updates.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow></math></span>, the **prior**: how plausible the hypothesis was before the data.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo></mrow></math></span>, the **likelihood**: how well the hypothesis predicts the data.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo></mrow></math></span>, the **posterior**: how plausible it is after the data.

The denominator is built by total probability over all hypotheses:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow><mrow><munder><mo>∑</mo><msup><mi>H</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></munder><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><msup><mi>H</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><msup><mi>H</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span></span>

## Counting people

Out of 10,000 people, 100 are sick and 99 of them test positive. 9,900 are healthy and 495 of them test positive anyway. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>99</mn><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mn>99</mn><mo>+</mo><mn>495</mn><mo stretchy="false">)</mo><mo>≈</mo><mn>17</mn><mi mathvariant="normal">%</mi></mrow></math></span> of positives are real.

~~~dg-raw
%%viz f_bars v=99,495 l=sick_and_positive,healthy_and_positive hi=none yl=people_out_of_10~000 | Positive tests among 10,000 people. Healthy people far outnumber sick ones, so most positives come from the 5% false-positive rate.
~~~

```
sens, spec, prev = 0.99, 0.95, 0.01
p_pos = sens * prev + (1 - spec) * (1 - prev)
print(round(p_pos, 4), round(sens * prev / p_pos, 3))  # 0.0594 0.167
rng = np.random.default_rng(0)
sick = rng.random(1_000_000) < prev
pos = np.where(sick, rng.random(sick.size) < sens, rng.random(sick.size) < 1 - spec)
print(round(sick[pos].mean(), 2))  # 0.17
```

## Updating twice

Today's posterior is tomorrow's prior.

~~~dg-flow Bayesian updating.
Prior P(H) -(observe D_1, multiply by likelihood)-> posterior -(observe D_2)-> *updated posterior
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two positive tests**

Assume the two tests are independent given the true state. New prior: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.167</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mo>+</mo><mo stretchy="false">)</mo><mo>=</mo><mn>0.99</mn><mo>×</mo><mn>0.167</mn><mo>+</mo><mn>0.05</mn><mo>×</mo><mn>0.833</mn><mo>≈</mo><mn>0.207</mn></mrow></math></span>. Posterior: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.99</mn><mo>×</mo><mn>0.167</mn><mi mathvariant="normal">/</mi><mn>0.207</mn><mo>≈</mo><mn>0.80</mn></mrow></math></span>.

One positive: 17%. Two positives: 80%.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**naive Bayes for spam**

Prior <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>spam</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.4</mn></mrow></math></span>. "free" appears in 30% of spam and 2% of ham; "meeting" in 1% of spam and 10% of ham.

An email contains both words. Spam score: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.4</mn><mo>×</mo><mn>0.3</mn><mo>×</mo><mn>0.01</mn><mo>=</mo><mn>0.0012</mn></mrow></math></span>. Ham score: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.6</mn><mo>×</mo><mn>0.02</mn><mo>×</mo><mn>0.1</mn><mo>=</mo><mn>0.0012</mn></mrow></math></span>.

Posterior <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>spam</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span>. The two words cancel out.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

posterior ∝ likelihood × prior; base rates matter most when the event is rare.

</div>

## Where this breaks in practice

- Confusing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo></mrow></math></span> is the **base-rate fallacy**.
- A classifier with high recall and a 5% false-positive rate has precision near 17% when positives are 1% of the data. On imbalanced problems report precision and recall, not accuracy.

---
id: "10.40"
course: "ml-maths-course"
module: "10"
slug: "conjugate-priors-beta-binomial-updating"
title: "Conjugate Priors: Beta-Binomial Updating"
keyIdeas: ["With a Beta(⟦3582⟧) prior on ⟦102⟧ and ⟦1199⟧ successes in ⟦233⟧ trials, the posterior is Beta(⟦3587⟧, ⟦3588⟧).", "A prior is conjugate when the posterior stays in the same family; updating is adding counts.", "The posterior mean ⟦3589⟧ under a uniform prior is Laplace smoothing."]
question: "Prior Beta(1, 1). You see 3 clicks in 10 views. Give the posterior and its mean."
answer: "Step 1, Beta(1 + 3, 1 + 7) = Beta(4, 8). Step 2, mean ⟦3593⟧, versus the raw ⟦3594⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

Bayes' theorem usually needs an integral for the denominator. With a conjugate prior the answer is a formula: add the data's counts to the prior's.

## Updating

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mtext>Beta</mtext><mo stretchy="false">(</mo><mi>α</mi><mo separator="true">,</mo><mi>β</mi><mo stretchy="false">)</mo><mover><mo stretchy="true" minsize="3.0em">→</mo><mpadded width="+0.6em" lspace="0.3em"><mrow><mi>k</mi><mtext> successes in </mtext><mi>n</mi></mrow></mpadded></mover><mtext>Beta</mtext><mo stretchy="false">(</mo><mi>α</mi><mo>+</mo><mi>k</mi><mo separator="true">,</mo><mtext> </mtext><mi>β</mi><mo>+</mo><mi>n</mi><mo>−</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></math></span></span>
The prior acts like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>α</mi></mrow></math></span> imaginary successes and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>β</mi></mrow></math></span> imaginary failures.

~~~dg-raw
%%viz f_curve f=1 g=1320*Math.pow(x,3)*Math.pow(1-x,7) x=0,1 y=0,3.5 tk=0.25 fl=prior_Beta(1~_1) gl=posterior_Beta(4~_8) | Before data, every click rate is equally plausible (green). After 3 clicks in 10 views the posterior (orange) peaks near 0.3 but is still wide.
~~~

```
from scipy import stats
post = stats.beta(1 + 3, 1 + 7)
print(round(post.mean(), 3), np.round(post.interval(0.9), 2))  # 0.333 [0.14 0.56]
```

10 views leave a 90% interval from about 0.14 to 0.56.

<div class="take">
<div class="lab">Key takeaway</div>

with a Beta prior and binary data, the posterior is a Beta with the counts added.

</div>

## Where this breaks in practice

- A strong prior (large <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>α</mi><mo>+</mo><mi>β</mi></mrow></math></span>) can swamp small data. State it and check how much it moves the answer.

---
id: "10.22"
course: "ml-maths-course"
module: "10"
slug: "markov-and-chebyshev-inequalities"
title: "Markov and Chebyshev Inequalities"
keyIdeas: ["Markov: for ⟦3450⟧, ⟦3451⟧.", "Chebyshev: ⟦3452⟧ for any distribution with finite variance.", "Both are loose but need almost no assumptions."]
question: "⟦3393⟧ with probability 0.1, else 0. What does Chebyshev say about ⟦3449⟧, and what is the truth?"
answer: "Step 1, ⟦3428⟧, so 9 is ⟦3457⟧ standard deviations. Chebyshev gives at most ⟦3458⟧. Step 2, the truth: ⟦3459⟧ only when ⟦3393⟧, probability ⟦3460⟧. The bound is nearly tight here."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

Sometimes you know only the mean, or only the mean and variance. These inequalities still give guarantees that hold for every distribution.

## The bounds

| Inequality | Needs | Statement |
| --- | --- | --- |
| Markov | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>≥</mo><mn>0</mn></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>≥</mo><mi>a</mi><mo stretchy="false">)</mo><mo>≤</mo><mi mathvariant="normal">𝔼</mi><mi>X</mi><mi mathvariant="normal">/</mi><mi>a</mi></mrow></math></span> |
| Chebyshev | finite variance | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mo stretchy="false">∣</mo><mi>X</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">∣</mo><mo>≥</mo><mi>k</mi><mi>σ</mi><mo stretchy="false">)</mo><mo>≤</mo><mn>1</mn><mi mathvariant="normal">/</mi><msup><mi>k</mi><mn>2</mn></msup></mrow></math></span> |

<div class="ex">
<div class="lab">Worked example</div>

**latency**

Mean latency 100 ms. Markov: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>latency</mtext><mo>≥</mo><mn>1000</mn><mo stretchy="false">)</mo><mo>≤</mo><mn>10</mn><mi mathvariant="normal">%</mi></mrow></math></span>, for any distribution of non-negative values.

Add standard deviation 50 ms. 900 ms above the mean is 18 standard deviations; Chebyshev gives at most <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>324</mn><mo>≈</mo><mn>0.3</mn><mi mathvariant="normal">%</mi></mrow></math></span>.

</div>

~~~dg-raw
%%viz f_bars v=25,4.6,11.1,0.27 l=bound_k=2,normal_k=2,bound_k=3,normal_k=3 hi=none yl=%_beyond_k_standard_deviations | Chebyshev's bound against the actual Gaussian tail, at k = 2 and 3 standard deviations. The bound holds for any distribution, so for a Gaussian it is loose.
~~~

```
from scipy import stats
for k in (2, 3):
    print(k, round(1 / k ** 2, 3), round(2 * stats.norm.sf(k), 4))
# 2 0.25 0.0455
# 3 0.111 0.0027
```

<div class="take">
<div class="lab">Key takeaway</div>

with only a mean (Markov) or a mean and variance (Chebyshev), tail probabilities are still bounded.

</div>

## Where this breaks in practice

- Using a Gaussian tail when the data are heavy-tailed underestimates rare events. Chebyshev is the safe fallback.

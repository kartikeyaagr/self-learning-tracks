---
id: "10.20"
course: "ml-maths-course"
module: "10"
slug: "higher-moments-skewness-and-kurtosis"
title: "Higher Moments: Skewness and Kurtosis"
keyIdeas: ["The ⟦1199⟧-th moment is ⟦3440⟧; central moments use ⟦3441⟧.", "Skewness ⟦3442⟧ measures lopsidedness; positive means a long right tail.", "Excess kurtosis ⟦3443⟧ measures tail weight relative to a Gaussian."]
question: "An exponential distribution has skewness 2. Which is larger, its mean or its median?"
answer: "Step 1, positive skew means a long right tail. Step 2, the tail pulls the mean up, so the mean (1/λ) is larger than the median (ln 2/λ)."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["SciPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

Mean and variance miss shape. Two distributions can share both and still differ in how often extreme values happen, which matters for losses, latencies and risk.

## Shape numbers

| Moment | Formula | Gaussian value |
| --- | --- | --- |
| skewness | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mn>3</mn></msup><mo stretchy="false">]</mo><mi mathvariant="normal">/</mi><msup><mi>σ</mi><mn>3</mn></msup></mrow></math></span> | 0 |
| excess kurtosis | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mn>4</mn></msup><mo stretchy="false">]</mo><mi mathvariant="normal">/</mi><msup><mi>σ</mi><mn>4</mn></msup><mo>−</mo><mn>3</mn></mrow></math></span> | 0 |

~~~dg-raw
%%viz f_curve f=Math.exp(-x) x=0,5 y=0,1 tk=1 pts=0.693,median_0.69;1,mean_1 | The exponential with rate 1 is skewed right: its mean (1) sits to the right of its median (0.69).
~~~

```
from scipy import stats
for name, d in [('normal', stats.norm()), ('expon', stats.expon()), ('t5', stats.t(5))]:
    s, k = d.stats(moments='sk')
    print(name, float(s), float(k))
# normal 0.0 0.0
# expon 2.0 6.0
# t5 0.0 6.0
```

<div class="take">
<div class="lab">Key takeaway</div>

skewness measures lopsidedness and kurtosis measures heavy tails.

</div>

## Where this breaks in practice

- Sample kurtosis is dominated by a few extreme points and is very noisy.

---
id: "10.06"
course: "ml-maths-course"
module: "10"
slug: "the-cumulative-distribution-function"
title: "The Cumulative Distribution Function"
keyIdeas: ["The CDF ⟦3285⟧ exists for every random variable: discrete, continuous or mixed.", "It never decreases, starts at 0 and ends at 1, and ⟦3286⟧.", "Where the density is continuous, ⟦2988⟧ (the fundamental theorem of calculus)."]
question: "For ⟦3283⟧ on ⟦4⟧, find ⟦2734⟧ and ⟦3284⟧."
answer: "Step 1, ⟦3293⟧. Step 2, ⟦3294⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous random variables"]
tools: ["SciPy"]
topic: {"id": "10.02", "title": "Continuous random variables"}
---

## The big picture

The density answers "how thick is the probability here". The CDF answers "how much probability is at or below here", which is what you usually want for thresholds, percentiles and tests.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>≤</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msubsup><mo>∫</mo><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow><mi>x</mi></msubsup><mi>p</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>t</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=1-Math.exp(-2*x) g=2*Math.exp(-2*x) x=0,3 y=0,2 tk=0.5 fl=CDF_F(x) gl=density_p(x) | The exponential with rate 2. The CDF (green) climbs from 0 to 1. Its slope at each x is the density (orange).
~~~

```
from scipy import stats
d = stats.expon(scale=0.5)
print(round(d.cdf(1) - d.cdf(0.5), 4), round(np.exp(-1) - np.exp(-2), 4))  # 0.2325 0.2325
```

<div class="ex">
<div class="lab">Worked example</div>

**a die's CDF**

For a fair die, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo stretchy="false">(</mo><mn>3</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo stretchy="false">(</mo><mn>3.7</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span> too. The CDF of a discrete variable is a staircase that jumps by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>6</mn></mrow></math></span> at each face.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

the CDF gives "probability at or below x", works for any variable, and its slope is the density.

</div>

## Where this breaks in practice

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>&lt;</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>≤</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> differ for discrete variables. Check which one a library function computes.

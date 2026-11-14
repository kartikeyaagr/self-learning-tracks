---
id: "10.26"
course: "ml-maths-course"
module: "10"
slug: "counting-with-binomial-coefficients"
title: "Counting With Binomial Coefficients"
keyIdeas: ["⟦3484⟧ counts the ways to choose ⟦1199⟧ items from ⟦233⟧ when order does not matter.", "Symmetry: ⟦3485⟧; and ⟦3486⟧.", "Factorials overflow fast; work with `math.comb` or log-gamma."]
question: "How many ways can exactly 2 of 5 emails be spam?"
answer: "Step 1, ⟦3492⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["Python"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

The binomial distribution needs to count "which <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> of the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> trials succeeded". That count is the binomial coefficient.

## Choosing

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mrow><mo fence="true">(</mo><mfrac linethickness="0px"><mi>n</mi><mi>k</mi></mfrac><mo fence="true">)</mo></mrow><mo>=</mo><mfrac><mrow><mi>n</mi><mo stretchy="false">!</mo></mrow><mrow><mi>k</mi><mo stretchy="false">!</mo><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mi>k</mi><mo stretchy="false">)</mo><mo stretchy="false">!</mo></mrow></mfrac></mrow></math></span></span>
Pick <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> items in order (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>⋯</mo><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mi>k</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> ways), then divide by the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo stretchy="false">!</mo></mrow></math></span> orders that give the same set.

~~~dg-raw
%%viz f_bars v=1,5,10,10,5,1 l=0,1,2,3,4,5 hi=none yl=ways_to_pick_k_of_5 | The binomial coefficients for n = 5. They are symmetric and add up to 2⁵ = 32.
~~~

```
from math import comb, lgamma, exp
print([comb(5, k) for k in range(6)], comb(100, 50) > 1e29)  # [1, 5, 10, 10, 5, 1] True
print(round(exp(lgamma(101) - 2 * lgamma(51)) / comb(100, 50), 6))  # 1.0
```

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mfrac linethickness="0px"><mi>n</mi><mi>k</mi></mfrac><mo fence="true">)</mo></mrow></math></span> counts subsets of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>100</mn><mo stretchy="false">!</mo></mrow></math></span> is far beyond float64's range for products; compute in log space.

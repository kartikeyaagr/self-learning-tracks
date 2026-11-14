---
id: "12.18"
course: "ml-maths-course"
module: "12"
slug: "intervals-for-a-mean-z-and-t"
title: "Intervals for a Mean: z and t"
keyIdeas: ["Large ⟦233⟧: ⟦3935⟧.", "Small ⟦233⟧ with roughly normal data: use ⟦3937⟧ quantiles, which are wider because ⟦1032⟧ itself is noisy.", "With 10 runs, 1.96 gives an interval about 13% too narrow."]
question: "Ten training runs: mean 5.0, ⟦3936⟧. Give the 95% ⟦166⟧ interval."
answer: "Step 1, ⟦3938⟧. Step 2, ⟦3939⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Interval estimation"]
tools: ["SciPy"]
topic: {"id": "12.03", "title": "Interval estimation"}
---

## The big picture

Benchmarks are often averaged over a handful of seeds. With so few runs, the standard deviation is itself uncertain, and the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> distribution accounts for that.

## The t multiplier

~~~dg-raw
%%viz f_bars v=12.71,2.78,2.26,2.09,1.96 l=n_=_2,n_=_5,n_=_10,n_=_20,large_n hi=none yl=95%_multiplier | The 95% multiplier for a mean from n values. It falls toward 1.96 as n grows; with 2 values it is 12.7.
~~~

```
from scipy import stats
t = stats.t(9).ppf(0.975)
print(round(t, 3), round(t * 2 / np.sqrt(10), 2), round(1.96 * 2 / np.sqrt(10), 2))  # 2.262 1.43 1.24
```

<div class="take">
<div class="lab">Key takeaway</div>

for small samples use <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>t</mi><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow></math></span> instead of 1.96.

</div>

## Where this breaks in practice

- Five seeds with a standard deviation of 1 point cannot separate models that differ by half a point.

---
id: "12.21"
course: "ml-maths-course"
module: "12"
slug: "bootstrap-intervals"
title: "Bootstrap Intervals"
keyIdeas: ["Resample the data with replacement, recompute the statistic many times, and take percentiles.", "It works for statistics with no simple standard-error formula: F1, AUC, median latency.", "Grouped data need a cluster bootstrap that resamples whole groups."]
question: "Why resample with replacement rather than without?"
answer: "Step 1, without replacement every resample of size ⟦233⟧ is the original data. Step 2, with replacement some rows repeat and others drop out, which mimics drawing a fresh sample."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Interval estimation"]
tools: ["NumPy"]
topic: {"id": "12.03", "title": "Interval estimation"}
---

## The big picture

The bootstrap pretends your sample is the population and samples from it. The spread of the recomputed statistic stands in for the spread you'd see across real new samples.

## The recipe

~~~dg-flow Percentile bootstrap for any statistic.
Data (n rows) -(resample n rows with replacement)-> bootstrap sample -(compute statistic)-> one replicate -(repeat B = 1000 to 10000 times)-> *2.5th and 97.5th percentiles = 95% interval
~~~

```
rng = np.random.default_rng(0)
correct = np.r_[np.ones(900), np.zeros(100)]
boot = [rng.choice(correct, size=1000, replace=True).mean() for _ in range(5000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(round(lo, 3), round(hi, 3))  # 0.881 0.918
```

For 90% accuracy on 1000 examples the bootstrap gives nearly the Wilson interval <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0.880</mn><mo separator="true">,</mo><mn>0.917</mn><mo stretchy="false">]</mo></mrow></math></span>, with no formula needed.

<div class="take">
<div class="lab">Key takeaway</div>

the bootstrap gives an interval for almost any statistic by resampling.

</div>

## Where this breaks in practice

- It fails for extremes (the maximum) and for tiny samples, where the data do not represent the population.

---
id: "12.04"
course: "ml-maths-course"
module: "12"
slug: "bessel-s-correction"
title: "Bessel's Correction"
keyIdeas: ["The sample variance ⟦3855⟧ divides by ⟦2438⟧.", "Dividing by ⟦233⟧ underestimates ⟦3780⟧ by the factor ⟦3856⟧, because ⟦3857⟧ was fitted to the same data.", "`np.var` divides by ⟦233⟧ unless you pass `ddof=1`; pandas `.var()` divides by ⟦2438⟧."]
question: "Data ⟦3854⟧. Compute the variance dividing by ⟦233⟧ and by ⟦2438⟧."
answer: "Step 1, mean 4, squared deviations ⟦3861⟧, sum 8. Step 2, ⟦3862⟧ and ⟦991⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

Deviations from the sample mean are smaller than deviations from the true mean, because the sample mean sits in the middle of the sample. Dividing by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></math></span> makes up for it.

## Why n − 1

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∑</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span> has only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></math></span> free pieces: the deviations must add up to zero, so the last one is fixed by the others.

```
rng = np.random.default_rng(0)
x = rng.normal(0, 2, size=(200000, 5))
print(round(x.var(axis=1).mean(), 1), round(x.var(axis=1, ddof=1).mean(), 1))  # 3.2 4.0
```

The true variance is 4. Dividing by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>5</mn></mrow></math></span> averages <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4</mn><mo>×</mo><mn>4</mn><mi mathvariant="normal">/</mi><mn>5</mn><mo>=</mo><mn>3.2</mn></mrow></math></span>.

~~~dg-raw
%%viz f_bars v=3.2,4,4 l=divide_by_n,divide_by_n_−_1,true_variance hi=none yl=average_estimate_(n_=_5) | With 5 points per sample, dividing by n underestimates the variance by 20% on average; dividing by n − 1 is right on average.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

divide by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></math></span> for an unbiased sample variance.

</div>

## Where this breaks in practice

- For large <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> the difference is negligible; for 5 seeds of a benchmark it is 20%.

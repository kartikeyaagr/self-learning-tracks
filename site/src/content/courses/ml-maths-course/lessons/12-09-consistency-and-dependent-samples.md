---
id: "12.09"
course: "ml-maths-course"
module: "12"
slug: "consistency-and-dependent-samples"
title: "Consistency and Dependent Samples"
keyIdeas: ["An estimator is consistent if it converges to the true value as ⟦233⟧ grows.", "Consistency is a minimum requirement, not a guarantee of good small-sample behaviour.", "With dependent samples the effective sample size is smaller than the row count."]
question: "100 users each contribute 50 nearly identical rows. Is the effective sample size 5000?"
answer: "Step 1, no: rows from one user carry mostly the same information. Step 2, the effective size is closer to 100, so the standard error is about 7 times larger than ⟦3888⟧ suggests."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

All the formulas so far assume independent draws. Real datasets often aren't: many rows per user, sequences in time, near-duplicate images.

## Clustered rows

```
rng = np.random.default_rng(0)
est = []
for _ in range(4000):
    user = rng.normal(0, 1, 100)                       # one effect per user
    rows = np.repeat(user, 50) + rng.normal(0, 0.1, 5000)
    est.append(rows.mean())
print(round(np.std(est), 2), round(1 / np.sqrt(5000), 3), round(1 / np.sqrt(100), 3))  # 0.1 0.014 0.1
```

The naive standard error, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mn>5000</mn></msqrt><mo>≈</mo><mn>0.014</mn></mrow></math></span>, is 7 times too small; the true spread matches 100 independent users.

~~~dg-raw
%%viz f_bars v=1.4,10 l=naive_(5000_rows),actual_(100_users) hi=none yl=standard_error_×_100 | Treating 5,000 correlated rows as independent understates the uncertainty by a factor of 7.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

consistency needs only convergence, but honest error bars need the effective number of independent units.

</div>

## Where this breaks in practice

- Splitting such data randomly into train and test leaks users across the split. Split by user.

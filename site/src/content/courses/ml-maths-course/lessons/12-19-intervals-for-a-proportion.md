---
id: "12.19"
course: "ml-maths-course"
module: "12"
slug: "intervals-for-a-proportion"
title: "Intervals for a Proportion"
keyIdeas: ["The Wald interval ⟦3940⟧ works for large ⟦233⟧ and ⟦3479⟧ away from 0 and 1.", "Near 0 or 1, use the Wilson interval.", "Rule of three: with zero events in ⟦233⟧ trials, the 95% upper bound is about ⟦3941⟧."]
question: "Zero errors in 200 test cases. What error rate can you rule out?"
answer: "Step 1, the Wald interval is ⟦3942⟧, which claims certainty. Step 2, the rule of three gives an upper bound of ⟦3943⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Interval estimation"]
tools: ["NumPy"]
topic: {"id": "12.03", "title": "Interval estimation"}
---

## The big picture

Accuracy, precision and error rates are proportions. Their intervals have a known failure mode near 0 and 1.

## Wald and Wilson

```
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(round(c - h, 4)), float(round(c + h, 4))
print(wilson(900, 1000), wilson(0, 200))  # (0.8798, 0.9171) (0.0, 0.0188)
```

~~~dg-raw
%%viz f_bars v=0,1.88,1.5 l=Wald,Wilson,rule_of_three hi=none yl=upper_bound_%_after_0_of_200 | Upper 95% bounds on an error rate after 0 errors in 200 cases. Wald says 0%; Wilson and the rule of three say it could be near 2%.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

use Wilson intervals for proportions, especially near 0 or 1.

</div>

## Where this breaks in practice

- "Zero failures in testing" is not "zero failure rate". Report the upper bound.

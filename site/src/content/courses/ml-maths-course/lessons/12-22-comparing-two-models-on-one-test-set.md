---
id: "12.22"
course: "ml-maths-course"
module: "12"
slug: "comparing-two-models-on-one-test-set"
title: "Comparing Two Models on One Test Set"
keyIdeas: ["Both models are scored on the same examples, so use a paired method, not overlapping intervals.", "McNemar's test looks only at the examples where the models disagree.", "A paired bootstrap resamples examples and computes the difference in scores each time."]
question: "A gets 900 of 1000 right, B gets 910. They disagree on 10 examples, all won by B. Is B better?"
answer: "Step 1, under \"no difference\" each disagreement is a fair coin. Step 2, 10 out of 10 has probability about 0.002: strong evidence that B is better."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Interval estimation"]
tools: ["SciPy"]
topic: {"id": "12.03", "title": "Interval estimation"}
---

## The big picture

The same one-point gap can be convincing or meaningless. What decides it is how many examples the two models disagree on.

## Pairing

```
from scipy.stats import binomtest
print(round(binomtest(10, 10).pvalue, 4), round(binomtest(105, 200).pvalue, 2))  # 0.002 0.52
```

Case 1: 10 disagreements, all won by B (p ≈ 0.002). Case 2: 200 disagreements, B wins 105 to 95 (p ≈ 0.52). Same accuracy gap.

~~~dg-raw
%%viz f_bars v=10,0,105,95 l=B_wins,A_wins,B_wins,A_wins hi=none yl=disagreements_(case_1_left~_case_2_right) | Two ways to get a 1-point gap. Left: every disagreement goes to B. Right: they split almost evenly, which looks like noise.
~~~

| Situation | Interval or test |
| --- | --- |
| mean, large <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><mo>±</mo><mn>1.96</mn><mtext> </mtext><mi>s</mi><mi mathvariant="normal">/</mi><msqrt><mi>n</mi></msqrt></mrow></math></span> |
| mean, small <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>t</mi><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow></math></span> |
| proportion | Wilson |
| complicated metric | bootstrap |
| two models, same test set | paired bootstrap or McNemar |

<div class="take">
<div class="lab">Key takeaway</div>

compare models with a paired method on the examples where they differ.

</div>

## Where this breaks in practice

- Testing many models on the same test set and reporting the best inflates the result; keep a final held-out set or correct for multiple comparisons.

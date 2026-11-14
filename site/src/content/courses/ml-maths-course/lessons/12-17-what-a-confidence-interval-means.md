---
id: "12.17"
course: "ml-maths-course"
module: "12"
slug: "what-a-confidence-interval-means"
title: "What a Confidence Interval Means"
keyIdeas: ["A 95% confidence interval comes from a procedure that covers the true value in 95% of repeated samples.", "One computed interval either contains the truth or not; \"95%\" describes the method.", "A Bayesian credible interval does make a probability statement about the parameter, given a prior."]
question: "You compute a 95% interval ⟦3934⟧. Is there a 95% probability the true accuracy is in it?"
answer: "Step 1, in the frequentist sense, no: the truth is fixed and the interval either contains it or not. Step 2, the 95% is the long-run hit rate of the procedure that produced it."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Interval estimation"]
tools: ["NumPy"]
topic: {"id": "12.03", "title": "Interval estimation"}
---

## The big picture

"Model B gets 91%, model A gets 90%." Is B better? Every evaluation number has noise, and an interval puts a bar on it.

## Coverage

For a mean with large <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><mo>±</mo><mn>1.96</mn><mtext> </mtext><mi>s</mi><mi mathvariant="normal">/</mi><msqrt><mi>n</mi></msqrt></mrow></math></span>, "the estimate plus or minus about two standard errors".

```
rng = np.random.default_rng(0)
samples = rng.exponential(1.0, size=(20000, 50))
m = samples.mean(1); se = samples.std(1, ddof=1) / np.sqrt(50)
covered = (m - 1.96 * se <= 1) & (1 <= m + 1.96 * se)
print(round(covered.mean(), 2))  # 0.93
```

Of 20,000 intervals from skewed samples of 50, about 93% contain the true mean, a bit short of 95% because the skew has not fully washed out.

~~~dg-raw
%%viz f_plane x=0,11 y=-2.5,3 tk=1 sg=0,0,11,0,k,d;1,-1.25,1,1.50,a;2,-0.90,2,1.90,a;3,-1.55,3,1.05,a;4,-0.50,4,2.25,a;5,-1.90,5,0.60,a;6,0.15,6,2.75,f;7,-1.10,7,1.45,a;8,-1.45,8,1.20,a;9,-0.70,9,2.00,a;10,-1.30,10,1.35,a | Ten intervals from ten samples (illustrative). Height is the distance from the true mean in units of 0.2, so the dashed line is the truth. Nine intervals cross it; one (orange) misses.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a confidence level is a property of the procedure, not of one interval.

</div>

## Where this breaks in practice

- Picking the best of many models on one test set makes the winner's interval too optimistic.

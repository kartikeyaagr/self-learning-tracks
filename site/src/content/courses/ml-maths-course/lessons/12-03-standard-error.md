---
id: "12.03"
course: "ml-maths-course"
module: "12"
slug: "standard-error"
title: "Standard Error"
keyIdeas: ["The standard error is the standard deviation of an estimator; for the mean it is ⟦3844⟧.", "In practice ⟦1239⟧ is unknown and estimated by the sample standard deviation ⟦1032⟧.", "Four times the data halves the standard error."]
question: "Test accuracy is 0.9 on 1000 examples. What is its standard error?"
answer: "Step 1, each example is a Bernoulli with variance ⟦3852⟧. Step 2, ⟦3853⟧, about one percentage point."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

A number without an error bar can't be compared with anything. The standard error is the error bar.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mtext>SE</mtext><mo stretchy="false">(</mo><mover accent="true"><mi>X</mi><mo>ˉ</mo></mover><mo stretchy="false">)</mo><mo>=</mo><mfrac><mi>σ</mi><msqrt><mi>n</mi></msqrt></mfrac><mo>≈</mo><mfrac><mi>s</mi><msqrt><mi>n</mi></msqrt></mfrac></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=100*Math.sqrt(0.09/x) x=100,10000 y=0,3.2 tk=2000 pts=1000,n_=_1000:_0.95;4000,n_=_4000:_0.47 | Standard error of a 90% accuracy, in percentage points, against test-set size. Going from 1,000 to 4,000 examples halves it.
~~~

```
p, n = 0.9, 1000
print(round(np.sqrt(p * (1 - p) / n), 4))  # 0.0095
```

<div class="take">
<div class="lab">Key takeaway</div>

standard error = spread ÷ √n; it tells you how much an estimate would move with new data.

</div>

## Where this breaks in practice

- A 0.5-point gain on a 1000-example test set is about half a standard error: noise, not progress.

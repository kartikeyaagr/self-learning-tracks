---
id: "11.11"
course: "ml-maths-course"
module: "11"
slug: "bayes-rule-with-densities"
title: "Bayes' Rule With Densities"
keyIdeas: ["⟦3680⟧.", "The numerator is easy; the denominator (the evidence) is an integral that rarely has a closed form.", "In one or two dimensions a grid approximation works."]
question: "A coin has unknown heads probability ⟦875⟧ with a uniform prior. You see 7 heads in 10 flips. What is the posterior?"
answer: "Step 1, the posterior is proportional to ⟦3682⟧. Step 2, that is Beta(8, 4), with mean ⟦3683⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two continuous variables"]
tools: ["NumPy"]
topic: {"id": "11.02", "title": "Two continuous variables"}
---

## The big picture

Bayesian inference is Bayes' theorem for a continuous parameter. It turns a prior and a likelihood into a full distribution over the parameter.

## On a grid

Evaluate prior × likelihood at many values of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi></mrow></math></span>, then divide by the total so it integrates to 1.

```
theta = np.linspace(0, 1, 1001)
unnorm = theta ** 7 * (1 - theta) ** 3
post = unnorm / (unnorm.sum() * (theta[1] - theta[0]))
print(round((theta * post).sum() * (theta[1] - theta[0]), 3))  # 0.667
```

~~~dg-raw
%%viz f_curve f=1320*Math.pow(x,7)*Math.pow(1-x,3) x=0,1 y=0,3.2 tk=0.25 pts=0.7,most_likely_0.7 | The posterior for a coin's heads probability after 7 heads in 10 flips, from a flat prior. It peaks at 0.7 but values from about 0.4 to 0.9 are still plausible.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

posterior density ∝ likelihood × prior; the normalising integral is the hard part.

</div>

## Where this breaks in practice

- Grids need <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>G</mi><mi>d</mi></msup></mrow></math></span> points in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions. Beyond a few parameters, use MCMC or variational inference.

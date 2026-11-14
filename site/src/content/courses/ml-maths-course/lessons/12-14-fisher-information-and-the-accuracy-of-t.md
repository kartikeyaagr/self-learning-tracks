---
id: "12.14"
course: "ml-maths-course"
module: "12"
slug: "fisher-information-and-the-accuracy-of-the-mle"
title: "Fisher Information and the Accuracy of the MLE"
keyIdeas: ["Fisher information ⟦3917⟧ is the expected curvature of the log-likelihood.", "For large ⟦233⟧, ⟦3918⟧.", "Cramér–Rao: no unbiased estimator has variance below ⟦3919⟧."]
question: "Find ⟦3916⟧ for a Bernoulli and the approximate variance of ⟦3479⟧."
answer: "Step 1, ⟦3922⟧. Step 2, ⟦3923⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["NumPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

A sharply peaked log-likelihood pins the parameter down; a flat one doesn't. Fisher information turns that curvature into an error bar.

## Curvature is information

~~~dg-raw
%%viz f_curve f=-50*(x*Math.log(x/0.5)+(1-x)*Math.log((1-x)/0.5))/(1) g=-500*(x*Math.log(x/0.5)+(1-x)*Math.log((1-x)/0.5)) x=0.3,0.7 y=-10,3 tk=0.1 ytk=2 fl=n_=_50 gl=n_=_500 | Log-likelihood of a coin's p around the MLE 0.5, for 50 and 500 flips. More data means a sharper peak and a smaller error bar.
~~~

```
rng = np.random.default_rng(0)
lam_hat = rng.poisson(2.0, size=(100000, 5)).mean(axis=1)
print(round(lam_hat.std(), 2), round(np.sqrt(2 / 5), 2))  # 0.63 0.63
```

For a Poisson, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo stretchy="false">(</mo><mi>λ</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>λ</mi></mrow></math></span>, so the MLE's standard deviation from 5 counts is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mn>2</mn><mi mathvariant="normal">/</mi><mn>5</mn></mrow></msqrt><mo>≈</mo><mn>0.63</mn></mrow></math></span>, matching the simulation.

<div class="take">
<div class="lab">Key takeaway</div>

the MLE's variance is about one over (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> times the Fisher information).

</div>

## Where this breaks in practice

- The approximation needs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> large compared with the number of parameters, which deep networks violate. Natural-gradient methods and K-FAC still use the Fisher matrix to precondition training.

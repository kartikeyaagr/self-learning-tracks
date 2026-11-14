---
id: "12.13"
course: "ml-maths-course"
module: "12"
slug: "training-losses-are-negative-log-likelihoods"
title: "Training Losses Are Negative Log-Likelihoods"
keyIdeas: ["MSE is the negative log-likelihood of a Gaussian noise model with fixed variance.", "Binary cross-entropy and softmax cross-entropy are negative log-likelihoods of Bernoulli and categorical models.", "Mean absolute error corresponds to Laplace noise."]
question: "Show that minimising the Gaussian negative log-likelihood with fixed ⟦1239⟧ is minimising MSE."
answer: "Step 1, each example's NLL is ⟦3915⟧. Step 2, the first term is constant and the second is squared error times a constant."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["NumPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

Choosing a loss is choosing a noise model. Seeing losses as likelihoods tells you what each one assumes.

## The dictionary

~~~dg-flow Every standard loss is a negative log-likelihood.
Gaussian noise, fixed variance -(negative log-likelihood)-> *mean squared error
Laplace noise -(negative log-likelihood)-> *mean absolute error
Bernoulli output σ(z) -(negative log-likelihood)-> *binary cross-entropy
Categorical output softmax(z) -(negative log-likelihood)-> *cross-entropy
~~~

```
y = np.array([1.0, 2, 4]); f = np.array([1.5, 2, 3])
sigma = 1.0
nll = (0.5 * np.log(2 * np.pi * sigma ** 2) + (y - f) ** 2 / (2 * sigma ** 2)).mean()
print(round(nll, 4), round(0.5 * np.log(2 * np.pi) + 0.5 * ((y - f) ** 2).mean(), 4))  # 1.1273 1.1273
```

<div class="take">
<div class="lab">Key takeaway</div>

MSE, MAE and cross-entropy are maximum likelihood under Gaussian, Laplace and categorical models.

</div>

## Where this breaks in practice

- Using MSE on counts or probabilities assumes Gaussian noise where it does not fit; Poisson or Bernoulli likelihoods are better.

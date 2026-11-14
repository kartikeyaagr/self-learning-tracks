---
id: "11.18"
course: "ml-maths-course"
module: "11"
slug: "linear-maps-of-random-vectors"
title: "Linear Maps of Random Vectors"
keyIdeas: ["⟦3731⟧ and ⟦3732⟧.", "For one direction ⟦364⟧, ⟦3733⟧.", "The shift ⟦682⟧ moves the mean but not the spread."]
question: "⟦3729⟧, ⟦1741⟧. Find the mean and variance of ⟦3730⟧."
answer: "Step 1, ⟦3736⟧, mean ⟦3737⟧. Step 2, ⟦3738⟧, so ⟦3739⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

A linear layer maps a random vector to another. These two formulas track its mean and covariance exactly.

## The rules

~~~dg-flow Pushing a mean and covariance through a linear layer.
X with mean μ, covariance Σ -(Y = A X + b)-> *Y with mean A μ + b, covariance A Σ Aᵀ
~~~

```
rng = np.random.default_rng(0)
mu = np.array([1.0, 2]); S = np.array([[4.0, 2], [2, 3]])
X = rng.multivariate_normal(mu, S, size=500000)
A = np.array([[1.0, -1], [1, 1]])
print(np.round(A @ S @ A.T, 2).tolist())  # [[3.0, 1.0], [1.0, 11.0]]
print(np.round(np.cov(X @ A.T, rowvar=False), 1).tolist())  # [[3.0, 1.0], [1.0, 11.0]]
```

<div class="take">
<div class="lab">Key takeaway</div>

means move as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>μ</mi><mo>+</mo><mi>b</mi></mrow></math></span> and covariances as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi mathvariant="normal">Σ</mi><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>.

</div>

## Where this breaks in practice

- These rules track only mean and covariance. A nonlinear layer changes the distribution's shape too.

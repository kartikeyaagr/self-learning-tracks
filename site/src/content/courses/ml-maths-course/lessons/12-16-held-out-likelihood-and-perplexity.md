---
id: "12.16"
course: "ml-maths-course"
module: "12"
slug: "held-out-likelihood-and-perplexity"
title: "Held-Out Likelihood and Perplexity"
keyIdeas: ["Compare probabilistic models by their average log-likelihood on held-out data, not training data.", "Perplexity is ⟦1577⟧ of the average negative log-likelihood per token: the effective number of equally likely choices.", "Training likelihood always improves with more parameters; held-out likelihood reveals overfitting."]
question: "A model assigns probabilities 0.1, 0.05, 0.2 and 0.02 to four true tokens. What is its perplexity?"
answer: "Step 1, the average NLL is ⟦3932⟧. Step 2, perplexity ⟦3933⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["NumPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

Likelihood on the training data rewards memorising. Likelihood on data the model never saw measures how well it generalises.

## Perplexity

```
p_true_token = np.array([0.1, 0.05, 0.2, 0.02])
nll = -np.log(p_true_token).mean()
print(round(nll, 3), round(np.exp(nll), 2))  # 2.705 14.95
```

A perplexity of 15 means the model is, on average, as uncertain as if it were choosing uniformly among 15 tokens.

## Overfitting shows up held out

```
rng = np.random.default_rng(0)
x = np.linspace(-1, 1, 20); y = np.sin(3 * x) + rng.normal(0, 0.2, 20)
xt = np.linspace(-0.95, 0.95, 200); yt = np.sin(3 * xt) + rng.normal(0, 0.2, 200)
for deg in (3, 15):
    c = np.polyfit(x, y, deg)
    print(deg, round(np.mean((np.polyval(c, x) - y) ** 2), 3), round(np.mean((np.polyval(c, xt) - yt) ** 2), 3))
# 3 0.016 0.055
# 15 0.006 0.105
```

Degree 15 has lower training error (0.006 against 0.016) but double the held-out error (0.105 against 0.055). Under a Gaussian model, lower squared error is higher likelihood.

~~~dg-raw
%%viz f_curve f=Math.exp(x) x=0,5 y=0,150 tk=1 ytk=25 pts=2.705,NLL_2.7:_perplexity_15 | Perplexity is e raised to the average negative log-likelihood, so small gains in NLL are large changes in perplexity.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

judge models by held-out log-likelihood; perplexity is its exponentiated per-token form.

</div>

## Where this breaks in practice

- Perplexities from different tokenisers are not comparable: the "choices" are different units.

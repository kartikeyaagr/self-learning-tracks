---
id: "11.23"
course: "ml-maths-course"
module: "11"
slug: "uncorrelated-is-not-independent"
title: "Uncorrelated Is Not Independent"
keyIdeas: ["Independent implies uncorrelated (when variances are finite), not the reverse.", "⟦3341⟧ and ⟦3323⟧ are uncorrelated but ⟦512⟧ is a function of ⟦513⟧.", "Correlation measures straight-line association only, and says nothing about causation."]
question: "Show ⟦3774⟧ for standard normal ⟦513⟧."
answer: "Step 1, ⟦3775⟧. Step 2, ⟦3776⟧ by symmetry and ⟦3777⟧, so the covariance is 0."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Covariance and correlation"]
tools: ["NumPy"]
topic: {"id": "11.04", "title": "Covariance and correlation"}
---

## The big picture

A correlation near zero is often read as "no relationship". That is only true for straight-line relationships.

## A perfect relationship with zero correlation

~~~dg-raw
%%viz f_curve f=x*x x=-2,2 y=-0.5,4.2 tk=1 | Y = X². Y is fully determined by X, but the U-shape has no overall tilt, so the correlation is 0.
~~~

```
rng = np.random.default_rng(0)
x = rng.normal(size=1_000_000)
print(abs(np.corrcoef(x, x ** 2)[0, 1]) < 0.01)  # True
```

~~~dg-flow What correlation does and does not tell you.
ρ near 0 -(could be)-> independent, or a strong nonlinear relation
ρ near 1 -(could be)-> causal link, common cause, or selection effect
*Check -(scatter plot, binned conditionals, mutual information)-> what the relationship actually is
~~~

<div class="take">
<div class="lab">Key takeaway</div>

zero correlation rules out a linear trend, not a relationship.

</div>

## Where this breaks in practice

- Combining groups can reverse a correlation (Simpson's paradox). Check for a hidden grouping variable.

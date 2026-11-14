---
id: "12.08"
course: "ml-maths-course"
module: "12"
slug: "shrinkage-biased-estimators-that-win"
title: "Shrinkage: Biased Estimators That Win"
keyIdeas: ["Shrinking an estimate toward a fixed point adds bias and removes variance.", "It pays off when data are scarce and the truth is near the point you shrink toward.", "Ridge regression, weight decay and early stopping are all shrinkage."]
question: "Estimate ⟦1238⟧ from 5 samples with ⟦3029⟧ using ⟦3881⟧. What ⟦119⟧ minimises the MSE when ⟦3882⟧?"
answer: "Step 1, MSE ⟦3884⟧. Step 2, set the derivative to zero: ⟦3885⟧, so ⟦3886⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

Regularisation is the deliberate choice of a biased estimator. This lesson shows the trade in its simplest form.

## MSE as a function of the shrink factor

~~~dg-raw
%%viz f_curve f=(x-1)*(x-1)+0.8*x*x x=0,1.2 y=0,1.2 tk=0.2 ytk=0.2 pts=0.556,best_c_≈_0.56;1,unbiased_c_=_1 | MSE of c·X̄ for μ = 1, σ = 2, n = 5. Shrinking to about 0.56 of the sample mean cuts the error from 0.8 to 0.44.
~~~

```
c = np.linspace(0, 1.2, 1201)
mse = (c - 1) ** 2 + 0.8 * c ** 2
print(round(c[mse.argmin()], 3), round(mse.min(), 3))  # 0.556 0.444
```

<div class="ex">
<div class="lab">Worked example</div>

**when shrinkage hurts**

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>=</mo><mn>10</mn></mrow></math></span>, shrinking by 0.8 adds a bias of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>2</mn></mrow></math></span> and an MSE of at least 4. Shrinkage helps only when the truth is near the target.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

with little data, a biased but stable estimate usually beats an unbiased noisy one.

</div>

## Where this breaks in practice

- The best amount of shrinkage depends on the unknown truth; choose it by cross-validation.

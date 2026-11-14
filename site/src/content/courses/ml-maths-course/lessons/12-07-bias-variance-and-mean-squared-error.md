---
id: "12.07"
course: "ml-maths-course"
module: "12"
slug: "bias-variance-and-mean-squared-error"
title: "Bias, Variance and Mean Squared Error"
keyIdeas: ["Bias is ⟦3873⟧; variance is ⟦3874⟧.", "⟦3875⟧.", "An unbiased estimator is not automatically better; MSE is what usually matters."]
question: "An estimator has bias ⟦3871⟧ and variance ⟦3872⟧. Another has bias 0 and variance ⟦3779⟧. Which has lower MSE?"
answer: "Step 1, first: ⟦3880⟧. Step 2, second: ⟦3779⟧. The biased one wins."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

Judge an estimator by how far it typically lands from the truth. That distance splits into a systematic part and a random part.

## Two sources of error

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">(</mo><mover accent="true"><mi>θ</mi><mo>^</mo></mover><mo>−</mo><mi>θ</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>=</mo><munder><munder><mrow><mo stretchy="false">(</mo><mi mathvariant="normal">𝔼</mi><mover accent="true"><mi>θ</mi><mo>^</mo></mover><mo>−</mo><mi>θ</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow><mo stretchy="true">⏟</mo></munder><msup><mtext>bias</mtext><mn>2</mn></msup></munder><mo>+</mo><munder><munder><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mover accent="true"><mi>θ</mi><mo>^</mo></mover></mrow><mo stretchy="true">⏟</mo></munder><mtext>variance</mtext></munder></mrow></math></span></span>
Think of darts: bias is how far the cloud's centre is from the bullseye; variance is how spread out the cloud is.

| Property | Meaning |
| --- | --- |
| Unbiased | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mover accent="true"><mi>θ</mi><mo>^</mo></mover><mo>=</mo><mi>θ</mi></mrow></math></span> |
| Consistent | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>θ</mi><mo>^</mo></mover><mo>→</mo><mi>θ</mi></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>→</mo><mi mathvariant="normal">∞</mi></mrow></math></span> |
| Low MSE | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mtext>bias</mtext><mn>2</mn></msup><mo>+</mo><mi mathvariant="normal">Var</mi><mo>⁡</mo><mover accent="true"><mi>θ</mi><mo>^</mo></mover></mrow></math></span> is small |

```
rng = np.random.default_rng(0)
xbar = rng.normal(1, 2, size=(400000, 5)).mean(1)
for c in (1.0, 0.8):
    e = c * xbar - 1
    print(c, round(e.mean(), 2), round(e.var(), 2), round((e ** 2).mean(), 2))
# 1.0 0.0 0.8 0.8
# 0.8 -0.2 0.51 0.55
```

<div class="take">
<div class="lab">Key takeaway</div>

MSE = bias² + variance; trading a little bias for less variance can lower it.

</div>

## Where this breaks in practice

- The bias-variance split for an estimator is the same idea as underfitting and overfitting for a model.

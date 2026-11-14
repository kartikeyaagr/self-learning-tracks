---
id: "10.19"
course: "ml-maths-course"
module: "10"
slug: "rules-for-variance"
title: "Rules for Variance"
keyIdeas: ["⟦3431⟧: shifting does nothing, scaling by ⟦364⟧ scales ⟦1239⟧ by ⟦3366⟧.", "For independent ⟦513⟧ and ⟦512⟧, ⟦3432⟧.", "The mean of ⟦233⟧ independent copies has variance ⟦3433⟧, so its standard deviation shrinks like ⟦3434⟧."]
question: "Per-example gradient components have variance 4. What is the standard deviation of a batch mean with ⟦3429⟧? With ⟦3430⟧?"
answer: "Step 1, ⟦3437⟧, standard deviation ⟦131⟧. Step 2, ⟦3438⟧, standard deviation ⟦3439⟧. Four times the batch halves the noise."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

These rules explain why bigger minibatches give less noisy gradients, and why averaging an ensemble helps.

## The rules

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>a</mi><mi>X</mi><mo>+</mo><mi>b</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>a</mi><mn>2</mn></msup><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi></mrow></math></span>.
- Independent: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>X</mi><mo>+</mo><mi>Y</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>+</mo><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>Y</mi></mrow></math></span>. In general add <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">Cov</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>X</mi><mo separator="true">,</mo><mi>Y</mi><mo stretchy="false">)</mo></mrow></math></span> (lesson 11.21).
- Mean of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> independent copies: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mover accent="true"><mi>X</mi><mo>ˉ</mo></mover><mo stretchy="false">)</mo><mo>=</mo><msup><mi>σ</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=2/Math.sqrt(x) x=1,256 y=0,2.1 tk=64 pts=64,B_=_64:_0.25;256,B_=_256:_0.125 | Standard deviation of a batch-mean gradient, 2/√B. Going from 64 to 256 examples only halves it.
~~~

```
rng = np.random.default_rng(0)
g = rng.normal(0, 2, size=(20000, 256))
print(round(g[:, :64].mean(1).std(), 3), round(g.mean(1).std(), 3))  # 0.25 0.125
```

<div class="take">
<div class="lab">Key takeaway</div>

variances add for independent pieces and scale with the square; averaging <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> copies divides variance by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- Correlated examples (duplicates, sequences from one user) average out more slowly than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mi>n</mi></msqrt></mrow></math></span>.

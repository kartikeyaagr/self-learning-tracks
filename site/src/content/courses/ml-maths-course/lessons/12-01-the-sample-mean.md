---
id: "12.01"
course: "ml-maths-course"
module: "12"
slug: "the-sample-mean"
title: "The Sample Mean"
keyIdeas: ["The sample mean ⟦3837⟧ estimates the population mean ⟦1238⟧.", "⟦3838⟧ by linearity, and ⟦3839⟧ for independent draws.", "Test accuracy, a batch loss and a Monte Carlo estimate are all sample means."]
question: "A die has variance ⟦3836⟧. What is the standard deviation of the mean of 25 rolls?"
answer: "Step 1, ⟦3845⟧. Step 2, standard deviation ⟦3846⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Sampling behavior and estimators"]
tools: ["NumPy"]
topic: {"id": "12.01", "title": "Sampling behavior and estimators"}
---

## The big picture

Every number you report from data is uncertain. Draw different data and you'd get a different average. This module makes that precise, starting with the most common estimate of all.

## Mean and spread of an average

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mover accent="true"><mi>X</mi><mo>ˉ</mo></mover><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>X</mi><mi>i</mi></msub><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">𝔼</mi><mover accent="true"><mi>X</mi><mo>ˉ</mo></mover><mo>=</mo><mi>μ</mi><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">Var</mi><mo>⁡</mo><mover accent="true"><mi>X</mi><mo>ˉ</mo></mover><mo>=</mo><mfrac><msup><mi>σ</mi><mn>2</mn></msup><mi>n</mi></mfrac></mrow></math></span></span>
Linearity gives the mean. Adding <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> independent variances and dividing by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup></mrow></math></span> gives the variance (lesson 10.19).

<div class="ex">
<div class="lab">Worked example</div>

**three samples, three estimates**

A fair die has mean 3.5. Roll it 5 times, three separate times: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>6</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> averages 3.8; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">)</mo></mrow></math></span> averages 3.4; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>6</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mn>6</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> averages 4.6.

Each is a reasonable guess; none is exactly 3.5.

</div>

```
rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=(100000, 25))
means = rolls.mean(axis=1)
print(round(means.mean(), 2), round(means.std(), 3), round(np.sqrt(35 / 12) / 5, 3))  # 3.5 0.342 0.342
```

~~~dg-raw
%%viz f_curve f=Math.exp(-(x-3.5)*(x-3.5)/(2*0.117))/Math.sqrt(2*Math.PI*0.117) x=2,5 y=0,1.25 tk=0.5 | Where the average of 25 die rolls lands. It centres on 3.5 with standard deviation 0.34, much narrower than a single roll.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

the sample mean is right on average, and its spread shrinks like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mi>n</mi></msqrt></mrow></math></span>.

</div>

## Where this breaks in practice

- Rows that are not independent (several per user) make the true spread larger than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mi mathvariant="normal">/</mi><msqrt><mi>n</mi></msqrt></mrow></math></span>.

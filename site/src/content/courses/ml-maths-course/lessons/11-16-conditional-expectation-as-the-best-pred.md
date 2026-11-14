---
id: "11.16"
course: "ml-maths-course"
module: "11"
slug: "conditional-expectation-as-the-best-predictor"
title: "Conditional Expectation as the Best Predictor"
keyIdeas: ["Among all functions ⟦90⟧, ⟦3719⟧ is smallest for ⟦3720⟧.", "Training with mean squared error estimates the conditional mean; absolute error estimates the conditional median.", "With no input, the best constant under MSE is ⟦3710⟧."]
question: "Data ⟦3718⟧. Compare MSE at the mean and the median, and absolute error at both."
answer: "Step 1, the mean is 4, MSE 12.5; the median is 2.5, MSE 14.75. Step 2, absolute error: 3 at the mean, 2.5 at the median. Each loss is minimised by its own centre."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

This is why the choice of loss matters: it decides which summary of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> your model learns.

## Which centre does each loss pick?

~~~dg-raw
%%viz f_curve f=((1-x)*(1-x)+(2-x)*(2-x)+(3-x)*(3-x)+(10-x)*(10-x))/4 g=(Math.abs(1-x)+Math.abs(2-x)+Math.abs(3-x)+Math.abs(10-x))/4 x=0,8 y=0,30 tk=1 ytk=5 fl=squared_error_(min_at_mean_4) gl=absolute_error_(min_at_median) | Loss of a constant guess c for data 1, 2, 3, 10. Squared error is lowest at the mean, 4; absolute error is lowest anywhere between 2 and 3, the median.
~~~

```
d = np.array([1.0, 2, 3, 10])
for c in (d.mean(), np.median(d)):
    print(c, ((d - c) ** 2).mean(), np.abs(d - c).mean())
# 4.0 12.5 3.0
# 2.5 14.75 2.5
```

<div class="take">
<div class="lab">Key takeaway</div>

MSE regression learns <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>Y</mi><mo>∣</mo><mi>X</mi><mo stretchy="false">]</mo></mrow></math></span>; absolute-error regression learns the conditional median.

</div>

## Where this breaks in practice

- With heavy-tailed targets the conditional mean is pulled by outliers; the median, or a quantile loss, may be the better target.

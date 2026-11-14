---
id: "10.37"
course: "ml-maths-course"
module: "10"
slug: "the-normal-distribution"
title: "The Normal Distribution"
keyIdeas: ["⟦3565⟧ has density ⟦3566⟧.", "About 68, 95 and 99.7% of the probability lies within 1, 2 and 3 standard deviations.", "Sums of many small independent effects are approximately normal (the central limit theorem, lesson 12.06)."]
question: "Test scores are ⟦3564⟧. What fraction lies between 50 and 90?"
answer: "Step 1, 50 and 90 are 2 standard deviations from the mean. Step 2, about 95%."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

The Gaussian is everywhere: measurement noise, weight initialisation, diffusion models, the latent space of a VAE. It is the one distribution to know by heart.

## Reading the formula

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span> is the centre, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi></mrow></math></span> the spread. The exponent makes the density fall off with the square of the distance from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mrow><mi>σ</mi><msqrt><mrow><mn>2</mn><mi>π</mi></mrow></msqrt></mrow></mfrac></mrow></math></span> makes the area 1 (lesson 09.51).

~~~dg-raw
%%viz f_curve f=Math.exp(-(x-70)*(x-70)/200)/(10*Math.sqrt(2*Math.PI)) x=30,110 y=0,0.045 tk=10 dom=50,90,cc | N(70, 10²). The shaded band from 50 to 90, two standard deviations each side, holds about 95% of the probability.
~~~

```
from scipy import stats
print([float(round(stats.norm.cdf(k) - stats.norm.cdf(-k), 4)) for k in (1, 2, 3)])  # [0.6827, 0.9545, 0.9973]
```

## Why it is everywhere

- Sums of many small independent effects are approximately Gaussian.
- Among distributions with a given mean and variance, it assumes the least (maximum entropy).
- Linear maps and conditionals of Gaussians are Gaussian (module 11).

<div class="take">
<div class="lab">Key takeaway</div>

the normal is set by its mean and standard deviation; 68-95-99.7 gives quick probabilities.

</div>

## Where this breaks in practice

- `np.random.normal(loc, scale)` and `torch.normal(mean, std)` take the standard deviation, while <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mi>μ</mi><mo separator="true">,</mo><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></math></span> is written with the variance. Mixing them up squares your noise level.

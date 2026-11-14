---
id: "10.18"
course: "ml-maths-course"
module: "10"
slug: "variance-and-standard-deviation"
title: "Variance and Standard Deviation"
keyIdeas: ["⟦3420⟧.", "The standard deviation ⟦3421⟧ is in the same units as ⟦513⟧.", "Two distributions with the same mean can have very different spreads."]
question: "⟦3393⟧ with probability 0.1, else 0. Find the variance and standard deviation."
answer: "Step 1, ⟦3425⟧ and ⟦3426⟧. Step 2, ⟦3427⟧, so ⟦3428⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

The second summary number is spread. The noise in a minibatch gradient, the uncertainty in an estimate and the risk of a strategy are all variances.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo stretchy="false">]</mo><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mo>−</mo><mo stretchy="false">(</mo><mi mathvariant="normal">𝔼</mi><mi>X</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span></span>
It is the average squared distance from the mean.

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI) g=Math.exp(-x*x/8)/(2*Math.sqrt(2*Math.PI)) x=-6,6 y=0,0.45 tk=2 fl=σ_=_1 gl=σ_=_2 | Two distributions with mean 0. Doubling σ makes the curve twice as wide and half as tall.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the die's spread**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>=</mo><mn>91</mn><mi mathvariant="normal">/</mi><mn>6</mn><mo>−</mo><mn>3.</mn><msup><mn>5</mn><mn>2</mn></msup><mo>≈</mo><mn>2.92</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo>≈</mo><mn>1.71</mn></mrow></math></span>.

</div>

```
x = np.arange(1, 7)
print(round(x.var(), 2), round(x.std(), 2))  # 2.92 1.71
```

<div class="take">
<div class="lab">Key takeaway</div>

variance is the mean squared distance from the mean; its square root is in the original units.

</div>

## Where this breaks in practice

- `np.var` divides by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> by default; `pandas` divides by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></math></span>. For small samples the difference matters (12.04).

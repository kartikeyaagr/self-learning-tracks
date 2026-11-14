---
id: "11.28"
course: "ml-maths-course"
module: "11"
slug: "conditioning-a-gaussian"
title: "Conditioning a Gaussian"
keyIdeas: ["⟦3818⟧ is Gaussian with mean ⟦3819⟧ and covariance ⟦3820⟧.", "The mean shifts in proportion to how surprising the observation was; the covariance shrinks by an amount that doesn't depend on the value seen.", "Gaussian process regression is exactly this conditioning."]
question: "⟦3602⟧ has means 0, variances 1 and correlation 0.8. Find the distribution of ⟦512⟧ given ⟦3617⟧."
answer: "Step 1, mean ⟦3825⟧. Step 2, variance ⟦3826⟧, standard deviation 0.6."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Gaussian combinations"]
tools: ["NumPy"]
topic: {"id": "11.05", "title": "Gaussian combinations"}
---

## The big picture

Observing some coordinates of a Gaussian vector tells you about the rest. The answer is again Gaussian, with a formula.

## The block formulas

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>μ</mi><mrow><mn>1</mn><mo>∣</mo><mn>2</mn></mrow></msub><mo>=</mo><msub><mi>μ</mi><mn>1</mn></msub><mo>+</mo><msub><mi mathvariant="normal">Σ</mi><mn>12</mn></msub><msubsup><mi mathvariant="normal">Σ</mi><mn>22</mn><mrow><mo>−</mo><mn>1</mn></mrow></msubsup><mo stretchy="false">(</mo><msub><mi>x</mi><mn>2</mn></msub><mo>−</mo><msub><mi>μ</mi><mn>2</mn></msub><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><msub><mi mathvariant="normal">Σ</mi><mrow><mn>1</mn><mo>∣</mo><mn>2</mn></mrow></msub><mo>=</mo><msub><mi mathvariant="normal">Σ</mi><mn>11</mn></msub><mo>−</mo><msub><mi mathvariant="normal">Σ</mi><mn>12</mn></msub><msubsup><mi mathvariant="normal">Σ</mi><mn>22</mn><mrow><mo>−</mo><mn>1</mn></mrow></msubsup><msub><mi mathvariant="normal">Σ</mi><mn>21</mn></msub></mrow></math></span></span>
In one dimension the mean formula is the regression line with slope <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ρ</mi><mtext> </mtext><msub><mi>σ</mi><mn>1</mn></msub><mi mathvariant="normal">/</mi><msub><mi>σ</mi><mn>2</mn></msub></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI) g=Math.exp(-(x-0.8)*(x-0.8)/0.72)/(0.6*Math.sqrt(2*Math.PI)) x=-3.5,3.5 y=0,0.72 tk=1 fl=Y_before_seeing_X gl=Y_given_X_=_1 | With correlation 0.8, seeing X = 1 moves Y's distribution to mean 0.8 and narrows it to standard deviation 0.6.
~~~

```
rng = np.random.default_rng(0)
S = np.array([[1.0, 0.8], [0.8, 1]])
xy = rng.multivariate_normal([0, 0], S, size=2_000_000)
near = np.abs(xy[:, 0] - 1) < 0.01
print(round(xy[near, 1].mean(), 1), round(xy[near, 1].std(), 1))  # 0.8 0.6
```

<div class="ex">
<div class="lab">Worked example</div>

**regression to the mean**

Father's and son's heights: mean 175 cm, standard deviation 7, correlation 0.5. A 189 cm father predicts a son of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>175</mn><mo>+</mo><mn>0.5</mn><mo>×</mo><mn>14</mn><mo>=</mo><mn>182</mn></mrow></math></span> cm.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

conditioning a Gaussian shifts the mean toward the evidence and shrinks the covariance.

</div>

## Where this breaks in practice

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi mathvariant="normal">Σ</mi><mn>22</mn><mrow><mo>−</mo><mn>1</mn></mrow></msubsup></mrow></math></span> for many observations is expensive and badly conditioned. Solve with a Cholesky factor and add jitter.

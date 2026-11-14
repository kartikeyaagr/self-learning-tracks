---
id: "10.12"
course: "ml-maths-course"
module: "10"
slug: "linear-transformations-and-standardising"
title: "Linear Transformations and Standardising"
keyIdeas: ["For ⟦3359⟧: ⟦3360⟧, ⟦3361⟧, ⟦3362⟧.", "Standardising, ⟦3363⟧, gives mean 0 and variance 1.", "A linear map of a Gaussian is Gaussian: ⟦3364⟧."]
question: "⟦3357⟧. What is ⟦3358⟧?"
answer: "Step 1, ⟦3367⟧. Step 2, ⟦3368⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Transforming random variables"]
tools: ["NumPy", "SciPy"]
topic: {"id": "10.03", "title": "Transforming random variables"}
---

## The big picture

Feature scaling, batch normalisation and z-scores are all the same linear transformation. Its effect on the distribution is simple and worth knowing exactly.

## The rule

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mi mathvariant="normal">∣</mi><mi>a</mi><mi mathvariant="normal">∣</mi></mrow></mfrac><mtext> </mtext><msub><mi>p</mi><mi>X</mi></msub><mtext> ⁣</mtext><mrow><mo fence="true">(</mo><mfrac><mrow><mi>y</mi><mo>−</mo><mi>b</mi></mrow><mi>a</mi></mfrac><mo fence="true">)</mo></mrow></mrow></math></span></span>
Shifting by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> slides the density; scaling by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> stretches it by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>a</mi><mi mathvariant="normal">∣</mi></mrow></math></span> and lowers it by the same factor.

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI) g=Math.exp(-(x-2)*(x-2)/8)/(2*Math.sqrt(2*Math.PI)) x=-4,8 y=0,0.45 tk=2 fl=Z_∼_N(0~_1) gl=2Z_+_2_∼_N(2~_4) | Scaling by 2 makes the curve twice as wide and half as tall; adding 2 moves its centre.
~~~

```
from scipy import stats
x = np.array([60.0, 70, 90])
z = (x - 70) / 10
print(z, round(stats.norm.sf(2), 4))  # [-1.  0.  2.] 0.0228
```

<div class="take">
<div class="lab">Key takeaway</div>

a linear map shifts and stretches a distribution; standardising undoes it.

</div>

## Where this breaks in practice

- Standardise test data with the training set's mean and standard deviation, not its own. Otherwise information from the test set leaks in.

---
id: "12.12"
course: "ml-maths-course"
module: "12"
slug: "maximum-likelihood-for-the-gaussian"
title: "Maximum Likelihood for the Gaussian"
keyIdeas: ["For ⟦3565⟧: ⟦3908⟧ and ⟦3909⟧.", "The MLE of the variance divides by ⟦233⟧, so it is slightly biased (lesson 12.04).", "With ⟦1239⟧ fixed, maximising the Gaussian likelihood is minimising squared error."]
question: "Data ⟦3854⟧. Give the Gaussian MLEs."
answer: "Step 1, ⟦3913⟧. Step 2, ⟦3914⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["NumPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

The Gaussian MLE explains two everyday facts: why the sample mean is the natural estimate, and why squared error is the natural loss.

## The derivation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">ℓ</mi><mo>=</mo><mo>−</mo><mfrac><mi>n</mi><mn>2</mn></mfrac><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>2</mn><mi>π</mi><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">)</mo><mo>−</mo><mfrac><mn>1</mn><mrow><mn>2</mn><msup><mi>σ</mi><mn>2</mn></msup></mrow></mfrac><munder><mo>∑</mo><mi>i</mi></munder><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span></span>
Setting <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi mathvariant="normal">ℓ</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>μ</mi><mo>=</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>μ</mi><mo>^</mo></mover><mo>=</mo><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover></mrow></math></span>; setting <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi mathvariant="normal">ℓ</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><msup><mi>σ</mi><mn>2</mn></msup><mo>=</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mover accent="true"><mi>σ</mi><mo>^</mo></mover><mn>2</mn></msup><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><mo>∑</mo><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span>.

```
from scipy import stats
x = np.array([2.0, 4, 6])
mu, sd = stats.norm.fit(x)
print(round(mu, 3), round(sd ** 2, 3))  # 4.0 2.667
```

~~~dg-raw
%%viz f_curve f=Math.exp(-(x-4)*(x-4)/(2*2.667))/Math.sqrt(2*Math.PI*2.667) x=-1,9 y=0,0.3 tk=1 pts=2;4;6 | The fitted Gaussian for the data 2, 4, 6 (dots on the curve). It is centred on the sample mean with variance 8/3.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

the Gaussian MLE is the sample mean and the divide-by-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> variance.

</div>

## Where this breaks in practice

- Outliers drag both estimates; a Student-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> likelihood is more robust.

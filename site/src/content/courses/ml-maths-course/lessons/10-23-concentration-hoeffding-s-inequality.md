---
id: "10.23"
course: "ml-maths-course"
module: "10"
slug: "concentration-hoeffding-s-inequality"
title: "Concentration: Hoeffding's Inequality"
keyIdeas: ["For ⟦233⟧ independent values in ⟦294⟧, the sample mean is within ⟦166⟧ of the true mean with probability at least ⟦3461⟧.", "The failure probability falls exponentially in ⟦233⟧, much faster than Chebyshev's ⟦365⟧.", "It gives the sample size needed to estimate an accuracy to a given precision."]
question: "How many test examples guarantee, with probability 95%, that measured accuracy is within 0.03 of the truth?"
answer: "Step 1, set ⟦3465⟧. Step 2, ⟦3466⟧, so 2,050 examples."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

How big must a test set be before you trust its accuracy? Hoeffding answers with no assumption beyond independence and bounded values.

## The bound

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mrow><mo fence="true">(</mo><mo stretchy="false">∣</mo><mover accent="true"><mi>X</mi><mo>ˉ</mo></mover><mo>−</mo><mi>μ</mi><mo stretchy="false">∣</mo><mo>≥</mo><mi>t</mi><mo fence="true">)</mo></mrow><mo>≤</mo><mn>2</mn><mi>exp</mi><mo>⁡</mo><mrow><mo fence="true">(</mo><mo>−</mo><mfrac><mrow><mn>2</mn><mi>n</mi><msup><mi>t</mi><mn>2</mn></msup></mrow><mrow><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>a</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></mfrac><mo fence="true">)</mo></mrow></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**a test set of 10,000**

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>0.02</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msup><mi>e</mi><mrow><mo>−</mo><mn>2</mn><mo>⋅</mo><mn>10000</mn><mo>⋅</mo><mn>0.0004</mn></mrow></msup><mo>=</mo><mn>2</mn><msup><mi>e</mi><mrow><mo>−</mo><mn>8</mn></mrow></msup><mo>≈</mo><mn>0.0007</mn></mrow></math></span>. The test accuracy is within 2 points of the truth with probability 99.9%.

</div>

~~~dg-raw
%%viz f_curve f=2*Math.exp(-2*x*0.0009) x=100,4000 y=0,1 tk=1000 ytk=0.25 pts=2050,n_=_2050:_0.05 | Hoeffding's bound on the chance that accuracy is off by 0.03 or more. It falls exponentially and reaches 5% at about 2,050 examples.
~~~

```
n = np.log(2 / 0.05) / (2 * 0.03 ** 2)
print(int(np.ceil(n)))  # 2050
```

<div class="take">
<div class="lab">Key takeaway</div>

averages of bounded independent values concentrate exponentially fast.

</div>

## Where this breaks in practice

- If you tune on the test set, its examples no longer act as independent fresh draws and the bound no longer applies.

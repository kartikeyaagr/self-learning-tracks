---
id: "12.15"
course: "ml-maths-course"
module: "12"
slug: "map-estimation-and-regularisation"
title: "MAP Estimation and Regularisation"
keyIdeas: ["MAP maximises ⟦3924⟧: the likelihood plus a log-prior.", "A Gaussian prior on weights gives L2 regularisation (weight decay); a Laplace prior gives L1.", "With a flat prior MAP equals MLE."]
question: "Beta(2, 2) prior, 3 heads in 3 flips. Compare the MLE and the MAP estimate."
answer: "Step 1, MLE ⟦3930⟧. Step 2, the posterior is Beta(5, 2) with peak ⟦3931⟧."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Likelihood-based estimation"]
tools: ["NumPy"]
topic: {"id": "12.02", "title": "Likelihood-based estimation"}
---

## The big picture

Regularisation has a probabilistic reading: it is a prior belief about the parameters, added to the likelihood.

## The log-prior as a penalty

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mover accent="true"><mi>θ</mi><mo>^</mo></mover><mtext>MAP</mtext></msub><mo>=</mo><mi>arg</mi><mo>⁡</mo><munder><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>θ</mi></munder><mtext> </mtext><mo stretchy="false">[</mo><mi mathvariant="normal">ℓ</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo>+</mo><mi>log</mi><mo>⁡</mo><mi>p</mi><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo></mrow></math></span></span>

| Prior on weights | Penalty added to the loss |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><msup><mi>τ</mi><mn>2</mn></msup><mi>I</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mrow><mn>2</mn><msup><mi>τ</mi><mn>2</mn></msup></mrow></mfrac><mi mathvariant="normal">∥</mi><mi>w</mi><msubsup><mi mathvariant="normal">∥</mi><mn>2</mn><mn>2</mn></msubsup></mrow></math></span> |
| Laplace<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mi>b</mi></mfrac><mi mathvariant="normal">∥</mi><mi>w</mi><msub><mi mathvariant="normal">∥</mi><mn>1</mn></msub></mrow></math></span> |
| flat | none |

~~~dg-raw
%%viz f_curve f=x*x*x/1 g=x*x*x*x*(1-x)/0.08192 x=0,1 y=0,1.05 tk=0.25 fl=likelihood gl=posterior | Three heads in three flips. The likelihood alone peaks at p = 1; the prior pulls the posterior's peak back to 0.8. Both curves are scaled to peak at 1.
~~~

```
p = np.linspace(0, 1, 10001)
print(p[(p ** 3).argmax()], p[(p ** 4 * (1 - p)).argmax()])  # 1.0 0.8
```

<div class="take">
<div class="lab">Key takeaway</div>

MAP = MLE plus a penalty from the prior; weight decay is a Gaussian prior.

</div>

## Where this breaks in practice

- MAP gives a single point, not the posterior's spread; it says nothing about uncertainty.

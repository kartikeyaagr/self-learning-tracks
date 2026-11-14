---
id: "11.24"
course: "ml-maths-course"
module: "11"
slug: "correlated-errors-and-ensembles"
title: "Correlated Errors and Ensembles"
keyIdeas: ["Averaging ⟦607⟧ models with error variance ⟦3780⟧ and pairwise correlation ⟦3767⟧ gives variance ⟦3781⟧.", "As ⟦607⟧ grows, the floor ⟦3782⟧ remains: correlated errors limit what averaging can do.", "Ensembles help most when members make different mistakes."]
question: "Two models with error variance 1. What is the variance of their average for ⟦3778⟧, ⟦3779⟧ and ⟦134⟧?"
answer: "Step 1, ⟦3784⟧. Step 2, that gives ⟦329⟧, ⟦1687⟧ and ⟦134⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Covariance and correlation"]
tools: ["NumPy"]
topic: {"id": "11.04", "title": "Covariance and correlation"}
---

## The big picture

Ensembling is averaging. How much it helps depends entirely on how correlated the members' errors are.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mtext>average of </mtext><mi>M</mi><mo stretchy="false">)</mo><mo>=</mo><mi>ρ</mi><msup><mi>σ</mi><mn>2</mn></msup><mo>+</mo><mfrac><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>ρ</mi><mo stretchy="false">)</mo><msup><mi>σ</mi><mn>2</mn></msup></mrow><mi>M</mi></mfrac></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=1/x g=0.8+0.2/x x=1,20 y=0,1.05 tk=5 ytk=0.25 fl=ρ_=_0 gl=ρ_=_0.8 | Error variance of an ensemble of M models. Uncorrelated errors keep falling; with correlation 0.8 the variance never drops below 0.8.
~~~

```
rng = np.random.default_rng(0)
for rho in (0.0, 0.8):
    C = np.full((5, 5), rho) + (1 - rho) * np.eye(5)
    E = rng.multivariate_normal(np.zeros(5), C, size=200000)
    print(rho, round(E.mean(1).var(), 2), round(rho + (1 - rho) / 5, 2))
# 0.0 0.2 0.2
# 0.8 0.84 0.84
```

<div class="take">
<div class="lab">Key takeaway</div>

averaging removes only the uncorrelated part of the errors.

</div>

## Where this breaks in practice

- Models trained on the same data with the same architecture make highly correlated errors. Vary data, features or architectures.

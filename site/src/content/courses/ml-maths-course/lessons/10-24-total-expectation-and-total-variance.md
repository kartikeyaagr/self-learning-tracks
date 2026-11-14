---
id: "10.24"
course: "ml-maths-course"
module: "10"
slug: "total-expectation-and-total-variance"
title: "Total Expectation and Total Variance"
keyIdeas: ["Total expectation: ⟦3467⟧. Average within groups, then average the group averages.", "Total variance: ⟦3468⟧, within plus between.", "In ensembles and Bayesian deep learning this splits aleatoric from epistemic uncertainty."]
question: "Two equal groups: means 160 and 180, each with variance 25. What is the total variance?"
answer: "Step 1, within: 25. Step 2, between: the means 160 and 180 have variance ⟦3470⟧. Step 3, total ⟦3471⟧, standard deviation about 11.2."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

Data often come in groups. Splitting spread into "within groups" and "between groups" tells you where the uncertainty comes from.

## The two laws

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mi>X</mi><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>X</mi><mo>∣</mo><mi>Y</mi><mo stretchy="false">]</mo><mo stretchy="false">]</mo><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>=</mo><munder><munder><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>X</mi><mo>∣</mo><mi>Y</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo></mrow><mo stretchy="true">⏟</mo></munder><mtext>within</mtext></munder><mo>+</mo><munder><munder><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>X</mi><mo>∣</mo><mi>Y</mi><mo stretchy="false">]</mo><mo stretchy="false">)</mo></mrow><mo stretchy="true">⏟</mo></munder><mtext>between</mtext></munder></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=Math.exp(-(x-160)*(x-160)/50)/(5*Math.sqrt(2*Math.PI)) g=Math.exp(-(x-180)*(x-180)/50)/(5*Math.sqrt(2*Math.PI)) x=140,200 y=0,0.09 tk=10 fl=group_A gl=group_B | Two groups, each with standard deviation 5. Most of the total spread comes from the gap between the group means.
~~~

```
rng = np.random.default_rng(0)
a = rng.normal(160, 5, 500_000); b = rng.normal(180, 5, 500_000)
print(round(np.concatenate([a, b]).var()))  # 125
```

<div class="take">
<div class="lab">Key takeaway</div>

total variance = average variance within groups + variance of the group means.

</div>

## Where this breaks in practice

- The split needs the right grouping variable. Group by the wrong one and "between" variance looks small even when a hidden factor drives the spread.

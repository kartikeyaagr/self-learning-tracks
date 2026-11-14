---
id: "10.17"
course: "ml-maths-course"
module: "10"
slug: "expectations-of-functions-lotus"
title: "Expectations of Functions (LOTUS)"
keyIdeas: ["LOTUS: ⟦3411⟧ or ⟦3412⟧. You do not need the distribution of ⟦3014⟧.", "In general ⟦3413⟧.", "Expected loss ⟦3414⟧ is computed this way."]
question: "For a fair die, compute ⟦3013⟧ and compare it with ⟦3410⟧."
answer: "Step 1, ⟦3418⟧. Step 2, ⟦3419⟧. They differ."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

You rarely want the mean of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> itself. You want the mean of a loss, a reward, or a square. LOTUS (the "law of the unconscious statistician") says: apply <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi></mrow></math></span> to each value and average with the same weights.

## The rule

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>g</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo><mo>=</mo><munder><mo>∑</mo><mi>x</mi></munder><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span></span>

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></math></span> | 1 | 4 | 9 | 16 | 25 | 36 |
| weight | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

```
x = np.arange(1, 7); p = np.full(6, 1 / 6)
print(round((x ** 2 * p).sum(), 2), round((x * p).sum() ** 2, 2))  # 15.17 12.25
```

<div class="take">
<div class="lab">Key takeaway</div>

to average <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo></mrow></math></span>, average <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>'s probabilities.

</div>

## Where this breaks in practice

- Plugging the mean into a nonlinear function, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi mathvariant="normal">𝔼</mi><mi>X</mi><mo stretchy="false">)</mo></mrow></math></span>, is a common shortcut and a common bias. Jensen's inequality (10.21) says which way it errs.

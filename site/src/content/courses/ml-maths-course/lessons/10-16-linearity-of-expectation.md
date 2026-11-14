---
id: "10.16"
course: "ml-maths-course"
module: "10"
slug: "linearity-of-expectation"
title: "Linearity of Expectation"
keyIdeas: ["⟦3402⟧, always, even when ⟦513⟧ and ⟦512⟧ are dependent.", "Counting problems become easy: write a count as a sum of indicators and add their probabilities.", "⟦3403⟧ needs more: it holds when ⟦513⟧ and ⟦512⟧ are uncorrelated."]
question: "A classifier gets each of 100 test examples right with probabilities ⟦3401⟧, possibly correlated. What is the expected number correct?"
answer: "Step 1, write the count as ⟦3407⟧. Step 2, each indicator has mean ⟦3408⟧. Step 3, by linearity the expected count is ⟦3409⟧, whatever the correlation."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

Linearity is the most-used property in probability. It needs no independence, which is why it works on messy, correlated data.

## The rule

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>a</mi><mi>X</mi><mo>+</mo><mi>b</mi><mi>Y</mi><mo stretchy="false">]</mo><mo>=</mo><mi>a</mi><mtext> </mtext><mi mathvariant="normal">𝔼</mi><mi>X</mi><mo>+</mo><mi>b</mi><mtext> </mtext><mi mathvariant="normal">𝔼</mi><mi>Y</mi></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**matching hats**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> people get their hats back in random order. How many get their own hat, on average?

Person <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> gets their own hat with probability <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span>. The indicators are dependent, but linearity does not care: the expected count is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>⋅</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><mo>=</mo><mn>1</mn></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
n = 20
hits = [(rng.permutation(n) == np.arange(n)).sum() for _ in range(100000)]
print(round(np.mean(hits), 2))  # 1.0
```

<div class="take">
<div class="lab">Key takeaway</div>

the mean of a sum is the sum of the means, with no independence needed.

</div>

## Where this breaks in practice

- Linearity does not extend to products or nonlinear functions: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mo mathvariant="normal">≠</mo><mo stretchy="false">(</mo><mi mathvariant="normal">𝔼</mi><mi>X</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span> in general.

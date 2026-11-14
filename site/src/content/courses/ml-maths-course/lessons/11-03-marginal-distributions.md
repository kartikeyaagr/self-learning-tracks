---
id: "11.03"
course: "ml-maths-course"
module: "11"
slug: "marginal-distributions"
title: "Marginal Distributions"
keyIdeas: ["The marginal of ⟦513⟧ sums out ⟦512⟧: ⟦3622⟧, the row sums.", "The marginal of ⟦512⟧ is the column sums.", "Marginals answer questions about one variable while ignoring the other."]
question: "For the table ⟦3621⟧, find both marginals."
answer: "Step 1, rows: ⟦3627⟧. Step 2, columns: ⟦3628⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two discrete variables"]
tools: ["NumPy", "pandas"]
topic: {"id": "11.01", "title": "Two discrete variables"}
---

## The big picture

Often you have the joint but care about one variable. Marginalising adds up over everything you don't care about.

## Summing out

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mo>∑</mo><mi>y</mi></munder><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span></span>
In the table, these are the row sums written in the margin, hence the name.

| | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mn>0</mn></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mn>1</mn></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>X</mi></msub></mrow></math></span> |
| --- | --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mn>0</mn></mrow></math></span> | 0.4 | 0.2 | 0.6 |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mn>1</mn></mrow></math></span> | 0.1 | 0.3 | 0.4 |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>Y</mi></msub></mrow></math></span> | 0.5 | 0.5 | 1 |

```
P = np.array([[0.4, 0.2], [0.1, 0.3]])
print(P.sum(axis=1), P.sum(axis=0))  # [0.6 0.4] [0.5 0.5]
```

<div class="ex">
<div class="lab">Worked example</div>

**marginalising a latent variable**

A mixture model has a hidden cluster <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> and data <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>. The model gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>z</mi><mo stretchy="false">)</mo></mrow></math></span>. The probability of the data alone is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mo>∑</mo><mi>z</mi></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>z</mi><mo stretchy="false">)</mo></mrow></math></span>, a marginal.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

to get one variable's distribution, add the joint over the other.

</div>

## Where this breaks in practice

- With many hidden variables the sum has too many terms. That is why latent-variable models need EM or variational inference.

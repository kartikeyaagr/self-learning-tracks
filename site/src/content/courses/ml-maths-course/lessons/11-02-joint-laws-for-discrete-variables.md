---
id: "11.02"
course: "ml-maths-course"
module: "11"
slug: "joint-laws-for-discrete-variables"
title: "Joint Laws for Discrete Variables"
keyIdeas: ["A joint PMF ⟦3612⟧ is a table whose entries are non-negative and add up to 1.", "Any question about the pair is a sum over cells: ⟦3613⟧.", "With ⟦3520⟧ variables of ⟦1204⟧ values each, the table has ⟦3614⟧ cells, which is why models factorise it."]
question: "With ⟦3606⟧, ⟦3607⟧, ⟦3608⟧, ⟦3609⟧, find ⟦3610⟧ and ⟦3611⟧."
answer: "Step 1, ⟦3619⟧. Step 2, ⟦3620⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two discrete variables"]
tools: ["NumPy"]
topic: {"id": "11.01", "title": "Two discrete variables"}
---

## The big picture

For discrete variables the joint distribution is just a table. Every probability about the pair is read off by adding cells.

## The table

| | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mn>0</mn></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mn>1</mn></mrow></math></span> |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mn>0</mn></mrow></math></span> | 0.4 | 0.2 |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mn>1</mn></mrow></math></span> | 0.1 | 0.3 |

~~~dg-raw
%%viz f_bars v=0.4,0.2,0.1,0.3 l=(0~0),(0~1),(1~0),(1~1) hi=none yl=p(x~_y) | The four cells of the joint table. They add up to 1.
~~~

```
P = np.array([[0.4, 0.2], [0.1, 0.3]])
print(P.sum(), P[0, 0] + P[1, 1], round(1 - P[0, 0], 2))  # 1.0 0.7 0.6
```

## Too many cells

Ten binary variables give 1024 cells; fifty give about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>15</mn></msup></mrow></math></span>. Models instead write the joint as a product of smaller conditionals (lesson 11.04), which is what Bayesian networks and autoregressive models do.

<div class="take">
<div class="lab">Key takeaway</div>

a discrete joint distribution is a table that adds up to 1; questions are sums of cells.

</div>

## Where this breaks in practice

- Tables estimated from data have zero cells for unseen combinations. Add-one smoothing prevents zero probabilities and infinite log-losses.

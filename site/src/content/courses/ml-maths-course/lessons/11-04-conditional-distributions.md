---
id: "11.04"
course: "ml-maths-course"
module: "11"
slug: "conditional-distributions"
title: "Conditional Distributions"
keyIdeas: ["The conditional ⟦3631⟧: take the row for ⟦7⟧ and rescale it to add up to 1.", "The chain rule rebuilds the joint: ⟦3632⟧.", "⟦3633⟧ and ⟦3634⟧ are different distributions from the same table."]
question: "For the table ⟦3621⟧, find ⟦3629⟧ and ⟦3630⟧."
answer: "Step 1, row ⟦3617⟧ is ⟦3638⟧ with sum 0.4, so ⟦3639⟧. Step 2, column ⟦3616⟧ is ⟦3640⟧ with sum 0.5, so ⟦3641⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two discrete variables"]
tools: ["NumPy"]
topic: {"id": "11.01", "title": "Two discrete variables"}
---

## The big picture

"Now that I know <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>, what about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span>?" is the question every predictive model answers. The conditional is one slice of the joint, rescaled.

## Slicing and rescaling

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span></span>

~~~dg-flow Three things you do to a joint table.
Joint p(x, y) -(sum over y)-> marginal p(x)
Joint p(x, y) -(take the row x, divide by its sum)-> *conditional p(y given x)
Marginal p(x) and conditional p(y given x) -(multiply)-> joint p(x, y)
~~~

```
P = np.array([[0.4, 0.2], [0.1, 0.3]])
pX = P.sum(axis=1)
cond = P / pX[:, None]
print(np.round(cond, 3).tolist())  # [[0.667, 0.333], [0.25, 0.75]]
print(np.allclose(pX[:, None] * cond, P))  # True
```

<div class="ex">
<div class="lab">Worked example</div>

**weather and umbrellas**

Joint: (rain, yes) 0.18, (rain, no) 0.02, (dry, yes) 0.08, (dry, no) 0.72.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>umbrella</mtext><mo>∣</mo><mtext>rain</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.18</mn><mi mathvariant="normal">/</mi><mn>0.20</mn><mo>=</mo><mn>0.9</mn></mrow></math></span>, but <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>rain</mtext><mo>∣</mo><mtext>umbrella</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.18</mn><mi mathvariant="normal">/</mi><mn>0.26</mn><mo>≈</mo><mn>0.69</mn></mrow></math></span>.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a conditional is a row (or column) of the joint, rescaled to add up to 1.

</div>

## Where this breaks in practice

- Conditioning on a value seen only a few times gives a noisy estimate. Smooth or pool rare categories.

---
id: "11.13"
course: "ml-maths-course"
module: "11"
slug: "expectations-of-functions-of-two-variables"
title: "Expectations of Functions of Two Variables"
keyIdeas: ["⟦3690⟧ or ⟦3691⟧.", "⟦3692⟧ always.", "⟦3693⟧; the covariance term vanishes for independent variables."]
question: "For the table ⟦3621⟧, find ⟦3688⟧ and compare it with ⟦3689⟧."
answer: "Step 1, only cell ⟦111⟧ has ⟦3695⟧, so ⟦3696⟧. Step 2, ⟦3697⟧. The gap, 0.1, is the covariance."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

LOTUS (lesson 10.17) works for pairs: weight each value of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi></mrow></math></span> by the probability of its cell.

## Weighted averages over cells

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>g</mi><mo stretchy="false">(</mo><mi>X</mi><mo separator="true">,</mo><mi>Y</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo><mo>=</mo><munder><mo>∑</mo><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi></mrow></munder><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span></span>

```
P = np.array([[0.4, 0.2], [0.1, 0.3]])
x = np.array([0, 1])[:, None]; y = np.array([0, 1])[None, :]
EXY = (x * y * P).sum(); EX = (x * P).sum(); EY = (y * P).sum()
print(round(EXY, 2), round(EX * EY, 2), round(EXY - EX * EY, 2))  # 0.3 0.2 0.1
```

<div class="take">
<div class="lab">Key takeaway</div>

average a function of two variables against their joint distribution; products only split under independence.

</div>

## Where this breaks in practice

- Treating <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>X</mi><mi>Y</mi><mo stretchy="false">]</mo></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mi>X</mi><mtext> </mtext><mi mathvariant="normal">𝔼</mi><mi>Y</mi></mrow></math></span> silently assumes no correlation.

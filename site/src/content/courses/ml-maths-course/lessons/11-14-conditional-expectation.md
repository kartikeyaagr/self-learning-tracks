---
id: "11.14"
course: "ml-maths-course"
module: "11"
slug: "conditional-expectation"
title: "Conditional Expectation"
keyIdeas: ["⟦3700⟧ is the mean of the slice at ⟦7⟧: one number per ⟦7⟧.", "⟦3701⟧, with ⟦513⟧ left random, is itself a random variable.", "For the uniform triangle, ⟦3702⟧."]
question: "For the table ⟦3621⟧, find ⟦3698⟧ and ⟦3699⟧."
answer: "Step 1, row 0 rescaled is ⟦3705⟧, mean ⟦2954⟧. Step 2, row 1 rescaled is ⟦3706⟧, mean ⟦3707⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

"On average, what is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> is this?" is what a regression curve shows. That curve is the conditional expectation.

## The mean of each slice

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>Y</mi><mo>∣</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo stretchy="false">]</mo><mo>=</mo><mo>∫</mo><mi>y</mi><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-0.2,1.3 y=-0.2,1.3 tk=1 pg=0:0_1:0_1:1,a ln=0.5,-1,0,E[Y∣X_=_x]_=_x/2,f,0.85 | In the uniform triangle, the mean of each vertical slice lies on the line y = x/2 (orange).
~~~

```
P = np.array([[0.4, 0.2], [0.1, 0.3]])
cond = P / P.sum(1, keepdims=True)
print(np.round(cond @ np.array([0, 1]), 3))  # [0.333 0.75 ]
```

<div class="take">
<div class="lab">Key takeaway</div>

conditional expectation is the average of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> within each slice of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- The conditional mean hides spread. Two inputs can share <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>Y</mi><mo>∣</mo><mi>X</mi><mo stretchy="false">]</mo></mrow></math></span> with very different uncertainty.

---
id: "11.15"
course: "ml-maths-course"
module: "11"
slug: "the-tower-property"
title: "The Tower Property"
keyIdeas: ["⟦3711⟧: average the group means, weighted by group size, to get the overall mean.", "It is the continuous version of the law of total expectation (lesson 10.24).", "It lets you compute a hard expectation in two easy stages."]
question: "In the uniform triangle, ⟦3708⟧ and ⟦3709⟧. Find ⟦3710⟧."
answer: "Step 1, ⟦3715⟧. Step 2, ⟦3716⟧. Step 3, ⟦3717⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

Condition on something that makes the problem easy, solve it, then average over what you conditioned on.

## Two stages

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mi>Y</mi><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">[</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>Y</mi><mo>∣</mo><mi>X</mi><mo stretchy="false">]</mo><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">]</mo></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**average class score**

Class A (30 students) averages 70; class B (10 students) averages 90. The overall mean is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.75</mn><mo>×</mo><mn>70</mn><mo>+</mo><mn>0.25</mn><mo>×</mo><mn>90</mn><mo>=</mo><mn>75</mn></mrow></math></span>, not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>70</mn><mo>+</mo><mn>90</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn><mo>=</mo><mn>80</mn></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
u = rng.random((400000, 2))
x, y = u.max(1), u.min(1)
print(round((x / 2).mean(), 2), round(y.mean(), 2))  # 0.33 0.33
```

<div class="take">
<div class="lab">Key takeaway</div>

the mean of the conditional means is the overall mean.

</div>

## Where this breaks in practice

- Averaging group means without weighting by group size gives the wrong overall mean.

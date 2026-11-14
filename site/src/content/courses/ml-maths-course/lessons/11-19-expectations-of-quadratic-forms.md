---
id: "11.19"
course: "ml-maths-course"
module: "11"
slug: "expectations-of-quadratic-forms"
title: "Expectations of Quadratic Forms"
keyIdeas: ["⟦3742⟧.", "With ⟦3743⟧: ⟦3744⟧.", "For ⟦3745⟧, ⟦3746⟧, so a typical vector has length about ⟦1890⟧."]
question: "⟦3740⟧. What are ⟦3741⟧ and the typical length?"
answer: "Step 1, ⟦3750⟧. Step 2, typical length about ⟦3751⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

Squared norms, Mahalanobis distances and losses like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>W</mi><mi>x</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> are quadratic forms. Their expectation has a simple formula.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>M</mi><mi>X</mi><mo stretchy="false">]</mo><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>M</mi><mi mathvariant="normal">Σ</mi><mo stretchy="false">)</mo><mo>+</mo><msup><mi>μ</mi><mi mathvariant="normal">⊤</mi></msup><mi>M</mi><mi>μ</mi></mrow></math></span></span>

```
rng = np.random.default_rng(0)
for d in (10, 100, 1000):
    print(d, round(np.linalg.norm(rng.normal(size=(20000, d)), axis=1).mean(), 1))
# 10 3.1
# 100 10.0
# 1000 31.6
```

~~~dg-raw
%%viz f_curve f=Math.sqrt(x) x=1,1000 y=0,35 tk=200 pts=100,d_=_100:_10;1000,d_=_1000:_31.6 | The typical length of a standard Gaussian vector grows like √d. In high dimensions almost no samples are near the origin, even though the density peaks there.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>M</mi><mi>X</mi><mo stretchy="false">]</mo></mrow></math></span> is a trace plus the mean's own quadratic form.

</div>

## Where this breaks in practice

- Intuition from 2D fails in high dimensions: a Gaussian's samples live on a thin shell, not near the centre.

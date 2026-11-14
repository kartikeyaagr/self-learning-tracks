---
id: "10.35"
course: "ml-maths-course"
module: "10"
slug: "the-uniform-distribution"
title: "The Uniform Distribution"
keyIdeas: ["Uniform(⟦604⟧) has constant density ⟦3549⟧ on ⟦294⟧.", "Mean ⟦3550⟧, variance ⟦3551⟧.", "Uniform on ⟦3546⟧ has variance ⟦3552⟧, which sets the range of uniform Xavier and He initialisation."]
question: "What range ⟦3546⟧ gives variance ⟦3547⟧ (He initialisation, ⟦3548⟧)?"
answer: "Step 1, ⟦3555⟧. Step 2, ⟦3556⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["NumPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

The uniform says "every value in this range is equally likely". It is the starting point for random number generation, random search and some weight initialisations.

## Its moments

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mi>X</mi><mo>=</mo><mfrac><mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow><mn>2</mn></mfrac><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>=</mo><mfrac><mrow><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>a</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow><mn>12</mn></mfrac></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=x>=0&&x<=1?1:0 x=-0.5,1.5 y=0,1.3 tk=0.5 | Uniform(0, 1): density 1 on [0, 1]. Mean 0.5, variance 1/12.
~~~

```
rng = np.random.default_rng(0)
a = np.sqrt(6 / 512)
w = rng.uniform(-a, a, 1_000_000)
print(round(a, 3), round(w.var() * 512, 2))  # 0.108 2.0
```

<div class="take">
<div class="lab">Key takeaway</div>

the uniform has mean at the centre and variance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>a</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mi mathvariant="normal">/</mi><mn>12</mn></mrow></math></span>.

</div>

## Where this breaks in practice

- `rng.random()` gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>; the endpoint 1 is excluded. Code that maps <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>u</mi><mo stretchy="false">)</mo></mrow></math></span> relies on this.

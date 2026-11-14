---
id: "10.08"
course: "ml-maths-course"
module: "10"
slug: "inverse-cdf-sampling"
title: "Inverse-CDF Sampling"
keyIdeas: ["To sample ⟦513⟧ with CDF ⟦2801⟧: draw ⟦3305⟧ and return ⟦3306⟧.", "It works because ⟦3307⟧.", "Top-⟦102⟧ and ordinary token sampling invert a discrete CDF."]
question: "Give a formula to sample ⟦513⟧ with density ⟦3304⟧ on ⟦4⟧ from a uniform ⟦207⟧."
answer: "Step 1, ⟦3311⟧. Step 2, solve ⟦3312⟧: ⟦3313⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous random variables"]
tools: ["NumPy"]
topic: {"id": "10.02", "title": "Continuous random variables"}
---

## The big picture

Computers produce uniform random numbers. Every other distribution has to be built from them. Inverting the CDF is the simplest general way.

## The method

~~~dg-flow Inverse-CDF sampling.
U ~ Uniform(0, 1) -(apply F inverse)-> *X = F⁻¹(U) has CDF F
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sampling an exponential**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup></mrow></math></span>. Solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>x</mi></mrow></msup></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo>−</mo><mi>ln</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>u</mi><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mi>λ</mi></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
u = rng.random(200000)
x = -np.log(1 - u) * 200
print(round(x.mean(), -1), round(np.median(x), -1))  # 200.0 140.0
y = u ** (1 / 3)
print(round((y > 0.5).mean(), 2))  # 0.87
```

## Discrete version

For a discrete distribution, the CDF is a staircase. Draw <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> and pick the first value whose cumulative probability exceeds <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span>.

```
p = np.array([0.5, 0.3, 0.2])
cdf = np.cumsum(p)
rng = np.random.default_rng(1)
idx = np.searchsorted(cdf, rng.random(100000), side='right')
print(np.round(np.bincount(idx) / idx.size, 2))  # [0.5 0.3 0.2]
```

~~~dg-raw
%%viz f_bars v=0.5,0.8,1 l=token_a,token_b,token_c hi=none yl=cumulative_probability | The staircase for three tokens with probabilities 0.5, 0.3 and 0.2. A uniform draw of 0.65 lands in token b's step.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

push a uniform through the inverse CDF and you get the distribution you want.

</div>

## Where this breaks in practice

- When <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>F</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> has no formula (the Gaussian), libraries use other methods such as Box–Muller or the ziggurat.

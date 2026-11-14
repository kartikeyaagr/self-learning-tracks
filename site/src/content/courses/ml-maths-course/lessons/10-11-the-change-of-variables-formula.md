---
id: "10.11"
course: "ml-maths-course"
module: "10"
slug: "the-change-of-variables-formula"
title: "The Change-of-Variables Formula"
keyIdeas: ["For a strictly monotone ⟦2832⟧, ⟦3343⟧.", "Where ⟦2832⟧ stretches the line the density drops; where it squeezes, the density rises.", "The log-normal ⟦3342⟧ has median ⟦3344⟧ but mean ⟦3345⟧."]
question: "⟦3341⟧ and ⟦3342⟧. Find ⟦3325⟧."
answer: "Step 1, ⟦3354⟧ with derivative ⟦3355⟧. Step 2, ⟦3356⟧ for ⟦3331⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Transforming random variables"]
tools: ["NumPy"]
topic: {"id": "10.03", "title": "Transforming random variables"}
---

## The big picture

The CDF method always works, but for monotone maps there is a shortcut that generalises straight to many dimensions, which is what normalising flows use.

## The formula

Probability in a small interval is preserved when it moves: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi mathvariant="normal">∣</mi><mi>d</mi><mi>y</mi><mi mathvariant="normal">∣</mi><mo>=</mo><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi mathvariant="normal">∣</mi><mi>d</mi><mi>x</mi><mi mathvariant="normal">∣</mi></mrow></math></span>. So
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><msup><mi>g</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mrow><mo fence="true">∣</mo><mfrac><mrow><mi>d</mi><msup><mi>g</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow><mrow><mi>d</mi><mi>y</mi></mrow></mfrac><mo fence="true">∣</mo></mrow></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**a linear change**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> uniform on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mn>3</mn><mi>X</mi><mo>+</mo><mn>2</mn></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>2</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">]</mo></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>g</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mi>y</mi><mo>−</mo><mn>2</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span> with derivative <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>2</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">]</mo></mrow></math></span>: the interval got 3 times longer, so the density got 3 times smaller.

</div>

~~~dg-raw
%%viz f_curve f=Math.exp(-Math.log(x)*Math.log(x)/2)/(x*Math.sqrt(2*Math.PI)) x=0.02,5 y=0,0.7 tk=1 pts=1,median_1;1.65,mean_1.65 | The log-normal density, e^X for standard normal X. The long right tail pulls the mean (1.65) above the median (1).
~~~

```
rng = np.random.default_rng(0)
y = np.exp(rng.normal(0, 1, 1_000_000))
print(round(np.median(y), 2), round(y.mean(), 2), round(np.exp(0.5), 2))  # 1.0 1.65 1.65
```

<div class="take">
<div class="lab">Key takeaway</div>

moving probability through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi></mrow></math></span> multiplies the density by the absolute derivative of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>g</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>.

</div>

## Where this breaks in practice

- Forgetting the factor bites when you reparameterise a prior: a flat prior on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi></mrow></math></span> is not flat on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mi>σ</mi></mrow></math></span>.

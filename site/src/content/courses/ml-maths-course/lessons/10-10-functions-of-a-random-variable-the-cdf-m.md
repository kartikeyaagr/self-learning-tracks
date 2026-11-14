---
id: "10.10"
course: "ml-maths-course"
module: "10"
slug: "functions-of-a-random-variable-the-cdf-method"
title: "Functions of a Random Variable: the CDF Method"
keyIdeas: ["To find the distribution of ⟦3326⟧, write ⟦3327⟧, solve for the ⟦7⟧ values, then differentiate.", "The method works even when ⟦2832⟧ is not monotone, such as ⟦2328⟧ over all reals.", "⟦3328⟧ for a standard normal ⟦3329⟧ is chi-squared with 1 degree of freedom."]
question: "⟦3322⟧ and ⟦3323⟧. Find ⟦3324⟧ and ⟦3325⟧."
answer: "Step 1, ⟦3338⟧ for ⟦3339⟧. Step 2, ⟦3340⟧, which is unbounded near 0 but integrates to 1."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Transforming random variables"]
tools: ["NumPy"]
topic: {"id": "10.03", "title": "Transforming random variables"}
---

## The big picture

You often know the distribution of one thing and need the distribution of something computed from it. Noise is Gaussian; what about its square? The CDF method always works.

## Three steps

1. Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>F</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>g</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo><mo>≤</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>.
2. Work out which <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> satisfy <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≤</mo><mi>y</mi></mrow></math></span>, and compute that probability using <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>'s distribution.
3. Differentiate to get <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>Y</mi></msub></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**the square of a standard normal**

For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>F</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mo>−</mo><msqrt><mi>y</mi></msqrt><mo>≤</mo><mi>Z</mi><mo>≤</mo><msqrt><mi>y</mi></msqrt><mo stretchy="false">)</mo><mo>=</mo><mn>2</mn><mi mathvariant="normal">Φ</mi><mo stretchy="false">(</mo><msqrt><mi>y</mi></msqrt><mo stretchy="false">)</mo><mo>−</mo><mn>1</mn></mrow></math></span>.

Differentiate: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mn>2</mn><mi>ϕ</mi><mo stretchy="false">(</mo><msqrt><mi>y</mi></msqrt><mo stretchy="false">)</mo><mo>⋅</mo><mfrac><mn>1</mn><mrow><mn>2</mn><msqrt><mi>y</mi></msqrt></mrow></mfrac><mo>=</mo><mfrac><mn>1</mn><msqrt><mrow><mn>2</mn><mi>π</mi><mi>y</mi></mrow></msqrt></mfrac><msup><mi>e</mi><mrow><mo>−</mo><mi>y</mi><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup></mrow></math></span>, the chi-squared density with 1 degree of freedom.

</div>

~~~dg-raw
%%viz f_curve f=1/(2*Math.sqrt(x)) x=0.01,1 y=0,5 tk=0.25 | The density of Y = X² for uniform X. Squaring pushes values toward 0, so the density piles up there, yet its area is still 1.
~~~

```
rng = np.random.default_rng(0)
y = rng.random(1_000_000) ** 2
print(round((y <= 0.25).mean(), 3), round(y.mean(), 2))  # 0.5 0.33
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>F</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mn>0.25</mn><mo stretchy="false">)</mo><mo>=</mo><msqrt><mn>0.25</mn></msqrt><mo>=</mo><mn>0.5</mn></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>Y</mi><mo stretchy="false">]</mo><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

get the CDF of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> from the CDF of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>, then differentiate.

</div>

## Where this breaks in practice

- Forgetting a branch: for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><msup><mi>X</mi><mn>2</mn></msup></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> on all reals, both <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><msqrt><mi>y</mi></msqrt></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo>−</mo><msqrt><mi>y</mi></msqrt></mrow></math></span> map to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>.

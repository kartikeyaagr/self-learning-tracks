---
id: "11.26"
course: "ml-maths-course"
module: "11"
slug: "linear-combinations-of-gaussians"
title: "Linear Combinations of Gaussians"
keyIdeas: ["⟦3797⟧: linear maps keep a Gaussian Gaussian.", "Independent Gaussians add means and covariances: ⟦3798⟧.", "Any projection ⟦2133⟧ is a one-dimensional Gaussian with variance ⟦2134⟧."]
question: "⟦3794⟧ and ⟦3795⟧ independently. What is the distribution of ⟦3796⟧?"
answer: "Step 1, mean ⟦3808⟧. Step 2, variance ⟦3809⟧. So ⟦3810⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Gaussian combinations"]
tools: ["NumPy"]
topic: {"id": "11.05", "title": "Gaussian combinations"}
---

## The big picture

The Gaussian's superpower: it stays Gaussian under everything linear. That makes calculations with it exact.

## The rules

| Operation | Result |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>X</mi><mo>+</mo><mi>b</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mi>A</mi><mi>μ</mi><mo>+</mo><mi>b</mi><mo separator="true">,</mo><mi>A</mi><mi mathvariant="normal">Σ</mi><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow></math></span>, independent | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><msub><mi>μ</mi><mi>X</mi></msub><mo>+</mo><msub><mi>μ</mi><mi>Y</mi></msub><mo separator="true">,</mo><msub><mi mathvariant="normal">Σ</mi><mi>X</mi></msub><mo>+</mo><msub><mi mathvariant="normal">Σ</mi><mi>Y</mi></msub><mo stretchy="false">)</mo></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>μ</mi><mo separator="true">,</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi mathvariant="normal">Σ</mi><mi>w</mi><mo stretchy="false">)</mo></mrow></math></span> |

```
rng = np.random.default_rng(0)
x = rng.normal(1, 2, 1_000_000); y = rng.normal(2, 3, 1_000_000)
s = 2 * x - y
print(round(s.mean(), 1), round(s.var()), round(((s - s.mean()) ** 3).mean() / s.std() ** 3, 2) + 0)  # 0.0 25 0.0
```

The skewness is 0, as it must be for a Gaussian.

<div class="ex">
<div class="lab">Worked example</div>

**sensor fusion**

Two independent measurements of one quantity: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mn>10</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mn>12</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Precisions (one over variance) add: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.25</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>1.25</mn></mrow></math></span>, so the combined variance is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.8</mn></mrow></math></span>.

The combined mean is the precision-weighted average: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>10</mn><mo>×</mo><mn>0.25</mn><mo>+</mo><mn>12</mn><mo>×</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>1.25</mn><mo>=</mo><mn>11.6</mn></mrow></math></span>.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

linear maps and sums of independent Gaussians are Gaussian, with means and covariances given by formula.

</div>

## Where this breaks in practice

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> Gaussian and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> Gaussian does not make the pair jointly Gaussian; the rules need the joint assumption.

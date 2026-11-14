---
id: "10.38"
course: "ml-maths-course"
module: "10"
slug: "variance-preserving-initialisation"
title: "Variance-Preserving Initialisation"
keyIdeas: ["A pre-activation ⟦3568⟧ has variance ⟦3569⟧ for independent zero-mean weights.", "Xavier: ⟦3570⟧. He, for ReLU: ⟦3571⟧.", "The 2 in He makes up for ReLU zeroing half the inputs."]
question: "Give the He standard deviation for ⟦3548⟧."
answer: "Step 1, ⟦3575⟧. Step 2, ⟦3576⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["NumPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
---

## The big picture

If each layer multiplies the variance of its signal by more than 1, activations explode through a deep network; by less than 1, they vanish. Initialisation picks the weight variance to keep it at 1.

## The calculation

For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><msub><mo>∑</mo><mi>j</mi></msub><msub><mi>w</mi><mi>j</mi></msub><msub><mi>a</mi><mi>j</mi></msub></mrow></math></span> with independent zero-mean weights, the variances add: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>z</mi><mo>=</mo><msub><mi>n</mi><mtext>in</mtext></msub><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo><mtext> </mtext><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>a</mi><mn>2</mn></msup><mo stretchy="false">]</mo></mrow></math></span>. After a ReLU, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>a</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mi mathvariant="normal">Var</mi><mo>⁡</mo><msub><mi>z</mi><mtext>prev</mtext></msub></mrow></math></span>. Setting the product to 1 gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo><mo>=</mo><mn>2</mn><mi mathvariant="normal">/</mi><msub><mi>n</mi><mtext>in</mtext></msub></mrow></math></span>.

```
rng = np.random.default_rng(0)
x = rng.normal(size=(2000, 512))
for name, s in [('too small', 0.03), ('He', np.sqrt(2 / 512)), ('too big', 0.09)]:
    h = x
    for _ in range(20):
        h = np.maximum(0, h @ rng.normal(0, s, (512, 512)))
    print(name, f"{h.std():.2g}")
# too small 3.6e-07
# He 0.7
# too big 1.5e+03
```

~~~dg-raw
%%viz f_bars v=-6.4,-0.2,3.2 l=σ_=_0.03,He_σ_=_0.0625,σ_=_0.09 hi=none yl=log₁₀_of_activation_std_after_20_layers | Activation size after 20 ReLU layers, on a log scale. The He scale stays near 10⁰ = 1; the others shrink to 10⁻⁶ or grow to 10³.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

pick the weight variance so each layer keeps the signal's variance near 1.

</div>

## Where this breaks in practice

- Normalisation layers and residual connections make networks less sensitive to initialisation, but very deep plain networks still need it.

---
id: "10.14"
course: "ml-maths-course"
module: "10"
slug: "the-reparameterisation-trick"
title: "The Reparameterisation Trick"
keyIdeas: ["Sampling ⟦3384⟧ directly gives no gradient with respect to ⟦1238⟧ or ⟦1239⟧.", "Writing ⟦3382⟧ with ⟦3385⟧ gives the same distribution and a differentiable path.", "Variational autoencoders train this way."]
question: "Loss ⟦3381⟧ with ⟦3382⟧. What is ⟦3383⟧, and what does the per-sample estimate average to?"
answer: "Step 1, exactly ⟦3390⟧, so the derivative is ⟦3391⟧. Step 2, per sample ⟦3392⟧, which averages to ⟦3391⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Transforming random variables"]
tools: ["NumPy", "PyTorch"]
topic: {"id": "10.03", "title": "Transforming random variables"}
---

## The big picture

A VAE samples a latent code inside the network. Backpropagation needs every step to be a differentiable function of the parameters, and "draw a random number" isn't one. The trick moves the randomness into an input.

## The trick

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>z</mi><mo>=</mo><mi>μ</mi><mo>+</mo><mi>σ</mi><mi>ε</mi><mo separator="true">,</mo><mspace width="2em"/><mi>ε</mi><mo>∼</mo><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span></span>
<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi></mrow></math></span> is a fixed input to the computation. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> is now an ordinary function of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi></mrow></math></span>, with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>z</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>μ</mi><mo>=</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>z</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>σ</mi><mo>=</mo><mi>ε</mi></mrow></math></span>.

~~~dg-flow Where the randomness goes.
ε ~ N(0, 1) -(scale by σ, shift by μ)-> z -(network)-> loss -(backprop)-> *gradients for μ and σ
~~~

```
rng = np.random.default_rng(0)
mu, sigma = 1.5, 0.5
eps = rng.normal(size=100000)
z = mu + sigma * eps
print(round((2 * z).mean(), 2), round((2 * z * eps).mean(), 2))  # 3.0 1.0
```

The second number estimates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>z</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>σ</mi><mo>=</mo><mn>2</mn><mi>σ</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

write a sample as a deterministic function of parameters and fixed noise, and gradients flow through it.

</div>

## Where this breaks in practice

- Discrete variables have no reparameterisation of this kind. Gumbel-softmax and straight-through estimators are approximations.

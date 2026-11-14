---
id: "09.27"
course: "ml-maths-course"
module: "09"
slug: "gradients-of-common-ml-losses"
title: "Gradients of Common ML Losses"
keyIdeas: ["MSE ⟦2766⟧ has gradient ⟦2767⟧.", "Logistic loss has gradient ⟦2768⟧; softmax cross-entropy has ⟦2769⟧.", "The pattern is always \"prediction minus target, pulled back through the inputs\"."]
question: "One example ⟦1868⟧, label ⟦831⟧, logistic weights ⟦2764⟧. Compute the gradient and one step with ⟦2765⟧."
answer: "Step 1, ⟦2778⟧, ⟦2779⟧. Step 2, gradient ⟦2780⟧. Step 3, ⟦2781⟧. Step 4, now ⟦2782⟧ and ⟦2783⟧: closer to the label."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
---

## The big picture

A handful of gradients cover most of classical ML. They all share one shape, which makes them easy to remember and to sanity-check.

## The table

| Loss | Gradient |
| --- | --- |
| MSE <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mrow><mn>2</mn><mi>n</mi></mrow></mfrac><mi mathvariant="normal">∥</mi><mi>X</mi><mi>w</mi><mo>−</mo><mi>y</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mi>n</mi></mfrac><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>X</mi><mi>w</mi><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> |
| logistic, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><mi>X</mi><mi>w</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mi>n</mi></mfrac><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>σ</mi><mo stretchy="false">(</mo><mi>X</mi><mi>w</mi><mo stretchy="false">)</mo><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> |
| softmax cross-entropy, logits <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>z</mi><mo>=</mo><mi mathvariant="normal">softmax</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>−</mo><mi>y</mi></mrow></math></span> |
| L2 penalty <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mi>λ</mi><mn>2</mn></mfrac><mi mathvariant="normal">∥</mi><mi>w</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mi>w</mi></mrow></math></span> |

<div class="ex">
<div class="lab">Worked example</div>

**deriving the logistic gradient**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℓ</mi><mo>=</mo><mo>−</mo><mi>y</mi><mi>log</mi><mo>⁡</mo><mi>σ</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>−</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>σ</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math></span>, with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>σ</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mi>σ</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>σ</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mrow></math></span>.

Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mrow><mi>d</mi><mi mathvariant="normal">ℓ</mi></mrow><mrow><mi>d</mi><mi>z</mi></mrow></mfrac><mo>=</mo><mo>−</mo><mi>y</mi><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>σ</mi><mo stretchy="false">)</mo><mo>+</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mi>σ</mi><mo>=</mo><mi>σ</mi><mo>−</mo><mi>y</mi></mrow></math></span>.

Chain rule with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi mathvariant="normal">∇</mi><mi>w</mi></msub><mi mathvariant="normal">ℓ</mi><mo>=</mo><mo stretchy="false">(</mo><mi>σ</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>x</mi></mrow></math></span>.

</div>

```
sig = lambda z: 1 / (1 + np.exp(-z))
x = np.array([1.0, 2]); y = 1.0; w = np.zeros(2)
for step in range(3):
    w = w - 0.1 * (sig(w @ x) - y) * x
    print(step, np.round(w, 4), round(sig(w @ x), 3))
```

## A numerical check

```
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3)); y = rng.normal(size=5); w = rng.normal(size=3)
loss = lambda w: ((X @ w - y) ** 2).sum() / (2 * 5)
grad = X.T @ (X @ w - y) / 5
h = 1e-6
num = np.array([(loss(w + h * e) - loss(w - h * e)) / (2 * h) for e in np.eye(3)])
print(np.allclose(grad, num))  # True
```

<div class="take">
<div class="lab">Key takeaway</div>

for linear-model losses the gradient is (prediction minus target) times the inputs; large when wrong, small when right.

</div>

## Where this breaks in practice

- PyTorch adds new gradients onto `.grad`. Forget `optimizer.zero_grad()` and each step uses the sum of all previous gradients.

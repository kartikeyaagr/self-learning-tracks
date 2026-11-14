---
id: "09.38"
course: "ml-maths-course"
module: "09"
slug: "backpropagation-through-a-linear-layer"
title: "Backpropagation Through a Linear Layer"
keyIdeas: ["For ⟦2872⟧ with upstream gradient ⟦2936⟧: ⟦2937⟧ and ⟦2938⟧.", "A bias added to every row gets ⟦2939⟧, the column sums of ⟦2873⟧.", "Each weight's gradient sums (input value) times (upstream gradient) over the batch."]
question: "⟦2933⟧, ⟦2934⟧. Compute ⟦2935⟧."
answer: "Step 1, ⟦2944⟧. Step 2, ⟦2945⟧. Step 3, shape ⟦2946⟧, same as ⟦1259⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Matrix calculus"]
tools: ["NumPy", "PyTorch"]
topic: {"id": "09.05", "title": "Matrix calculus"}
---

## The big picture

Every dense layer, attention projection and MLP block uses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mi>X</mi><mi>W</mi></mrow></math></span>. Its backward pass is two matrix products. Knowing them lets you read and debug any backprop code.

## The derivation

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>L</mi><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>G</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mi>Y</mi><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>Y</mi><mo>=</mo><mi>d</mi><mi>X</mi><mtext> </mtext><mi>W</mi><mo>+</mo><mi>X</mi><mtext> </mtext><mi>d</mi><mi>W</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>G</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mtext> </mtext><mi>d</mi><mi>W</mi><mo stretchy="false">)</mo></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>W</mi><mo>=</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>G</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>G</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mi>X</mi><mtext> </mtext><mi>W</mi><mo stretchy="false">)</mo><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>W</mi><msup><mi>G</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mi>X</mi><mo stretchy="false">)</mo></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>X</mi><mo>=</mo><mi>G</mi><msup><mi>W</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>.

~~~dg-flow Backward through Y = X W: the upstream gradient G goes two ways.
G = dL/dY -(times W transposed, on the right)-> *dL/dX = G Wᵀ
G = dL/dY -(times X transposed, on the left)-> *dL/dW = Xᵀ G
~~~

```
X = np.array([[1.0, 2], [3, 4]]); G = np.array([[1.0], [-1]])
print(X.T @ G)
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3)); W = rng.normal(size=(3, 2)); b = rng.normal(size=2)
L = lambda b: ((X @ W + b) ** 2).sum() / 2
G = X @ W + b
h = 1e-6
num = np.array([(L(b + h * e) - L(b - h * e)) / (2 * h) for e in np.eye(2)])
print(np.allclose(G.sum(0), num))  # True
```

The bias gradient is the column sum of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>G</mi></mrow></math></span>: the bias was copied to all rows, so its gradient adds them up.

<div class="take">
<div class="lab">Key takeaway</div>

backward through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi><mo>=</mo><mi>X</mi><mi>W</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>G</mi></mrow></math></span> for the weights and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>G</mi><msup><mi>W</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> for the inputs; biases get column sums.

</div>

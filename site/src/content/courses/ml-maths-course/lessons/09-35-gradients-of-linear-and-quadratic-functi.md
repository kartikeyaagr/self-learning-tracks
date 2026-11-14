---
id: "09.35"
course: "ml-maths-course"
module: "09"
slug: "gradients-of-linear-and-quadratic-functions"
title: "Gradients of Linear and Quadratic Functions"
keyIdeas: ["⟦2881⟧ and ⟦2882⟧, which is ⟦2704⟧ for symmetric ⟦8⟧.", "⟦2883⟧.", "These are matrix versions of ⟦2705⟧, ⟦2706⟧ and ⟦2884⟧."]
question: "Derive ⟦2880⟧ by indices."
answer: "Step 1, ⟦2893⟧. Step 2, terms with ⟦2894⟧ have ⟦2895⟧ or ⟦2896⟧. Step 3, ⟦2897⟧. Step 4, so ⟦2898⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Matrix calculus"]
tools: ["NumPy"]
topic: {"id": "09.05", "title": "Matrix calculus"}
---

## The big picture

Three identities cover a large share of ML derivations. Each mirrors a one-variable rule you already know.

## Index by index

<div class="ex">
<div class="lab">Worked example</div>

**a dot product**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>=</mo><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><msub><mi>a</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>a</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><msub><mi>a</mi><mn>3</mn></msub><msub><mi>x</mi><mn>3</mn></msub></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>k</mi></msub><mo>=</mo><msub><mi>a</mi><mi>k</mi></msub></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>L</mi><mo>=</mo><mi>a</mi></mrow></math></span>.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a quadratic form**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>k</mi></msub><mo>=</mo><mo stretchy="false">(</mo><mi>A</mi><mi>x</mi><msub><mo stretchy="false">)</mo><mi>k</mi></msub><mo>+</mo><mo stretchy="false">(</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><msub><mo stretchy="false">)</mo><mi>k</mi></msub></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mo>=</mo><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mi>x</mi></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
A = rng.normal(size=(3, 3)); x = rng.normal(size=3); h = 1e-6
f = lambda v: v @ A @ v
num = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(3)])
print(np.allclose(num, (A + A.T) @ x))  # True
```

## Least squares

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>−</mo><mn>2</mn><msup><mi>b</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>+</mo><msup><mi>b</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span>, so by the two rules
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi mathvariant="normal">∇</mi><mi>x</mi></msub><mi mathvariant="normal">∥</mi><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><mn>2</mn><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>−</mo><mn>2</mn><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi><mo>=</mo><mn>2</mn><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Setting it to zero gives the normal equations (lesson 08.01).

| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span> | One-variable analogue |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mi>x</mi><mo>→</mo><mi>a</mi></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><msup><mi>x</mi><mn>2</mn></msup><mo>→</mo><mn>2</mn><mi>a</mi><mi>x</mi></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mi>x</mi><mo>−</mo><mi>b</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>→</mo><mn>2</mn><mi>a</mi><mo stretchy="false">(</mo><mi>a</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> |

<div class="take">
<div class="lab">Key takeaway</div>

linear terms give constant gradients, quadratic forms give <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mi>x</mi></mrow></math></span>, and least squares gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span>.

</div>

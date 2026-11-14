---
id: "09.36"
course: "ml-maths-course"
module: "09"
slug: "the-differential-and-trace-method"
title: "The Differential and Trace Method"
keyIdeas: ["Compute ⟦2900⟧, rewrite it as ⟦2901⟧, and read off ⟦2902⟧.", "Trace rules: ⟦2903⟧, ⟦2904⟧, a scalar equals its trace.", "Useful differentials: ⟦2905⟧, ⟦2906⟧, ⟦2907⟧."]
question: "Use differentials to find ⟦2899⟧."
answer: "Step 1, ⟦2916⟧. Step 2, match with ⟦2917⟧: ⟦2918⟧. Step 3, ⟦2919⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Matrix calculus"]
tools: ["NumPy"]
topic: {"id": "09.05", "title": "Matrix calculus"}
---

## The big picture

Index-chasing gets messy for matrices. Differentials let you derive whole-matrix gradients in a few lines, using only the product rule and the cyclic property of the trace.

## Why it works

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>G</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mi>W</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mo>∑</mo><mrow><mi>i</mi><mo separator="true">,</mo><mi>j</mi></mrow></munder><msub><mi>G</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mtext> </mtext><mi>d</mi><msub><mi>W</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mrow></math></span></span>
That is "partial derivative times small change, summed over entries", exactly what <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>L</mi></mrow></math></span> is. So whatever multiplies <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>W</mi></mrow></math></span> inside the trace (transposed) is the gradient.

```
rng = np.random.default_rng(1)
A = rng.normal(size=(2, 3)); B = rng.normal(size=(3, 4)); C = rng.normal(size=(4, 2))
print(np.isclose(np.trace(A @ B @ C), np.trace(C @ A @ B)))  # True
```

<div class="ex">
<div class="lab">Worked example</div>

**the quadratic form again**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>f</mi><mo>=</mo><mi>d</mi><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>x</mi><mo>+</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span>. The first term is a scalar, so it equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mi>x</mi></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>f</mi><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo>=</mo><mo stretchy="false">(</mo><mi>A</mi><mo>+</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">)</mo><mi>x</mi></mrow></math></span>.

</div>

## Log-determinant

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>log</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo><mi mathvariant="normal">Σ</mi><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>d</mi><mi mathvariant="normal">Σ</mi><mo stretchy="false">)</mo></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi mathvariant="normal">∇</mi><mi mathvariant="normal">Σ</mi></msub><mi>log</mi><mo>⁡</mo><mi>det</mi><mo>⁡</mo><mi mathvariant="normal">Σ</mi><mo>=</mo><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mi mathvariant="normal">⊤</mi></mrow></msup><mo>=</mo><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> for symmetric <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span>. It appears in Gaussian maximum likelihood.

```
rng = np.random.default_rng(2)
S = rng.normal(size=(3, 3)); S = S @ S.T + 3 * np.eye(3)
h = 1e-6; num = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        E = np.zeros((3, 3)); E[i, j] = h
        num[i, j] = (np.linalg.slogdet(S + E)[1] - np.linalg.slogdet(S - E)[1]) / (2 * h)
print(np.allclose(num, np.linalg.inv(S), atol=1e-6))  # True
```

<div class="take">
<div class="lab">Key takeaway</div>

write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>L</mi><mo>=</mo><mi mathvariant="normal">tr</mi><mo>⁡</mo><mo stretchy="false">(</mo><msup><mi>G</mi><mi mathvariant="normal">⊤</mi></msup><mi>d</mi><mi>W</mi><mo stretchy="false">)</mo></mrow></math></span> and the gradient is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>G</mi></mrow></math></span>; the cyclic trace rule does the rearranging.

</div>

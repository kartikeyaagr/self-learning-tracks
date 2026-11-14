---
id: "04.05"
course: "ml-maths-course"
module: "04"
slug: "near-dependence-in-data"
title: "Near-Dependence in Data"
keyIdeas: ["Real data is rarely exactly dependent, but often nearly so (height and shoe size).", "Near-dependence shows up as a tiny singular value: ⟦1369⟧ close to 0.", "It makes coefficients unstable: large uncertainty and sign flips when the data is resampled."]
question: "Columns ⟦1366⟧ and ⟦1367⟧. Are they independent? Are they nearly dependent? What does `matrix_rank` say with the default and with tolerance ⟦1368⟧?"
answer: "Step 1, the second column is not exactly twice the first (the middle entry differs by ⟦1175⟧), so technically they are independent. Step 2, but it is extremely close to twice the first, so they are nearly dependent; the ratio ⟦1369⟧ is below ⟦1175⟧. Step 3, `matrix_rank` with the default tolerance gives 2; with tolerance ⟦1368⟧ it gives 1, the honest answer."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Vectors, spans and dependence"]
tools: ["NumPy"]
topic: {"id": "04.01", "title": "Vectors, spans and dependence"}
---

## The big picture

On a computer, exact dependence almost never happens: rounding and noise make every set of columns technically independent. What matters is how close to dependent they are, because that decides how stable a fitted model is.

## Measuring closeness

The singular values of a matrix (module 07) measure how much it stretches each direction. A near-zero singular value means some combination of the columns is nearly zero: a near-redundant direction.

```
V = np.array([[1, 2], [2, 4.000001], [3, 6]])
s = np.linalg.svd(V, compute_uv=False)
print(s[-1] / s[0] < 1e-6)  # True
print(np.linalg.matrix_rank(V), np.linalg.matrix_rank(V, tol=1e-5))  # 2 1
```

## What it does to a regression

~~~dg-flow How near-dependence appears in a regression.
Correlated features -(small sigma_min)-> *Ill-conditioned X -(solve)-> Unstable coefficients
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two nearly identical features**

Features: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub></mrow></math></span> = height, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub></mrow></math></span> = height plus a little noise. True model: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msub><mi>x</mi><mn>1</mn></msub></mrow></math></span>.

The data fits almost equally well with weights <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0.5</mn><mo stretchy="false">)</mo></mrow></math></span> or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>10</mn><mo separator="true">,</mo><mo>−</mo><mn>9</mn><mo stretchy="false">)</mo></mrow></math></span>.

Resample the data and the fitted weights can jump between these. Predictions stay fine; the individual weights are meaningless.

</div>

```
rng = np.random.default_rng(0)
h = rng.normal(170, 10, 50)
X = np.column_stack([h, h + rng.normal(0, 0.01, 50)])
s = np.linalg.svd(X - X.mean(0), compute_uv=False)
print(s[1] / s[0] < 1e-3)  # True
```

## Diagnosing it

- **Condition number** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> (after centring): large means nearly dependent.
- **Variance inflation factors** (VIF): above about 10 flags a feature that is nearly a combination of the others.
- **Correlation matrix**: pairs near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>±</mo><mn>1</mn></mrow></math></span> are a warning, though dependence can involve three or more features with no single high correlation.

## Fixes

- Drop or merge redundant features.
- Ridge regularisation, which picks a stable compromise.
- PCA, which rotates to independent directions (module 08).

<div class="take">
<div class="lab">Key takeaway</div>

in data, ask how close to dependent the columns are. A tiny singular value means unstable, uninterpretable coefficients.

</div>

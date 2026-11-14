---
id: "11.17"
course: "ml-maths-course"
module: "11"
slug: "mean-vectors-and-covariance-matrices"
title: "Mean Vectors and Covariance Matrices"
keyIdeas: ["For a random vector ⟦3725⟧: ⟦3726⟧ and ⟦3727⟧.", "The diagonal of ⟦930⟧ holds variances; the off-diagonal holds covariances.", "⟦930⟧ is symmetric and positive semidefinite, because ⟦1772⟧."]
question: "⟦513⟧ has covariance ⟦3721⟧. What are ⟦3722⟧, ⟦3723⟧ and ⟦3724⟧?"
answer: "Step 1, the diagonal gives ⟦453⟧ and ⟦22⟧. Step 2, the off-diagonal gives ⟦132⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

An embedding, an input vector or a layer's activations are random vectors. Their mean vector and covariance matrix summarise them the way a mean and variance summarise one number.

## The definitions

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>μ</mi><mo>=</mo><mi mathvariant="normal">𝔼</mi><mi>X</mi><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">Σ</mi><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">]</mo><mo separator="true">,</mo><mspace width="2em"/><msub><mi mathvariant="normal">Σ</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mi mathvariant="normal">Cov</mi><mo>⁡</mo><mo stretchy="false">(</mo><msub><mi>X</mi><mi>i</mi></msub><mo separator="true">,</mo><msub><mi>X</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow></math></span></span>

~~~dg-raw
%%viz f_plane x=-6,8 y=-4,8 tk=2 pg=5.00:4.00_4.94:4.46_4.76:4.85_4.46:5.15_4.06:5.35_3.57:5.45_3.00:5.45_2.37:5.34_1.69:5.13_1.00:4.83_0.31:4.44_-0.37:3.97_-1.00:3.45_-1.57:2.88_-2.06:2.29_-2.46:1.68_-2.76:1.09_-2.94:0.52_-3.00:0.00_-2.94:-0.46_-2.76:-0.85_-2.46:-1.15_-2.06:-1.35_-1.57:-1.45_-1.00:-1.45_-0.37:-1.34_0.31:-1.13_1.00:-0.83_1.69:-0.44_2.37:0.03_3.00:0.55_3.57:1.12_4.06:1.71_4.46:2.32_4.76:2.91_4.94:3.48,a pt=1,2,μ,k | A cloud of samples with mean (1, 2) and covariance [[4, 2], [2, 3]] mostly falls inside this ellipse. It tilts because the covariance is positive.
~~~

```
rng = np.random.default_rng(0)
X = rng.multivariate_normal([1.0, 2], [[4.0, 2], [2, 3]], size=500000)
print(np.round(X.mean(0), 1), np.round(np.cov(X, rowvar=False), 1).tolist())  # [1. 2.] [[4.0, 2.0], [2.0, 3.0]]
```

<div class="take">
<div class="lab">Key takeaway</div>

a random vector has a mean vector and a symmetric positive semidefinite covariance matrix.

</div>

## Where this breaks in practice

- `np.cov` treats rows as variables by default; pass `rowvar=False` for a data matrix with one row per example.

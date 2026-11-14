---
id: "11.25"
course: "ml-maths-course"
module: "11"
slug: "the-multivariate-normal"
title: "The Multivariate Normal"
keyIdeas: ["⟦3786⟧.", "The exponent is minus half the squared Mahalanobis distance; contours are ellipses along the eigenvectors of ⟦930⟧.", "Compute ⟦3787⟧ from the Cholesky factor, ⟦3788⟧, not from `det`."]
question: "⟦3785⟧. Along which directions do the contours stretch, and by how much?"
answer: "Step 1, eigenvalues ⟦1479⟧ and ⟦3225⟧ with eigenvectors ⟦111⟧ and ⟦1352⟧. Step 2, spread ⟦3792⟧ along the diagonal and ⟦3793⟧ across it."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Gaussian combinations"]
tools: ["NumPy", "SciPy"]
topic: {"id": "11.05", "title": "Gaussian combinations"}
---

## The big picture

The multivariate Gaussian is the workhorse distribution for vectors: Gaussian processes, diffusion models, VAEs and Kalman filters all use it.

## Reading the formula

- The exponent is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow></math></span> times the squared Mahalanobis distance: distance from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span> in units of the data's own spread.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi mathvariant="normal">Σ</mi><msup><mi mathvariant="normal">∣</mi><mrow><mo>−</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mi>π</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mi>d</mi><mi mathvariant="normal">/</mi><mn>2</mn></mrow></msup></mrow></math></span> make the total probability 1.

~~~dg-raw
%%viz f_plane x=-3,3 y=-3,3 tk=1 pg=1.00:0.80_0.98:0.89_0.94:0.96_0.87:0.99_0.77:1.00_0.64:0.97_0.50:0.92_0.34:0.84_0.17:0.73_0.00:0.60_-0.17:0.45_-0.34:0.29_-0.50:0.12_-0.64:-0.05_-0.77:-0.23_-0.87:-0.39_-0.94:-0.55_-0.98:-0.68_-1.00:-0.80_-0.98:-0.89_-0.94:-0.96_-0.87:-0.99_-0.77:-1.00_-0.64:-0.97_-0.50:-0.92_-0.34:-0.84_-0.17:-0.73_-0.00:-0.60_0.17:-0.45_0.34:-0.29_0.50:-0.12_0.64:0.05_0.77:0.23_0.87:0.39_0.94:0.55_0.98:0.68,a;2.00:1.60_1.97:1.78_1.88:1.91_1.73:1.99_1.53:2.00_1.29:1.95_1.00:1.84_0.68:1.67_0.35:1.46_0.00:1.20_-0.35:0.90_-0.68:0.58_-1.00:0.24_-1.29:-0.11_-1.53:-0.45_-1.73:-0.79_-1.88:-1.09_-1.97:-1.37_-2.00:-1.60_-1.97:-1.78_-1.88:-1.91_-1.73:-1.99_-1.53:-2.00_-1.29:-1.95_-1.00:-1.84_-0.68:-1.67_-0.35:-1.46_-0.00:-1.20_0.35:-0.90_0.68:-0.58_1.00:-0.24_1.29:0.11_1.53:0.45_1.73:0.79_1.88:1.09_1.97:1.37,a va=0,0,0.95,0.95,√1.8_along_(1~_1),f;0,0,0.32,-0.32,√0.2,b | Contours of the Gaussian with correlation 0.8. The ellipses stretch along (1, 1), the top eigenvector, and are thin across it.
~~~

```
from scipy import stats
mu = np.zeros(2); S = np.array([[1.0, 0.8], [0.8, 1]]); x = np.array([1.0, 0.5])
L = np.linalg.cholesky(S)
z = np.linalg.solve(L, x - mu)
logp = -0.5 * z @ z - np.log(np.diag(L)).sum() - np.log(2 * np.pi)
print(round(logp, 4), round(stats.multivariate_normal(mu, S).logpdf(x), 4))  # -1.9521 -1.9521
```

<div class="take">
<div class="lab">Key takeaway</div>

a multivariate normal is set by its mean and covariance; its contours are ellipses along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span>'s eigenvectors.

</div>

## Where this breaks in practice

- Covariances estimated with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> close to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> are nearly singular. Add a small diagonal or use a shrinkage estimator before inverting.

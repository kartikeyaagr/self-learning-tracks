---
id: "08.05"
course: "ml-maths-course"
module: "08"
slug: "ridge-regression"
title: "Ridge Regression"
keyIdeas: ["Ridge minimises ⟦2386⟧, giving ⟦2387⟧.", "Through the SVD it multiplies the OLS component along singular direction ⟦274⟧ by ⟦2388⟧.", "It keeps well-measured directions and suppresses noisy ones; weight decay in neural networks is the same penalty."]
question: "A centred ⟦513⟧ has singular values 10 and 0.1, and ⟦1588⟧. By what factor is each direction shrunk, and why is that usually good?"
answer: "Step 1, direction 1: ⟦2395⟧, almost untouched. Step 2, direction 2: ⟦2396⟧, almost removed. Step 3, the OLS variance along direction ⟦274⟧ is proportional to ⟦2397⟧, so direction 2 is ⟦2398⟧ times noisier. Ridge trades a little bias for a large drop in variance exactly where the data says little."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Linear regression"]
tools: ["NumPy", "scikit-learn"]
topic: {"id": "08.02", "title": "Linear regression"}
---

## The big picture

When features are collinear, OLS coefficients swing wildly. Adding a small penalty on their size fixes this. The SVD shows exactly what the penalty does, and the same idea explains weight decay.

## The estimator

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover><mi>λ</mi></msub><mo>=</mo><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><mo>+</mo><mi>λ</mi><mi>I</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi></mrow></math></span></span>
Adding <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mi>I</mi></mrow></math></span> raises every eigenvalue of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span>, so the matrix is always invertible for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>.

## The SVD view

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mi>U</mi><mi mathvariant="normal">Σ</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover><mi>λ</mi></msub><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><mfrac><msub><mi>σ</mi><mi>i</mi></msub><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo>+</mo><mi>λ</mi></mrow></mfrac><mo stretchy="false">(</mo><msubsup><mi>u</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><msub><mi>v</mi><mi>i</mi></msub></mrow></math></span></span>
OLS has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span> in that place. So ridge multiplies each direction by
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo>+</mo><mi>λ</mi></mrow></mfrac></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=x*x/(x*x+1) x=0,5 y=0,1 tk=1 pts=1,σ_=_1:_0.5;3,σ_=_3:_0.9 | Ridge's shrink factor σ²/(σ² + λ) with λ = 1, against the singular value σ. Strong directions (large σ) keep almost all of their OLS coefficient; weak ones are pushed toward 0.
~~~

```
rng = np.random.default_rng(1)
X = rng.normal(size=(50, 3)); y = X @ np.array([1.0, 2, 3]) + 0.1 * rng.normal(size=50)
lam = 5.0
ridge = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
via_svd = Vt.T @ (s / (s ** 2 + lam) * (U.T @ y))
print(np.allclose(ridge, via_svd))  # True
```

## Choosing λ

~~~dg-flow Choosing lambda.
Split data -(fit ridge for a grid of λ)-> validation error per λ -(pick the minimum)-> *refit on train + validation, report test error once
~~~

<div class="ex">
<div class="lab">Worked example</div>

**do not penalise the intercept**

Adding 1000 to every target should raise only the intercept by 1000.

If the intercept is inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mi mathvariant="normal">∥</mi><mi>β</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span>, the penalty resists that shift and the slopes change too.

Centre <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> first, or leave the intercept out of the penalty. scikit-learn's `Ridge` does this with `fit_intercept=True`.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

ridge shrinks each singular direction by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo>+</mo><mi>λ</mi><mo stretchy="false">)</mo></mrow></math></span>: strong directions stay, noisy ones fade.

</div>

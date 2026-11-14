---
id: "08.04"
course: "ml-maths-course"
module: "08"
slug: "regression-as-a-probability-model"
title: "Regression as a Probability Model"
keyIdeas: ["With independent Gaussian noise ⟦2366⟧, maximising the likelihood is minimising squared error.", "Then ⟦2367⟧ is unbiased with ⟦2368⟧, and ⟦2369⟧ is unbiased.", "Nearly collinear features make ⟦895⟧ nearly singular and the coefficient variances huge. ⟦2370⟧ measures in-sample fit only."]
question: "For the four points ⟦2318⟧ the line fit has SSE ⟦2363⟧. Compute SST, ⟦2364⟧ and ⟦2365⟧."
answer: "Step 1, ⟦2331⟧; SST ⟦2381⟧. Step 2, ⟦2382⟧. Step 3, ⟦2383⟧, ⟦2384⟧, so ⟦2385⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Linear regression"]
tools: ["NumPy"]
topic: {"id": "08.02", "title": "Linear regression"}
---

## The big picture

Least squares gives a best fit. Adding a story about how the data was generated (a line plus Gaussian noise) lets you say how uncertain each coefficient is and why correlated features make coefficients jumpy.

## Squared error from Gaussian noise

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>log</mi><mo>⁡</mo><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo>∣</mo><mi>X</mi><mo separator="true">,</mo><mi>β</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mfrac><mi>n</mi><mn>2</mn></mfrac><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>2</mn><mi>π</mi><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">)</mo><mo>−</mo><mfrac><mn>1</mn><mrow><mn>2</mn><msup><mi>σ</mi><mn>2</mn></msup></mrow></mfrac><mi mathvariant="normal">∥</mi><mi>y</mi><mo>−</mo><mi>X</mi><mi>β</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span></span>
Only the last term depends on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>β</mi></mrow></math></span>, with a minus sign. So the maximum-likelihood <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>β</mi></mrow></math></span> is the least-squares <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>β</mi></mrow></math></span>. Module 12 returns to this.

## Uncertainty of the coefficients

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mover accent="true"><mi>β</mi><mo>^</mo></mover><mo stretchy="false">]</mo><mo>=</mo><mi>β</mi></mrow></math></span> (unbiased).
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Cov</mi><mo>⁡</mo><mo stretchy="false">(</mo><mover accent="true"><mi>β</mi><mo>^</mo></mover><mo stretchy="false">)</mo><mo>=</mo><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mover accent="true"><mi>σ</mi><mo>^</mo></mover><mn>2</mn></msup><mo>=</mo><mi mathvariant="normal">∥</mi><mi>y</mi><mo>−</mo><mi>X</mi><mover accent="true"><mi>β</mi><mo>^</mo></mover><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></math></span>. Divide by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>p</mi></mrow></math></span> because the residuals are forced to be perpendicular to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> columns.

## Collinearity

Small eigenvalues of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi></mrow></math></span> become large eigenvalues of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>: huge variance for some coefficient combinations.

```
rng = np.random.default_rng(0)
n = 200; x1 = rng.normal(size=n)
for noise in (1.0, 0.01):
    x2 = x1 + noise * rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    print(noise, np.round(np.sqrt(np.diag(np.linalg.inv(X.T @ X)))[1:], 3))
```

~~~dg-raw
%%viz f_bars v=0.098,7.091 l=x₂_independent,x₂_≈_copy_of_x₁ hi=none yl=standard_error_of_β₁_(per_unit_σ) | When x₂ is nearly a copy of x₁, the standard error of each coefficient grows about 70 times (0.098 to 7.09). The fit is fine; the split of credit between x₁ and x₂ is not.
~~~

## Reading the fit

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>R</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn><mo>−</mo><mfrac><mtext>SSE</mtext><mtext>SST</mtext></mfrac><mo separator="true">,</mo><mspace width="2em"/><mtext>SST</mtext><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>−</mo><mover accent="true"><mi>y</mi><mo>ˉ</mo></mover><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span></span>
SST is the error of "always predict the mean", so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>R</mi><mn>2</mn></msup></mrow></math></span> is the share of that error the model removes.

| Number | Tells you | Does not tell you |
| --- | --- | --- |
| training <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>R</mi><mn>2</mn></msup></mrow></math></span> | variance explained in-sample | anything about new data |
| validation MSE | error on similar new data | causation |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover><mi>j</mi></msub></mrow></math></span> | change in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover></mrow></math></span> per unit <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>j</mi></msub></mrow></math></span>, others fixed | importance, if scales differ |

<div class="take">
<div class="lab">Key takeaway</div>

Gaussian noise makes least squares the maximum-likelihood fit, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">(</mo><msup><mi>X</mi><mi mathvariant="normal">⊤</mi></msup><mi>X</mi><msup><mo stretchy="false">)</mo><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> says how uncertain the coefficients are.

</div>

## Where this breaks in practice

- Adding features never lowers training <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>R</mi><mn>2</mn></msup></mrow></math></span>, even pure noise ones. Compare models on held-out error.
- If the noise variance changes with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>β</mi><mo>^</mo></mover></mrow></math></span> stays unbiased but the textbook standard errors are wrong; use robust ("sandwich") standard errors.

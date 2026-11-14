---
id: "04.21"
course: "ml-maths-course"
module: "04"
slug: "numerical-rank"
title: "Numerical Rank"
keyIdeas: ["Measured data is never exactly low rank; noise makes every matrix full rank.", "Numerical rank at tolerance ⟦1575⟧ counts singular values above ⟦1575⟧.", "Look for a cliff in the singular values; cutting there gives the best low-rank approximation (Eckart-Young, module 07)."]
question: "A ⟦1574⟧ matrix of exact rank 3 has small noise added. What does `matrix_rank` report, and how would you find the \"real\" rank?"
answer: "Step 1, with noise, all 100 singular values are non-zero, so `matrix_rank` with its default tolerance reports 100. Step 2, plot or print the singular values: three are large, then they drop off a cliff to the noise level. Step 3, so the honest answer is \"rank 3 plus noise\". Choose the tolerance at the cliff."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Dimension, rank and nullity"]
tools: ["NumPy"]
topic: {"id": "04.04", "title": "Dimension, rank and nullity"}
---

## The big picture

Exact rank is a fragile number: the tiniest noise changes it. What you really want to know is how many directions carry real signal. Singular values answer that, and reading their shape is a core skill for PCA, compression and LoRA.

## Exact rank is fragile

```
rng = np.random.default_rng(0)
A = rng.normal(size=(100, 3)) @ rng.normal(size=(3, 100))
N = A + 0.01 * rng.normal(size=(100, 100))
s = np.linalg.svd(N, compute_uv=False)
print(np.linalg.matrix_rank(N))  # 100
print((s > 1).sum())  # 3
```

~~~dg-raw
%%viz f_bars v=112.7,98.2,84.4,0.2,0.18,0.18,0.18,0.17 l=1,2,3,4,5,6,7,8 hi=none yl=singular_value | The first eight singular values of the noisy matrix (rounded). Three large ones, then a cliff to the noise level: numerical rank 3.
~~~

## Numerical rank

The **numerical rank** at tolerance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>τ</mi></mrow></math></span> is the number of singular values above <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>τ</mi></mrow></math></span>. Choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>τ</mi></mrow></math></span> where the singular values drop off a cliff, or relative to the largest one (for example <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>τ</mi><mo>=</mo><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup><msub><mi>σ</mi><mn>1</mn></msub></mrow></math></span>).

~~~dg-flow Exact rank is fragile; numerical rank asks how many directions carry real signal.
Data matrix -(SVD)-> Singular values -(threshold tau)-> *Numerical rank k
~~~

~~~dg-raw
%%viz lowrank live | Rebuild a small image from its top k singular values. Add noise to see why rank needs a tolerance.
~~~

## Effective rank

Sometimes there is no clean cliff. The **effective rank**, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>exp</mi><mo>⁡</mo></mrow></math></span> of the entropy of the normalised singular values, is a smooth summary: it equals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> singular values are equal and the rest zero.

```
def eff_rank(s):
    p = s / s.sum()
    return np.exp(-(p * np.log(p)).sum())

print(round(eff_rank(np.array([1.0, 1, 1, 1e-12])), 3))  # 3.0
print(round(eff_rank(np.array([4.0, 2, 1, 0.5])), 3))  # 3.117
```

## Where this breaks in practice

- `matrix_rank` with its default tolerance can call a noisy but clearly rank-3 matrix "rank 100". Look at the singular values before trusting the number.

<div class="take">
<div class="lab">Key takeaway</div>

count singular values above a sensible tolerance. A cliff in the singular values marks the real rank.

</div>

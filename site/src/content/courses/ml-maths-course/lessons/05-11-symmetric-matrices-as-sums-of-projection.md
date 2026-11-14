---
id: "05.11"
course: "ml-maths-course"
module: "05"
slug: "symmetric-matrices-as-sums-of-projections"
title: "Symmetric Matrices as Sums of Projections"
keyIdeas: ["The spectral theorem can be written ⟦1749⟧.", "Each ⟦1750⟧ projects onto the direction ⟦1751⟧; ⟦8⟧ projects onto each eigen-direction, scales, and adds.", "Dropping the terms with small ⟦1736⟧ gives a low-rank approximation."]
question: "Rebuild ⟦1483⟧ from ⟦1746⟧, ⟦1747⟧ and ⟦1748⟧, ⟦1682⟧."
answer: "Step 1, ⟦1755⟧, ⟦1756⟧. Step 2, ⟦1757⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Spectral decomposition"]
tools: ["NumPy"]
topic: {"id": "05.02", "title": "Spectral decomposition"}
---

## The big picture

The same theorem has a second form that is often more useful: a symmetric matrix is a weighted sum of simple projections, one per eigen-direction. Keeping only the largest pieces gives the best low-rank approximation, which is PCA.

## The sum form

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>λ</mi><mi>i</mi></msub><mtext> </mtext><msub><mi>q</mi><mi>i</mi></msub><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span></span>
Each <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mi>i</mi></msub><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span> is a rank-1 matrix that **projects** onto <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mi>i</mi></msub></mrow></math></span>: it keeps the part of a vector along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mi>i</mi></msub></mrow></math></span> and drops the rest. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><msub><mi>q</mi><mi>i</mi></msub><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mo stretchy="false">)</mo><mi>x</mi><mo>=</mo><mo stretchy="false">(</mo><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>x</mi><mo stretchy="false">)</mo><msub><mi>q</mi><mi>i</mi></msub></mrow></math></span>.

So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi></mrow></math></span> = "for each eigen-direction, take the part of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> along it, scale by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>i</mi></msub></mrow></math></span>, add up".

```
q1 = np.array([1, 1]) / np.sqrt(2); q2 = np.array([1, -1]) / np.sqrt(2)
print(3 * np.outer(q1, q1) + 1 * np.outer(q2, q2))
```

~~~dg-layers A symmetric matrix decomposes into perpendicular directions, each scaled by its eigenvalue.
*Largest eigenvalue direction | dominant stretch; top principal component of a covariance
Middle directions | moderate stretch
Smallest eigenvalue direction | weakest stretch; flattest direction of a Hessian
~~~

## Truncating

Sort the eigenvalues by size and keep the top <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> terms:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>A</mi><mi>k</mi></msub><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>k</mi></munderover><msub><mi>λ</mi><mi>i</mi></msub><msub><mi>q</mi><mi>i</mi></msub><msubsup><mi>q</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span></span>
For a positive semidefinite matrix this is the best rank-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> approximation.

```
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) @ np.diag([5.0, 3, 0.3, 0.2, 0.1])
C = np.cov(X.T)
vals, Q = np.linalg.eigh(C)
top = Q[:, -2:] @ np.diag(vals[-2:]) @ Q[:, -2:].T
print(round(np.linalg.norm(C - top) / np.linalg.norm(C), 3) < 0.01)  # True
```

Two of five directions hold almost all of this covariance.

## In ML

- **PCA** keeps the top terms of the covariance.
- **Spectral clustering** uses eigenvectors of a graph Laplacian.
- **Kernel methods** approximate a kernel matrix by its top terms (Nyström).

<div class="take">
<div class="lab">Key takeaway</div>

a symmetric matrix is a sum of eigenvalue-weighted projections. Keep the big terms for a low-rank approximation.

</div>

---
id: "08.08"
course: "ml-maths-course"
module: "08"
slug: "computing-pca-with-the-svd"
title: "Computing PCA with the SVD"
keyIdeas: ["With ⟦2431⟧, the components are the columns of ⟦1204⟧ and the variances are ⟦2432⟧.", "Scores are ⟦2433⟧ with ⟦2434⟧; reconstruction is ⟦2435⟧.", "The mean squared reconstruction error per sample equals the sum of the discarded eigenvalues."]
question: "Data has covariance eigenvalues 3.5 and 0.5. Keeping one component, what is the average squared reconstruction error per sample?"
answer: "Step 1, the discarded eigenvalue is 0.5. Step 2, by the Pythagoras split of lesson 08.07, average squared error equals discarded variance: 0.5 (up to the ⟦233⟧ versus ⟦2438⟧ factor). Step 3, that is ⟦2439⟧ of the total."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Principal component analysis"]
tools: ["NumPy", "scikit-learn"]
topic: {"id": "08.03", "title": "Principal component analysis"}
---

## The big picture

You never need to form the covariance matrix. The SVD of the centred data gives the components and their variances directly, and more accurately.

## From SVD to PCA

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>X</mi><mi>c</mi></msub><mo>=</mo><mi>U</mi><mi mathvariant="normal">Σ</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup><mspace width="1em"/><mo>⟹</mo><mspace width="1em"/><mi>S</mi><mo>=</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></mfrac></mstyle><msubsup><mi>X</mi><mi>c</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>X</mi><mi>c</mi></msub><mo>=</mo><mi>V</mi><mtext> </mtext><mstyle displaystyle="false" scriptlevel="0"><mfrac><msup><mi mathvariant="normal">Σ</mi><mn>2</mn></msup><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></mfrac></mstyle><mtext> </mtext><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>
So the components are the columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> and the variances are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>i</mi></msub><mo>=</mo><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

~~~dg-flow PCA via SVD.
X -(subtract column means)-> X_c -(thin SVD)-> U S Vᵀ -(take V[:, :k])-> W -(Z = X_c W)-> *k-dimensional scores
~~~

~~~dg-raw
%%viz f_plane x=-3.5,3.5 y=-3.5,3.5 tk=1 pg=2.65:2.65_2.78:2.43_2.83:2.14_2.79:1.79_2.67:1.38_2.47:0.93_2.19:0.46_1.84:-0.03_1.44:-0.53_1.00:-1.00_0.53:-1.44_0.03:-1.84_-0.46:-2.19_-0.93:-2.47_-1.38:-2.67_-1.79:-2.79_-2.14:-2.83_-2.43:-2.78_-2.65:-2.65_-2.78:-2.43_-2.83:-2.14_-2.79:-1.79_-2.67:-1.38_-2.47:-0.93_-2.19:-0.46_-1.84:0.03_-1.44:0.53_-1.00:1.00_-0.53:1.44_-0.03:1.84_0.46:2.19_0.93:2.47_1.38:2.67_1.79:2.79_2.14:2.83_2.43:2.78,a v=1.32,1.32,PC1:_λ_=_3.5,f;-0.5,0.5,PC2:_λ_=_0.5,b | Data with covariance [[2, 1.5], [1.5, 2]]: the shaded ellipse holds about 86% of the points (2 standard deviations). PC1 points along the long axis (1, 1); PC2 along the short axis (1, −1). Arrow lengths are √λ, the standard deviation along each.
~~~

```
rng = np.random.default_rng(0)
X = rng.multivariate_normal([0, 0], [[2, 1.5], [1.5, 2]], size=20000)
Xc = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print(np.round(s ** 2 / (len(X) - 1), 1))  # [3.5 0.5]
print(np.round(np.abs(Vt[0]), 1))  # [0.7 0.7]
Z = Xc @ Vt[:1].T
Xhat = Z @ Vt[:1]
print(round(((Xc - Xhat) ** 2).sum(1).mean(), 1))  # 0.5
```

The reconstruction error with one component is the discarded eigenvalue, 0.5. Signs of components are arbitrary, hence the `abs`.

## In scikit-learn

`sklearn.decomposition.PCA` does exactly this. It stores `components_` as rows (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>W</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>) and `explained_variance_` as the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>i</mi></msub></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

PCA is the SVD of the centred data; the right singular vectors are the components and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mi>n</mi><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> their variances.

</div>

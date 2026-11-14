---
id: "08.07"
course: "ml-maths-course"
module: "08"
slug: "principal-components-as-directions-of-maximum-variance"
title: "Principal Components as Directions of Maximum Variance"
keyIdeas: ["The first principal component maximises ⟦2404⟧ over unit vectors; it is the top eigenvector of ⟦385⟧, and the variance it captures is the top eigenvalue.", "Each next component is the best direction perpendicular to the earlier ones: the next eigenvector.", "Maximising variance kept is the same as minimising squared reconstruction error."]
question: "⟦2418⟧. Find the principal components and the share of variance the first one explains."
answer: "Step 1, trace 4, determinant ⟦2425⟧, so ⟦2426⟧ and ⟦2427⟧. Step 2, eigenvectors ⟦2428⟧ for 3.5 and ⟦2429⟧ for 0.5. Step 3, the first explains ⟦2430⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Principal component analysis"]
tools: ["NumPy"]
topic: {"id": "08.03", "title": "Principal component analysis"}
---

## The big picture

Most data clouds are long in a few directions and thin in the rest. **Principal component analysis** (PCA) finds the long directions. Keep only those and you can plot, compress or denoise the data while losing as little as possible.

## The optimisation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><munder><mrow><mi>max</mi><mo>⁡</mo></mrow><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo>=</mo><mn>1</mn></mrow></munder><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>S</mi><mi>w</mi></mrow></math></span></span>
In the eigenbasis, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>S</mi><mi>w</mi><mo>=</mo><msub><mo>∑</mo><mi>i</mi></msub><msub><mi>λ</mi><mi>i</mi></msub><msubsup><mi>c</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∑</mo><msubsup><mi>c</mi><mi>i</mi><mn>2</mn></msubsup><mo>=</mo><mn>1</mn></mrow></math></span>. This is largest when all the weight goes on the biggest <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span>. So:

- PC1 = top eigenvector of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>, variance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mn>1</mn></msub></mrow></math></span>.
- PC2 = second eigenvector, the best direction perpendicular to PC1, variance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mn>2</mn></msub></mrow></math></span>. And so on.

~~~dg-raw
%%viz pca live | Turn your direction and watch the variance of the projected points. PC1 is where it peaks.
~~~

## Two views, one answer

- **Max variance**: keep the directions where the data spreads most.
- **Min reconstruction error**: among <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span>-dimensional subspaces, the top-<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> one makes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∑</mo><mi>i</mi></msub><mi mathvariant="normal">∥</mi><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><msub><mover accent="true"><mi>x</mi><mo>^</mo></mover><mi>i</mi></msub><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> smallest.
By Pythagoras, variance kept plus reconstruction error equals the total variance, which is fixed. So maximising one minimises the other.

```
S = np.array([[2.0, 1.5], [1.5, 2]])
lam, V = np.linalg.eigh(S)
print(lam[::-1], np.round(np.abs(V[:, -1]), 4))  # [3.5 0.5] [0.7071 0.7071]
print(lam[-1] / lam.sum())  # 0.875
```

<div class="take">
<div class="lab">Key takeaway</div>

principal components are the eigenvectors of the covariance matrix, ranked by eigenvalue; each eigenvalue is the variance along its component.

</div>

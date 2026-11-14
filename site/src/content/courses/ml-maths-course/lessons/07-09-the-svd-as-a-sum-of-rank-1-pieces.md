---
id: "07.09"
course: "ml-maths-course"
module: "07"
slug: "the-svd-as-a-sum-of-rank-1-pieces"
title: "The SVD as a Sum of Rank-1 Pieces"
keyIdeas: ["⟦2271⟧: a sum of rank-1 matrices ordered by importance.", "The rank ⟦118⟧ is the number of non-zero singular values.", "⟦2272⟧, so ⟦2273⟧ is piece ⟦274⟧'s share of the \"energy\"."]
question: "A matrix has singular values ⟦2269⟧. What is its rank, what share of ⟦2270⟧ does the first piece carry, and the first two together?"
answer: "Step 1, all four are non-zero, so rank 4. Step 2, ⟦2279⟧. Step 3, first piece: ⟦2280⟧. Step 4, first two: ⟦2281⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Singular value decomposition"]
tools: ["NumPy"]
topic: {"id": "07.02", "title": "Singular value decomposition"}
---

## The big picture

Multiplying out <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi><mi mathvariant="normal">Σ</mi><msup><mi>V</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span> column by column turns the SVD into a list of simple layers, each one an outer product scaled by a singular value. Sorting by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span> sorts them by importance, which is what makes compression possible.

## Outer-product form

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>=</mo><msub><mi>σ</mi><mn>1</mn></msub><msub><mi>u</mi><mn>1</mn></msub><msubsup><mi>v</mi><mn>1</mn><mi mathvariant="normal">⊤</mi></msubsup><mo>+</mo><msub><mi>σ</mi><mn>2</mn></msub><msub><mi>u</mi><mn>2</mn></msub><msubsup><mi>v</mi><mn>2</mn><mi mathvariant="normal">⊤</mi></msubsup><mo>+</mo><mo>⋯</mo><mo>+</mo><msub><mi>σ</mi><mi>r</mi></msub><msub><mi>u</mi><mi>r</mi></msub><msubsup><mi>v</mi><mi>r</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span></span>
Each <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mi>i</mi></msub><msubsup><mi>v</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span> is a rank-1 matrix: every row is a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>v</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup></mrow></math></span> and every column a multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>u</mi><mi>i</mi></msub></mrow></math></span>.

```
A = np.array([[3.0, 0], [4, 5]])
U, s, Vt = np.linalg.svd(A)
pieces = [s[i] * np.outer(U[:, i], Vt[i]) for i in range(2)]
print(np.allclose(sum(pieces), A), [int(np.linalg.matrix_rank(P)) for P in pieces])  # True [1, 1]
```

## Energy

The squared Frobenius norm (sum of squared entries) equals the sum of squared singular values:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><msubsup><mi mathvariant="normal">∥</mi><mi>F</mi><mn>2</mn></msubsup><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup></mrow></math></span></span>

~~~dg-raw
%%viz f_bars v=79.4,19.8,0.8,0.01 l=σ₁=10,σ₂=5,σ₃=1,σ₄=0.1 hi=none yl=share_of_‖A‖²_(%) | Share of the total energy carried by each rank-1 piece when σ = 10, 5, 1, 0.1. The first two pieces carry 99.2%.
~~~

```
s = np.array([10, 5, 1, 0.1])
print(np.round(100 * np.cumsum(s ** 2) / (s ** 2).sum(), 1))  # [ 79.4  99.2 100.  100. ]
```

<div class="take">
<div class="lab">Key takeaway</div>

the SVD is a sum of rank-1 layers sorted by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span>; squared singular values say how much of the matrix each layer carries.

</div>

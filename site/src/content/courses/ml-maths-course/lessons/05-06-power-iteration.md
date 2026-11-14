---
id: "05.06"
course: "ml-maths-course"
module: "05"
slug: "power-iteration"
title: "Power Iteration"
keyIdeas: ["Power iteration: multiply a random vector by ⟦8⟧, rescale to length 1, repeat. It converges to the top eigenvector.", "The error shrinks by ⟦1683⟧ per step.", "⟦1684⟧ for a unit ⟦776⟧ estimates the top eigenvalue."]
question: "Power iteration on a matrix with ⟦1680⟧ and ⟦1681⟧. Roughly how many steps to shrink the error by a factor of 1,000? What if ⟦1682⟧?"
answer: "Step 1, the error shrinks by ⟦1687⟧ per step. Step 2, ⟦1688⟧ needs ⟦1689⟧ steps. Step 3, with ⟦1682⟧ the ratio is 0.1, so 3 steps."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
---

## The big picture

The previous lesson showed that repeated multiplication lines a vector up with the top eigenvector. Turned into an algorithm, that is power iteration. It needs only matrix-vector products, so it works on matrices far too big to store, such as the web graph in PageRank.

## The algorithm

1. Start with a random vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>.
2. Replace <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>v</mi></mrow></math></span>, then divide by its length.
3. Repeat until <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> stops changing.

~~~dg-raw
%%viz f_plane x=-0.2,1.2 y=-0.2,1.2 tk=0.5 v=1,0,0,m;0.894,0.447,1,m;0.781,0.625,2,m;0.733,0.680,3,a ln=1,-1,0,top_eigenvector,f,0.9 | Power iteration on A = [[2, 1], [1, 2]] from (1, 0). After three steps the unit vector already lies almost on the top eigenvector direction (1, 1).
~~~

```
A = np.array([[2.0, 1], [1, 2]])
v = np.array([1.0, 0.3])
for _ in range(20):
    v = A @ v; v /= np.linalg.norm(v)
print(np.round(v, 4), round(v @ A @ v, 4))  # [0.7071 0.7071] 3.0
```

The final <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>v</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mi>v</mi></mrow></math></span> is the estimated top eigenvalue (it works because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> has length 1).

## Speed

After <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> steps, the leftover from the second direction is about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><msub><mi>λ</mi><mn>2</mn></msub><mi mathvariant="normal">/</mi><msub><mi>λ</mi><mn>1</mn></msub><msup><mi mathvariant="normal">∣</mi><mi>k</mi></msup></mrow></math></span>. Close eigenvalues mean slow convergence.

<div class="ex">
<div class="lab">Worked example</div>

**two gaps**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mn>1</mn></msub><mo>=</mo><mn>10</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mn>2</mn></msub><mo>=</mo><mn>9</mn></mrow></math></span>: ratio 0.9, about 66 steps for 3 digits.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mn>1</mn></msub><mo>=</mo><mn>10</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mn>2</mn></msub><mo>=</mo><mn>1</mn></mrow></math></span>: ratio 0.1, 3 steps for 3 digits.

</div>

## Where it is used

- **PageRank** finds the top eigenvector of a huge link matrix.
- **Spectral normalisation** in GANs runs one power step per training step to estimate the largest singular value of each weight matrix.
- **Hessian top eigenvalue** for learning-rate analysis, using Hessian-vector products.

<div class="take">
<div class="lab">Key takeaway</div>

multiply and normalise, repeatedly. It finds the top eigenvector, fast when the top eigenvalue is well separated.

</div>

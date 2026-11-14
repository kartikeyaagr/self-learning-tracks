---
id: "07.12"
course: "ml-maths-course"
module: "07"
slug: "the-pseudoinverse-and-other-svd-tools"
title: "The Pseudoinverse and Other SVD Tools"
keyIdeas: ["The pseudoinverse ⟦2308⟧ inverts the non-zero singular values and leaves zeros alone.", "⟦2307⟧ is the least-squares solution of ⟦816⟧, and the shortest one when there are many.", "The SVD also gives rank, norms, condition number and bases for the four fundamental subspaces."]
question: "⟦2305⟧ and ⟦2003⟧. All solutions of ⟦816⟧ satisfy ⟦2306⟧. Which one does ⟦2307⟧ return?"
answer: "Step 1, ⟦8⟧ has rank 1, so there is a line of solutions. Step 2, the shortest point on the line ⟦2306⟧ is its projection of the origin: along the normal ⟦1049⟧, ⟦2316⟧. Step 3, that is what ⟦2307⟧ gives."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Singular value decomposition"]
tools: ["NumPy"]
topic: {"id": "07.02", "title": "Singular value decomposition"}
---

## The big picture

Many matrices have no inverse: they are rectangular or singular. The SVD gives the next best thing, plus a one-stop summary of everything else about the matrix.

## The pseudoinverse

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mo>+</mo></msup><mo>=</mo><mi>V</mi><msup><mi mathvariant="normal">Σ</mi><mo>+</mo></msup><msup><mi>U</mi><mi mathvariant="normal">⊤</mi></msup></mrow></math></span></span>
where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">Σ</mi><mo>+</mo></msup></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> transposed with each non-zero <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span> replaced by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><msup><mi>A</mi><mo>+</mo></msup><mi>b</mi></mrow></math></span>:

- minimises <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi><mi mathvariant="normal">∥</mi></mrow></math></span> (least squares), and
- among all minimisers, has the smallest <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi></mrow></math></span>.

```
A = np.array([[1.0, 2], [2, 4], [3, 6]])
print(np.round(np.linalg.pinv(A) @ np.array([1.0, 2, 3]), 4))  # [0.2 0.4]
```

~~~dg-raw
%%viz f_plane x=-0.5,1.5 y=-0.5,1 tk=0.5 ln=1,2,1,solutions:_x₁+2x₂=1,a,0.15 v=0.2,0.4,A⁺b,f | Every point on the green line solves Ax = b. The pseudoinverse picks the one closest to the origin, (0.2, 0.4), where the line meets its perpendicular.
~~~

## The SVD as a summary

| Quantity | From the SVD |
| --- | --- |
| numerical rank | number of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span> above a tolerance |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><msub><mi mathvariant="normal">∥</mi><mn>2</mn></msub></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>A</mi><msub><mi mathvariant="normal">∥</mi><mi>F</mi></msub></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mrow><mo>∑</mo><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup></mrow></msqrt></mrow></math></span> |
| condition number | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>σ</mi><mn>1</mn></msub><mi mathvariant="normal">/</mi><msub><mi>σ</mi><mi>n</mi></msub></mrow></math></span> |
| column space | first <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span> columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> |
| null space | last <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mi>r</mi></mrow></math></span> columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi></mrow></math></span> |

<div class="take">
<div class="lab">Key takeaway</div>

the pseudoinverse inverts the non-zero singular values; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mo>+</mo></msup><mi>b</mi></mrow></math></span> is the shortest least-squares solution.

</div>

## Where this breaks in practice

- Tiny singular values make <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msub><mi>σ</mi><mi>i</mi></msub></mrow></math></span> huge and amplify noise. `np.linalg.pinv` zeroes those below a tolerance (`rcond`); ridge regression shrinks them smoothly instead.
- A full SVD costs about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mi>n</mi><mi>min</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>m</mi><mo separator="true">,</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></math></span> operations. For the top <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span> of a big matrix use a randomized SVD.
- Singular vectors have arbitrary signs: fix a sign convention before comparing runs.

---
id: "03.27"
course: "ml-maths-course"
module: "03"
slug: "solving-systems-in-numpy"
title: "Solving Systems in NumPy"
keyIdeas: ["Use `np.linalg.solve(A, b)` for square invertible systems, never `np.linalg.inv(A) @ b`.", "Use `np.linalg.lstsq` for non-square or singular systems; it returns the least-squares, minimum-norm solution.", "A small residual ⟦1115⟧ does not guarantee an accurate ⟦7⟧ when ⟦8⟧ is badly conditioned."]
question: "Which call would you use for (a) a ⟦804⟧ invertible system, (b) 100 equations in 3 unknowns, (c) 2 equations in 5 unknowns?"
answer: "Step 1, (a) `np.linalg.solve`, which uses LU with pivoting. Step 2, (b) `np.linalg.lstsq`: there is generally no exact solution, and it returns the least-squares fit. Step 3, (c) `np.linalg.lstsq` as well: there are infinitely many solutions, and it returns the one with the smallest norm."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

You will rarely write elimination yourself. You will call a library, and the main skill is picking the right call and knowing what it does when the system is not nice.

## The calls

| situation | call | what you get |
| --- | --- | --- |
| square, invertible | `np.linalg.solve(A, b)` | the unique solution, via LU with pivoting |
| symmetric positive definite | `scipy.linalg.cho_solve` | same, via Cholesky, twice as fast |
| more equations than unknowns | `np.linalg.lstsq(A, b)` | the least-squares fit |
| fewer equations than unknowns | `np.linalg.lstsq(A, b)` | the solution of smallest length |

```
A = np.array([[2.0, 1], [1, 3]]); b = np.array([3.0, 5])
print(np.round(np.linalg.solve(A, b), 6))  # [0.8 1.4]
X = np.array([[1.0, 0], [1, 1], [1, 2]]); y = np.array([1.0, 2, 2])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
print(np.round(coef, 4))  # [1.1667 0.5   ]
```

The second call fits the line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mi>a</mi><mo>+</mo><mi>b</mi><mi>x</mi></mrow></math></span> through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> from lesson 03.23.

## Why not the inverse?

`np.linalg.inv(A) @ b` costs about three times as much as `solve` and is usually less accurate. You almost never need <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> itself, only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi>b</mi></mrow></math></span>.

## Checking a result

The residual <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>=</mo><mi>A</mi><mi>x</mi><mo>−</mo><mi>b</mi></mrow></math></span> measures how well <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> satisfies the equations:

```
A = np.array([[1, 1], [1, 1.0001]]); b = np.array([2, 2.0001])
x = np.linalg.solve(A, b)
print(np.abs(A @ x - b).max() < 1e-12)  # True
```

A tiny residual is necessary, but not sufficient: for a badly conditioned <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, very different <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> can have tiny residuals (lesson 03.31).

<div class="take">
<div class="lab">Key takeaway</div>

`solve` for square systems, `lstsq` for everything else, and no explicit inverses.

</div>

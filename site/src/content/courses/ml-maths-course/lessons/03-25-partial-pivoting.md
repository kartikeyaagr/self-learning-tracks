---
id: "03.25"
course: "ml-maths-course"
module: "03"
slug: "partial-pivoting"
title: "Partial Pivoting"
keyIdeas: ["Elimination divides by pivots. A zero pivot stops it; a tiny pivot silently amplifies rounding errors.", "Partial pivoting swaps up the row with the largest absolute entry in the current column, so every multiplier is at most 1.", "`np.linalg.solve` always pivots."]
question: "Solve ⟦1081⟧, ⟦1042⟧ with and without pivoting, in floating point."
answer: "Step 1, without pivoting the multiplier is ⟦1083⟧. Row 2 becomes ⟦1084⟧, which rounds to ⟦1087⟧, so ⟦831⟧. Then ⟦1088⟧, so ⟦387⟧. Wrong. Step 2, with a swap the pivot is 1 and the multiplier ⟦1082⟧. Row 2 becomes ⟦1089⟧, so ⟦1090⟧ and ⟦1091⟧. Correct."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

On paper, any non-zero pivot works. On a computer, numbers carry about 16 significant digits, and dividing by a tiny pivot can wipe all of them out. A simple row swap fixes this, and it is used in every serious solver.

## The problem with small pivots

<div class="ex">
<div class="lab">Worked example</div>

**why pivots must be large**

System: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>20</mn></mrow></msup><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>2</mn></mrow></math></span>. The true solution is very close to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

Without swapping, the pivot is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>20</mn></mrow></msup></mrow></math></span> and the multiplier is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mn>20</mn></msup></mrow></math></span>. Row 2 becomes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mn>1</mn><msup><mn>0</mn><mn>20</mn></msup><mo stretchy="false">)</mo><mi>y</mi><mo>=</mo><mn>2</mn><mo>−</mo><mn>1</mn><msup><mn>0</mn><mn>20</mn></msup></mrow></math></span>. In floating point both sides round to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn><msup><mn>0</mn><mn>20</mn></msup></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

Back-substituting: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><msup><mn>0</mn><mrow><mo>−</mo><mn>20</mn></mrow></msup><mi>x</mi><mo>=</mo><mn>1</mn><mo>−</mo><mn>1</mn><mo>=</mo><mn>0</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>. Completely wrong.

</div>

```
A = np.array([[1e-20, 1.0], [1.0, 1.0]]); b = np.array([1.0, 2.0])
m = A[1, 0] / A[0, 0]
y = (b[1] - m * b[0]) / (A[1, 1] - m * A[0, 1])
x = (b[0] - A[0, 1] * y) / A[0, 0]
print(x, y)  # 0.0 1.0
print(np.linalg.solve(A, b))  # [1. 1.]
```

## The fix

**Partial pivoting:** in each column, swap up the row whose entry has the biggest absolute value, then eliminate. Every multiplier is then at most 1 in size, so errors are never multiplied up.

## Elimination with pivoting

```
def solve_pp(A, b):
    A = A.astype(float).copy(); b = b.astype(float).copy(); n = len(b)
    for j in range(n):
        p = j + np.argmax(abs(A[j:, j]))
        A[[j, p]] = A[[p, j]]; b[[j, p]] = b[[p, j]]
        for i in range(j + 1, n):
            m = A[i, j] / A[j, j]
            A[i] -= m * A[j]; b[i] -= m * b[j]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x

print(np.round(solve_pp(np.array([[1e-20, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0])), 6))  # [1. 1.]
```

## Where this breaks in practice

- **Pivoting does not fix a bad problem.** If the matrix itself is nearly singular, the answer is sensitive whatever the algorithm does (lesson 03.31). Pivoting only stops the algorithm from making things worse.

<div class="take">
<div class="lab">Key takeaway</div>

always pivot on the largest entry in the column. It keeps multipliers small and rounding errors under control.

</div>

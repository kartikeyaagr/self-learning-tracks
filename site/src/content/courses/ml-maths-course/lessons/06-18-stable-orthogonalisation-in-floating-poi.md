---
id: "06.18"
course: "ml-maths-course"
module: "06"
slug: "stable-orthogonalisation-in-floating-point"
title: "Stable Orthogonalisation in Floating Point"
keyIdeas: ["Classical Gram-Schmidt loses orthogonality in floating point when the columns are nearly dependent.", "Modified Gram-Schmidt subtracts each projection from the running vector and does much better.", "Householder QR (used by `np.linalg.qr`) keeps ⟦1731⟧ to machine precision regardless of conditioning."]
question: "Classical Gram-Schmidt on a matrix with ⟦2109⟧ in float64 can give ⟦2110⟧ near what size? And Householder QR?"
answer: "Step 1, classical Gram-Schmidt's loss of orthogonality grows with roughly ⟦2115⟧ where ⟦2116⟧, so with ⟦2117⟧ it can reach order 1: orthogonality is lost completely. Step 2, Householder QR stays near ⟦2116⟧ whatever ⟦1792⟧ is."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Projection and orthonormalization"]
tools: ["NumPy"]
topic: {"id": "06.03", "title": "Projection and orthonormalization"}
---

## The big picture

Gram-Schmidt is correct in exact arithmetic. In floating point the order of operations matters a great deal. This lesson shows the failure and the standard fixes.

## Classical versus modified

- **Classical**: compute every coefficient <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>q</mi><mi>j</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>a</mi><mi>k</mi></msub></mrow></math></span> from the original <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>k</mi></msub></mrow></math></span>, then subtract all at once.
- **Modified**: subtract each projection from the running <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> before computing the next coefficient, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>q</mi><mi>j</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>v</mi></mrow></math></span>.
In exact arithmetic they agree. In floating point the modified version corrects for errors made earlier in the same step.

```
def cgs(A):
    Q = np.zeros_like(A)
    for k in range(A.shape[1]):
        v = A[:, k] - Q[:, :k] @ (Q[:, :k].T @ A[:, k])
        Q[:, k] = v / np.linalg.norm(v)
    return Q

def mgs(A):
    Q = A.copy()
    for k in range(A.shape[1]):
        for j in range(k):
            Q[:, k] -= (Q[:, j] @ Q[:, k]) * Q[:, j]
        Q[:, k] /= np.linalg.norm(Q[:, k])
    return Q

n = 10
A = 1 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0)
err = lambda Q: np.abs(Q.T @ Q - np.eye(n)).max()
print(f"{np.linalg.cond(A):.0e}")  # 2e+13
print(err(cgs(A)) > 0.1, err(mgs(A)) < 1e-2, err(np.linalg.qr(A)[0]) < 1e-14)  # True True True
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>10</mn><mo>×</mo><mn>10</mn></mrow></math></span> Hilbert matrix, famously ill-conditioned. Classical Gram-Schmidt loses orthogonality badly; modified is far better; Householder QR is orthogonal to machine precision.

~~~dg-raw
%%viz f_bars v=0,3.8,15.1 l=classical_GS,modified_GS,Householder hi=none yl=correct_digits_of_orthogonality | On the 10 × 10 Hilbert matrix: classical Gram-Schmidt keeps no correct digits of orthogonality (error about 1), modified keeps about 4 (error 2 × 10⁻⁴), Householder about 15 (error 9 × 10⁻¹⁶).
~~~

## Householder QR

Instead of building <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> column by column, Householder QR applies reflections <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>H</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>H</mi><mn>2</mn></msub><mo separator="true">,</mo><mo>…</mo></mrow></math></span> that zero out the entries below the diagonal of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, one column at a time. Each reflection is exactly orthogonal, so their product is too. This is what LAPACK and `np.linalg.qr` use.

<div class="take">
<div class="lab">Key takeaway</div>

in code, use `np.linalg.qr` (Householder); write Gram-Schmidt yourself only in its modified form and only for well-conditioned inputs.

</div>

## Where this breaks in practice

- Rotation matrices built by multiplying many small rotations drift away from being orthogonal. Re-orthogonalise every so often with QR.
- Iterative methods like Lanczos (used for Hessian top-eigenvalue estimates) lose orthogonality among their vectors and need periodic re-orthogonalisation.

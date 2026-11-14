---
id: "06.16"
course: "ml-maths-course"
module: "06"
slug: "the-gram-schmidt-process"
title: "The Gram-Schmidt Process"
keyIdeas: ["Gram-Schmidt turns independent vectors ⟦2068⟧ into orthonormal ⟦1936⟧ with the same span.", "Step ⟦1199⟧: subtract from ⟦2069⟧ its projections onto ⟦2070⟧, then normalise.", "What is left at each step is the genuinely new direction in ⟦2069⟧."]
question: "Apply Gram-Schmidt to ⟦2066⟧ and ⟦2067⟧."
answer: "Step 1, ⟦2087⟧. Step 2, ⟦2088⟧, so the part of ⟦2089⟧ along ⟦2090⟧ is ⟦2091⟧. Step 3, ⟦2092⟧, with length ⟦2093⟧. Step 4, ⟦2094⟧. Check ⟦2095⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Projection and orthonormalization"]
tools: ["NumPy"]
topic: {"id": "06.03", "title": "Projection and orthonormalization"}
---

## The big picture

Orthonormal bases make projection easy (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi><msup><mi>Q</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span>) and coordinates free. Gram-Schmidt builds one from any independent set, one vector at a time.

## The algorithm

1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>=</mo><msub><mi>a</mi><mn>1</mn></msub></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mn>1</mn></msub><mo>=</mo><msub><mi>v</mi><mn>1</mn></msub><mi mathvariant="normal">/</mi><mi mathvariant="normal">∥</mi><msub><mi>v</mi><mn>1</mn></msub><mi mathvariant="normal">∥</mi></mrow></math></span>.
2. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>=</mo><mn>2</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mi>n</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>k</mi></msub><mo>=</mo><msub><mi>a</mi><mi>k</mi></msub><mo>−</mo><msub><mo>∑</mo><mrow><mi>j</mi><mo>&lt;</mo><mi>k</mi></mrow></msub><mo stretchy="false">(</mo><msubsup><mi>q</mi><mi>j</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>a</mi><mi>k</mi></msub><mo stretchy="false">)</mo><mtext> </mtext><msub><mi>q</mi><mi>j</mi></msub></mrow></math></span>. Remove every direction already covered.
3. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mi>k</mi></msub><mo>=</mo><msub><mi>v</mi><mi>k</mi></msub><mi mathvariant="normal">/</mi><mi mathvariant="normal">∥</mi><msub><mi>v</mi><mi>k</mi></msub><mi mathvariant="normal">∥</mi></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-1,4 y=-1,3 tk=1 v=2,0,a₁,m;1,2,a₂,k;1,0,q₁,a;0,1,q₂,a va=0,0,1,0,,m sg=1,0,1,2,f,d | a₁ = (2, 0) gives q₁ = (1, 0). a₂ = (1, 2) has component 1 along q₁; subtracting it leaves (0, 2) (orange dashed), which normalises to q₂ = (0, 1).
~~~

<div class="ex">
<div class="lab">Worked example</div>

**in 2D**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mn>1</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>q</mi><mn>1</mn><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>a</mi><mn>2</mn></msub><mo>=</mo><mn>1</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo>−</mo><mn>1</mn><mo>⋅</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>q</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=float)
    for k in range(A.shape[1]):
        v = A[:, k].astype(float)
        for j in range(k):
            v = v - (Q[:, j] @ v) * Q[:, j]
        Q[:, k] = v / np.linalg.norm(v)
    return Q

A = np.array([[1, 1], [1, 0], [0, 1]])
Q = gram_schmidt(A)
print(np.round(Q * np.sqrt(6), 4))
print(np.allclose(Q.T @ Q, np.eye(2)))  # True
```

The second column printed is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, matching the hand calculation; the first is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>6</mn></msqrt><mtext> </mtext><msub><mi>q</mi><mn>1</mn></msub><mo>=</mo><msqrt><mn>3</mn></msqrt><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

## If a vector is dependent

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>k</mi></msub></mrow></math></span> lies in the span of the earlier vectors, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>v</mi><mi>k</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span> and you cannot normalise. In exact arithmetic that signals dependence; in floating point a tiny <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><msub><mi>v</mi><mi>k</mi></msub><mi mathvariant="normal">∥</mi></mrow></math></span> signals near-dependence.

<div class="take">
<div class="lab">Key takeaway</div>

Gram-Schmidt subtracts projections onto the directions already found, then normalises what is left.

</div>

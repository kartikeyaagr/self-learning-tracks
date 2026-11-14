---
id: "05.01"
course: "ml-maths-course"
module: "05"
slug: "eigenvalues-and-eigenvectors-for-2x2-matrices"
title: "Eigenvalues and Eigenvectors for 2x2 Matrices"
keyIdeas: ["An eigenvector ⟦1578⟧ of a square matrix ⟦8⟧ satisfies ⟦1579⟧: ⟦8⟧ only stretches it, without turning it. ⟦559⟧ is the eigenvalue.", "Any non-zero multiple of an eigenvector is also one, so an eigenvector is really a direction.", "To test a candidate, multiply and check whether the result is a multiple of it."]
question: "For ⟦1483⟧, which of ⟦111⟧, ⟦1352⟧, ⟦310⟧ are eigenvectors, and with which eigenvalues?"
answer: "Step 1, ⟦1585⟧: eigenvector, ⟦1586⟧. Step 2, ⟦1587⟧: eigenvector, ⟦1588⟧. Step 3, ⟦1589⟧, which is not a multiple of ⟦310⟧: not an eigenvector."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.2-4.4"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 5, 7"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 14"}]
tags: ["Invariant directions and spectra"]
tools: ["NumPy"]
topic: {"id": "05.01", "title": "Invariant directions and spectra"}
steps: [{"kind": "explain", "title": "Stretch without turning", "body": "Take $A = \\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}$. Most vectors get turned: $A(1, 0) = (2, 1)$ points a new way. But $A(1, 1) = (3, 3)$ points exactly along $(1, 1)$, just 3 times longer.\n\nA non-zero vector $v$ that the matrix only stretches is an **eigenvector**:\n\n$$Av = \\lambda v$$\n\nThe stretch factor $\\lambda$ (\"lambda\") is its **eigenvalue**. \"Eigen\" is German for \"own\": these are the matrix's own directions.", "question": {"q": "What is the eigenvalue that goes with $v = (1, 1)$?", "options": ["3", "(3, 3)", "2"], "answer": 0, "why": "$A(1, 1) = (3, 3) = 3 \\cdot (1, 1)$, so $\\lambda = 3$. The eigenvalue is the number, and $(3, 3)$ is the vector $Av$.", "hint": "Write $Av$ as some number times $v$. That number is the answer."}}, {"kind": "explain", "title": "Testing a candidate", "body": "To test whether $v$ is an eigenvector, multiply and look: is $Av$ a number times $v$?\n\nWith the same $A$, try $v = (1, 0)$. $Av = (2, 1)$, and no single number turns $(1, 0)$ into $(2, 1)$, because the second entry would need $\\lambda \\cdot 0 = 1$. So $(1, 0)$ is not an eigenvector. Every entry must be scaled by the **same** factor.", "question": {"q": "Is $(1, -1)$ an eigenvector of $\\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}$?", "options": ["Yes, with λ = −1", "No, Av points elsewhere", "Yes, with λ = 1"], "answer": 2, "why": "$A(1, -1) = (2 - 1, 1 - 2) = (1, -1)$: unchanged, so $\\lambda = 1$. A minus sign inside $v$ doesn't make the eigenvalue negative.", "hint": "Multiply it out and compare $Av$ with $v$ entry by entry."}}, {"kind": "explain", "title": "What λ does", "body": "The eigenvalue says what happens along its direction:\n\n- $\\lambda > 1$: stretched;\n- $0 < \\lambda < 1$: shrunk;\n- $\\lambda < 0$: flipped to point the other way, and scaled by $|\\lambda|$;\n- $\\lambda = 0$: crushed to zero, so $v$ lies in the **null space** of $A$.", "question": {"q": "A matrix sends $v = (4, 2)$ to $(-2, -1)$. What does it do along $v$?", "options": ["Halves it, same way", "Flips it and halves it", "Flips it and doubles it"], "answer": 1, "why": "$(-2, -1) = -0.5 \\cdot (4, 2)$, so $\\lambda = -0.5$: the negative sign flips the direction, and $|\\lambda| = 0.5$ halves the length.", "hint": "Find the single number that turns $(4, 2)$ into $(-2, -1)$, then read off its sign and size."}}, {"kind": "explain", "title": "Directions, not vectors", "body": "If $Av = \\lambda v$, then for any number $c \\neq 0$:\n\n$$A(cv) = cAv = \\lambda(cv)$$\n\nSo every non-zero multiple of an eigenvector is an eigenvector too, with the same $\\lambda$. $(2, 2)$ and $(-5, -5)$ both work for our $A$. An eigenvector is really a **direction**.\n\nThat is why libraries return them scaled to length 1, with a sign that can go either way: `np.linalg.eig` may give $(0.71, 0.71)$ or $(-0.71, -0.71)$.", "question": {"q": "$(-5, -5)$ is an eigenvector of $\\begin{pmatrix} 2 & 1 \\\\ 1 & 2 \\end{pmatrix}$. What is its eigenvalue?", "options": ["−15", "−3", "3"], "answer": 2, "why": "$A(-5, -5) = (-15, -15) = 3 \\cdot (-5, -5)$. Scaling the vector, even by a negative number, never changes $\\lambda$.", "hint": "Compare $Av$ with $v$ itself, not with $(1, 1)$. Both have negative entries."}}, {"kind": "explain", "title": "Triangular shortcut", "body": "For a **triangular** matrix, with zeros on one side of the diagonal, the eigenvalues are the diagonal entries.\n\n$T = \\begin{pmatrix} 4 & 1 \\\\ 0 & 2 \\end{pmatrix}$ has eigenvalues 4 and 2. For $\\lambda = 4$, the vector $(1, 0)$ works: $T(1, 0) = (4, 0)$. The eigenvector for $\\lambda = 2$ is not an axis, so you still have to find it by testing.", "question": {"q": "Which vector is an eigenvector of $T$ for $\\lambda = 2$?", "options": ["(0, 1)", "(1, −2)", "(1, 2)"], "answer": 1, "why": "$T(1, -2) = (4 - 2, -4) = (2, -4) = 2 \\cdot (1, -2)$. The axis $(0, 1)$ goes to $(1, 2)$, which points elsewhere.", "hint": "Multiply each candidate by $T$ and check whether the result is exactly twice the candidate."}}, {"kind": "explain", "title": "Applied again and again", "body": "Eigenvectors explain what happens when you apply a matrix repeatedly. Each application multiplies an eigenvector by $\\lambda$ again:\n\n$$A^n v = \\lambda^n v$$\n\nWith $\\lambda = 3$, ten applications multiply $v$ by $3^{10} = 59{,}049$. With $\\lambda = 1$, $v$ never changes. This is the arithmetic behind exploding and vanishing gradients in deep networks, and behind PageRank settling on one direction.", "question": {"q": "$v$ is an eigenvector with $\\lambda = 0.5$. What is $A^3 v$?", "options": ["0.125 v", "1.5 v", "0.5 v"], "answer": 0, "why": "$0.5^3 = 0.125$. Repeated application multiplies the eigenvalues, so the vector shrinks fast; adding them would give $1.5$.", "hint": "Each application multiplies by 0.5 once more. Do that three times."}}, {"kind": "question", "title": "Recap and a check", "body": "An eigenvector is a direction the matrix only stretches, $Av = \\lambda v$, and $\\lambda$ is the stretch factor. To test a candidate, multiply and check whether the result is a multiple of it.", "question": {"q": "For $B = \\begin{pmatrix} 1 & 2 \\\\ 2 & 1 \\end{pmatrix}$, which is true?", "options": ["(1, 0) is one, with λ = 1", "(1, −1) is not one", "(1, −1) is one, with λ = −1"], "answer": 2, "why": "$B(1, -1) = (1 - 2, 2 - 1) = (-1, 1) = -1 \\cdot (1, -1)$: the direction is flipped, so $\\lambda = -1$. $B(1, 0) = (1, 2)$ points elsewhere.", "hint": "Multiply each candidate by $B$. A result pointing exactly the opposite way still counts as a multiple."}}]
---

## The big picture

A matrix usually both stretches and turns a vector. But most matrices have a few special directions where they **only stretch**. Those are the eigenvectors, and the stretch factors are the eigenvalues. Knowing them tells you what a matrix does when applied over and over, which explains exploding gradients, PageRank and slow optimisation.

## The definition

An **eigenvector** of a square matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is a non-zero vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> with
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mi>v</mi></mrow></math></span></span>
Read it: "A times v is just a multiple of v". The number <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi></mrow></math></span> (Greek "lambda") is the **eigenvalue**. "Eigen" is German for "own": the matrix's own directions.

What different eigenvalues do to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>&gt;</mo><mn>1</mn></mrow></math></span>: stretches it.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mi>λ</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span>: shrinks it.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>&lt;</mo><mn>0</mn></mrow></math></span>: flips it to point the other way.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mn>0</mn></mrow></math></span>: crushes it to zero, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> is in the null space.

~~~dg-raw
%%viz m_eig live | Drag x around the circle. Where Ax points the same way as x, you have found an eigenvector.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**test a few vectors**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo><mo>=</mo><mn>3</mn><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Same direction, 3 times longer. Eigenvector with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mn>3</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Unchanged. Eigenvector with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> points a different way. Not an eigenvector.

</div>

```
A = np.array([[2, 1], [1, 2]])
for v in ([1, 1], [1, -1], [1, 0]):
    print(v, A @ np.array(v))
```

## Directions, not vectors

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mi>v</mi></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mi>c</mi><mi>v</mi><mo stretchy="false">)</mo><mo>=</mo><mi>c</mi><mi>A</mi><mi>v</mi><mo>=</mo><mi>λ</mi><mo stretchy="false">(</mo><mi>c</mi><mi>v</mi><mo stretchy="false">)</mo></mrow></math></span>. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>5</mn><mo separator="true">,</mo><mo>−</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> are eigenvectors too. Libraries return them scaled to length 1, with an arbitrary sign.

## A triangular shortcut

For a triangular matrix the eigenvalues are the diagonal entries. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span> has eigenvalues 4 and 2; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> is an eigenvector for 4 because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

an eigenvector is a direction the matrix only stretches; the eigenvalue is the stretch factor.

</div>

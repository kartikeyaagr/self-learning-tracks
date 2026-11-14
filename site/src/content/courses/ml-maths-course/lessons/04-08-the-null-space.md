---
id: "04.08"
course: "ml-maths-course"
module: "04"
slug: "the-null-space"
title: "The Null Space"
keyIdeas: ["The null space ⟦1403⟧ is the set of inputs ⟦8⟧ sends to zero. It lives in the input space.", "If ⟦1062⟧ solves ⟦816⟧, every solution is ⟦1404⟧ with ⟦1405⟧.", "Find a basis from RREF (one vector per free column) or from the SVD."]
question: "A full-rank linear layer maps ⟦1402⟧. What is the dimension of its null space, and what does that mean for a later layer trying to recover the input?"
answer: "Step 1, ⟦1259⟧ is ⟦1416⟧ with rank 1024. Step 2, rank-nullity: ⟦1417⟧. Step 3, two inputs that differ by a null-space vector give the same output, ⟦1418⟧. No later layer can tell them apart, so information along those 3072 directions is lost."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Subspaces, columns and null spaces"]
tools: ["NumPy", "SciPy"]
topic: {"id": "04.02", "title": "Subspaces, columns and null spaces"}
---

## The big picture

The second question about a matrix is: **what does it ignore?** Some input directions are sent to zero and vanish without trace. Those directions form the null space. It decides how many solutions a system has, and which information a layer throws away.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>:</mo><mi>A</mi><mi>x</mi><mo>=</mo><mn>0</mn><mo stretchy="false">}</mo></mrow></math></span></span>
It always contains <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn></mrow></math></span>, and it is a subspace (lesson 04.06).

~~~dg-raw
%%viz m_nul | A rank-1 matrix squashes the plane onto a line. Every x on the dashed line is sent to 0: that line is the null space.
~~~

## Finding it by hand

<div class="ex">
<div class="lab">Worked example</div>

**a null space by hand**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>4</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>6</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span>. Elimination gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center center center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>2</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>3</mn></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></math></span>.

Columns 2 and 3 are free, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>=</mo><mo>−</mo><mn>2</mn><msub><mi>x</mi><mn>2</mn></msub><mo>−</mo><mn>3</mn><msub><mi>x</mi><mn>3</mn></msub></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>1</mn><mo separator="true">,</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>0</mn><mo separator="true">,</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mn>1</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>3</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> is the plane spanned by these two.

</div>

## Null space and the number of solutions

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>p</mi></msub></mrow></math></span> is one solution of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>, then every solution is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>p</mi></msub><mo>+</mo><msub><mi>x</mi><mi>n</mi></msub></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>n</mi></msub><mo>∈</mo><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span>. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> has **at most one** solution exactly when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><mn>0</mn><mo stretchy="false">}</mo></mrow></math></span>.

```
A = np.array([[1, 2, 3], [2, 4, 6]])
print(A @ np.array([-2, 1, 0]), A @ np.array([-3, 0, 1]))  # [0 0] [0 0]
xp = np.array([1, 0, 0])
print(A @ (xp + 5 * np.array([-2, 1, 0])))  # [1 2]
```

## In code

The SVD gives an orthonormal basis of the null space, which is numerically stable:

```
A = np.array([[1.0, 2, 3], [2, 4, 6]])
U, s, Vt = np.linalg.svd(A)
r = (s > 1e-10).sum()
null_basis = Vt[r:].T
print(r, null_basis.shape)  # 1 (3, 2)
print(np.allclose(A @ null_basis, 0))  # True
```

`scipy.linalg.null_space(A)` does the same.

## Where this breaks in practice

- **Over-parameterised regression** (fewer examples than features) has a big null space, so infinitely many weight vectors fit the training data. Gradient descent from zero picks the shortest one.

<div class="take">
<div class="lab">Key takeaway</div>

the null space is what <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> throws away. Every solution of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> is one solution plus a null-space vector.

</div>

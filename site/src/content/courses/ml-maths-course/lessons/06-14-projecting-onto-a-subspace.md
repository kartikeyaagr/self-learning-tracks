---
id: "06.14"
course: "ml-maths-course"
module: "06"
slug: "projecting-onto-a-subspace"
title: "Projecting onto a Subspace"
keyIdeas: ["To project ⟦682⟧ onto ⟦1388⟧, solve the normal equations ⟦2030⟧; the projection is ⟦2031⟧.", "They come from requiring the residual to be perpendicular to every column of ⟦8⟧.", "Projecting onto a column space is exactly least-squares fitting."]
question: "Project ⟦1395⟧ onto the plane spanned by ⟦1321⟧ and ⟦2029⟧."
answer: "Step 1, ⟦2038⟧ and ⟦2039⟧, so ⟦2040⟧. Step 2, ⟦2041⟧. Step 3, the residual ⟦2042⟧ points straight up, perpendicular to the floor."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Projection and orthonormalization"]
tools: ["NumPy"]
topic: {"id": "06.03", "title": "Projection and orthonormalization"}
---

## The big picture

Now the target set is a whole subspace, the column space of a matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>. The idea is the same as for a line: the leftover must be perpendicular to the set. This one condition gives least squares.

## The normal equations

Points in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>C</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span> are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span>. Require <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>−</mo><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> to be perpendicular to every column of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn><mspace width="1em"/><mo>⟹</mo><mspace width="1em"/><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mtext> </mtext><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span></span>
"Normal" here means perpendicular. If the columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> are independent, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi></mrow></math></span> is invertible and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> is unique.

~~~dg-flow Projection splits b into a part inside the subspace and a perpendicular residual.
b -(project)-> *p = A x-hat, in C(A)
b -(subtract p)-> residual, perpendicular to C(A)
~~~

## A line fit as a projection

```
A = np.array([[1.0, 0], [1, 1], [1, 2]]); b = np.array([1.0, 2, 4])
xh = np.linalg.solve(A.T @ A, A.T @ b)
p = A @ xh
print(np.round(xh, 4), np.round(p, 4))  # [0.8333 1.5   ] [0.8333 2.3333 3.8333]
print(np.allclose(A.T @ (b - p), 0))  # True
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> has a column of ones and a column <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>: this fits a line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><msub><mi>c</mi><mn>0</mn></msub><mo>+</mo><msub><mi>c</mi><mn>1</mn></msub><mi>t</mi></mrow></math></span> through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>x</mi><mo>^</mo></mover></mrow></math></span> holds intercept <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.83</mn></mrow></math></span> and slope <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1.5</mn></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> holds the fitted values.

~~~dg-raw
%%viz f_curve f=0.8333+1.5*x x=-0.3,2.5 y=0,5 tk=1 pts=0;1;2 | The fitted line y = 0.83 + 1.5t. Its values at t = 0, 1, 2 (dots) are the projection p of b = (1, 2, 4) onto the column space.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

projecting onto a column space means solving <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>A</mi><mover accent="true"><mi>x</mi><mo>^</mo></mover><mo>=</mo><msup><mi>A</mi><mi mathvariant="normal">⊤</mi></msup><mi>b</mi></mrow></math></span>; the projection is the least-squares fit.

</div>

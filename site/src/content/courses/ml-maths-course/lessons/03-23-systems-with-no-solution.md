---
id: "03.23"
course: "ml-maths-course"
module: "03"
slug: "systems-with-no-solution"
title: "Systems with No Solution"
keyIdeas: ["A system has no solution when elimination produces a row ⟦1044⟧ with ⟦1045⟧, which says ⟦1046⟧.", "Equivalently, ⟦682⟧ is not a combination of the columns of ⟦8⟧; the rank of ⟦959⟧ is bigger than the rank of ⟦8⟧.", "Overdetermined systems from real data almost never have exact solutions, which is why least squares exists."]
question: "Show that ⟦1042⟧, ⟦1043⟧ has no solution."
answer: "Step 1, R2 − 2·R1 ⟦1056⟧. Step 2, that row says ⟦1057⟧, i.e. ⟦1058⟧, impossible. Step 3, so there is no solution. Geometrically the two lines are parallel (⟦1042⟧ and ⟦1059⟧)."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

Not every system can be solved. Contradictory equations are easy to spot in small cases, but in large systems elimination is how you find out. And in ML, when there are more data points than parameters, "no exact solution" is the normal situation.

## The tell-tale row

If elimination ever produces a row of the form
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">[</mo><mn>0</mn><mtext> </mtext><mn>0</mn><mtext> </mtext><mo>⋯</mo><mtext> </mtext><mn>0</mn><mo>∣</mo><mi>c</mi><mo stretchy="false">]</mo><mo separator="true">,</mo><mspace width="2em"/><mi>c</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span></span>
that row says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>=</mo><mi>c</mi></mrow></math></span>. The system is **inconsistent**: it has no solution.

~~~dg-raw
%%viz f_plane x=-1,4 y=-1,4 ln=1,1,2,x+y=2,a,0.2;1,1,2.5,x+y=2.5,f,0.8 | x + y = 2 and 2x + 2y = 5 (which is x + y = 2.5): parallel lines that never meet.
~~~

## The rank test

The system has a solution exactly when adding <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> as a column does not increase the rank:

```
A = np.array([[1, 1], [2, 2]]); b = np.array([2, 5])
rA = np.linalg.matrix_rank(A)
rAb = np.linalg.matrix_rank(np.column_stack([A, b]))
print(rA, rAb)  # 1 2
```

Rank 2 > rank 1, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> points in a direction the columns of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> cannot reach.

<div class="ex">
<div class="lab">Worked example</div>

**too many data points**

Fit a line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mi>a</mi><mo>+</mo><mi>b</mi><mi>x</mi></mrow></math></span> through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>.

Equations: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>=</mo><mn>1</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>+</mo><mi>b</mi><mo>=</mo><mn>2</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>+</mo><mn>2</mn><mi>b</mi><mo>=</mo><mn>2</mn></mrow></math></span>. The first two give <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mn>1</mn></mrow></math></span>, but then the third says <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>2</mn></mrow></math></span>. No solution.

The points aren't on one line. Least squares (module 08) finds the line closest to all three instead.

</div>

## Where this breaks in practice

- **`np.linalg.solve` and inconsistent systems.** For a square singular <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> it raises `LinAlgError: Singular matrix`. For a non-square system it refuses to run; use `np.linalg.lstsq`.

<div class="take">
<div class="lab">Key takeaway</div>

a row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>⋯</mo><mn>0</mn><mo>∣</mo><mi>c</mi><mo stretchy="false">]</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>c</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span> means no solution. With real data, expect it, and use least squares.

</div>

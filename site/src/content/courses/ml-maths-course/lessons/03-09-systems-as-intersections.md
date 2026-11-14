---
id: "03.09"
course: "ml-maths-course"
module: "03"
slug: "systems-as-intersections"
title: "Systems as Intersections"
keyIdeas: ["A system ⟦816⟧ asks for the points on all the hyperplanes ⟦827⟧ at once.", "There are exactly three possibilities: no solution, exactly one, or infinitely many.", "In 2D: crossing lines give one, parallel lines give none, the same line twice gives infinitely many."]
question: "Classify each system: (a) ⟦824⟧; (b) ⟦825⟧; (c) ⟦826⟧."
answer: "Step 1, (a) the lines have different slopes, so they cross once. Adding gives ⟦830⟧, so ⟦836⟧. Step 2, (b) same left side, different right sides: parallel lines, no solution. Step 3, (c) the second equation is twice the first: the same line, infinitely many solutions."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

Solving a linear system sounds like algebra, but it is geometry: find the points that lie on every hyperplane at once. Seeing it this way explains why there are only three possible outcomes, and sets up the algorithms in the rest of this module.

## Three outcomes

Each equation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>a</mi><mi>i</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>x</mi><mo>=</mo><msub><mi>b</mi><mi>i</mi></msub></mrow></math></span> is one hyperplane. A system <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> asks for the points that lie on **all** of them. In 2D, two lines can meet in three ways:

~~~dg-layers Three lines in the plane: the three possible outcomes of a linear system.
*Crossing lines | exactly one solution, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> invertible
Parallel lines | no solution, inconsistent system
Same line | infinitely many solutions, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> singular
~~~

~~~dg-raw
%%viz f_plane x=-1,4 y=-2,3 ln=1,1,3,x+y=3;1,-1,1,x−y=1,f pt=2,1,(2~_1) | Crossing lines: the one common point is the one solution.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the three cases with numbers**

Crossing: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>3</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>−</mo><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>. Add them: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>x</mi><mo>=</mo><mn>4</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1</mn></mrow></math></span>.

Parallel: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>3</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>5</mn></mrow></math></span>. The same left side can't equal both 3 and 5.

Same line: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>=</mo><mn>3</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>x</mi><mo>+</mo><mn>2</mn><mi>y</mi><mo>=</mo><mn>6</mn></mrow></math></span>. Every point of the line works.

</div>

```
A = np.array([[1, 1], [1, -1]]); b = np.array([3, 1])
print(np.linalg.solve(A, b))  # [2. 1.]
print(np.linalg.matrix_rank(np.array([[1, 1], [2, 2]])))  # 1
```

The rank of 1 says the second row adds no new direction: the lines are parallel or identical.

## Why never exactly two?

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></mrow></math></span> both solve <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>, then every point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>t</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>−</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> on the line through them also solves it, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>t</mi><mo stretchy="false">(</mo><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>−</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><mi>b</mi><mo>+</mo><mi>t</mi><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo><mo>=</mo><mi>b</mi></mrow></math></span>. So two solutions always bring infinitely many.

## In more dimensions

The same three outcomes hold for any number of equations and unknowns. Which one you get depends on the rank of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, studied in module 04.

<div class="take">
<div class="lab">Key takeaway</div>

a linear system has no solution, one solution, or infinitely many. Geometrically, that is how its hyperplanes meet.

</div>

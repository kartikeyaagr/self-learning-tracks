---
id: "09.39"
course: "ml-maths-course"
module: "09"
slug: "area-under-a-curve-by-partitions"
title: "Area Under a Curve by Partitions"
keyIdeas: ["To find the area under ⟦90⟧ on ⟦294⟧, cut the interval into thin strips and approximate each by a rectangle.", "With ⟦233⟧ strips of width ⟦2947⟧ and heights ⟦2948⟧, the area is about ⟦2949⟧.", "Thinner strips give better approximations."]
question: "Approximate the area under ⟦380⟧ on ⟦4⟧ with 4 strips using right endpoints."
answer: "Step 1, ⟦2955⟧, points ⟦2956⟧. Step 2, heights ⟦2957⟧, sum ⟦2958⟧. Step 3, area ⟦2959⟧. The exact value is ⟦2954⟧; right endpoints overestimate an increasing function."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

An integral adds up infinitely many tiny pieces. The idea starts with something concrete: chop the region under a curve into rectangles and add their areas.

## Strips

~~~dg-raw
%%viz f_curve f=x*x x=0,1 y=0,1 tk=0.25 pts=0.25;0.5;0.75;1 rs=0,1,4,r | f(x) = x² on [0, 1] with four right-endpoint rectangles of width 0.25. Together they add to 0.469, above the true area 1/3.
~~~

```
for n in (4, 16, 100):
    x = np.arange(1, n + 1) / n
    print(n, round((x ** 2).sum() / n, 5))
```

<div class="ex">
<div class="lab">Worked example</div>

**left versus right**

Left endpoints <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo separator="true">,</mo><mn>0.25</mn><mo separator="true">,</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0.75</mn></mrow></math></span>: heights sum to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.875</mn></mrow></math></span>, area <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≈</mo><mn>0.219</mn></mrow></math></span>.

Right endpoints gave <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.469</mn></mrow></math></span>. The true <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span> lies between, and the gap shrinks as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> grows.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

approximate an area by thin rectangles; thinner rectangles give a better answer.

</div>

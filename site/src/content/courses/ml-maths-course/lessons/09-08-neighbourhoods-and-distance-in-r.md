---
id: "09.08"
course: "ml-maths-course"
module: "09"
slug: "neighbourhoods-and-distance-in-r"
title: "Neighbourhoods and Distance in Rⁿ"
keyIdeas: ["The open ball ⟦2521⟧ is the set of points within distance ⟦118⟧ of ⟦364⟧.", "\"Near ⟦364⟧\" in several variables means inside a small ball, approached from any direction.", "A set is open if every point has a small ball around it inside the set; open sets are where derivatives make sense."]
question: "Is ⟦2519⟧ in the open unit ball around the origin? Is the set ⟦2520⟧ open?"
answer: "Step 1, ⟦2525⟧: yes. Step 2, for a point with ⟦2526⟧, the ball of radius ⟦364⟧ around it stays in ⟦487⟧. So the half-plane is open."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

In one variable, "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> close to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span>" means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>x</mi><mo>−</mo><mi>a</mi><mi mathvariant="normal">∣</mi></mrow></math></span> is small. In several variables the same idea uses the distance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>a</mi><mi mathvariant="normal">∥</mi></mrow></math></span>. The crucial new feature: you can approach a point from infinitely many directions, not just left and right.

## Balls

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>B</mi><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>r</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>:</mo><mi mathvariant="normal">∥</mi><mi>x</mi><mo>−</mo><mi>a</mi><mi mathvariant="normal">∥</mi><mo>&lt;</mo><mi>r</mi><mo stretchy="false">}</mo></mrow></math></span></span>
In 1D it is an interval, in 2D a disc without its edge, in 3D a solid ball.

~~~dg-raw
%%viz f_region kind=disk c=1,1 r=1 edge=open x=-1,3 y=-1,3 pts=1.6,1.7,inside,f;2.2,2,outside,f | The open ball B((1, 1), 1): all points within distance 1 of (1, 1), not including the boundary circle (dashed).
~~~

```
a = np.array([1.0, 1]); r = 1.0
for p in ([1.6, 1.7], [2.2, 2.0], [2.0, 1.0]):
    print(p, np.linalg.norm(np.array(p) - a) < r)
```

The last point lies exactly on the boundary, so it is not in the open ball.

## Open sets

A set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> is **open** if every point of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> has some ball around it that fits inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span>. Open sets have no edge points, so you can step a little in every direction and stay inside. That is exactly what a derivative needs.

<div class="take">
<div class="lab">Key takeaway</div>

in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span>, "near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span>" means inside a small ball around <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span>, and a point can be approached from every direction.

</div>

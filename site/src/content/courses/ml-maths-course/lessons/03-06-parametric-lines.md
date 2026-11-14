---
id: "03.06"
course: "ml-maths-course"
module: "03"
slug: "parametric-lines"
title: "Parametric Lines"
keyIdeas: ["A line can be written as a point plus multiples of a direction: ⟦773⟧.", "This form works in any dimension; the equation form gives a line only in 2D.", "Interpolating between two points ⟦364⟧ and ⟦682⟧ is the parametric line ⟦774⟧."]
question: "Write the line through ⟦769⟧ and ⟦770⟧ in parametric form, and give the points at ⟦771⟧ and ⟦772⟧."
answer: "Step 1, direction ⟦784⟧. Step 2, the line is ⟦785⟧. Step 3, ⟦771⟧: ⟦786⟧, the midpoint. Step 4, ⟦772⟧: ⟦787⟧, past ⟦682⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

The equation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> says which points are on a line. The parametric form says how to walk along it. Walking along a line is exactly what optimisers do (step in a direction), and what researchers do when they interpolate between two trained models.

## Point plus direction

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><msub><mi>x</mi><mn>0</mn></msub><mo>+</mo><mi>t</mi><mi>v</mi><mo>:</mo><mi>t</mi><mo>∈</mo><mi mathvariant="normal">ℝ</mi><mo stretchy="false">}</mo></mrow></math></span></span>
Read it: "start at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>0</mn></msub></mrow></math></span>, walk <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> steps of size <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>". <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow></math></span> is the start, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow></math></span> is one step, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mo>−</mo><mn>2</mn></mrow></math></span> is two steps backwards.

~~~dg-raw
%%viz f_plane x=-1,6 y=-1,4 sg=-1,3,6,-0.5,m,d pt=1,2,t_=_0;3,1,t_=_1,f;2,1.5,t_=_0.5,o;5,0,t_=_2,o va=1,2,3,1,v,a | Start at (1, 2) and step along v = (2, −1). Each value of t gives one point on the line.
~~~

This works in any dimension. In 3D the equation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> gives a plane, and you would need two equations to pin down a line. The parametric form needs only a point and a direction.

## Interpolating between two points

The line through <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>+</mo><mi>t</mi><mo stretchy="false">(</mo><mi>b</mi><mo>−</mo><mi>a</mi><mo stretchy="false">)</mo></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> between 0 and 1 you get the segment between them.

<div class="ex">
<div class="lab">Worked example</div>

**interpolating between two checkpoints**

Weights <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>θ</mi><mn>0</mn></msub></mrow></math></span> from early training and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>θ</mi><mn>1</mn></msub></mrow></math></span> from the end. The straight path is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>θ</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>θ</mi><mn>0</mn></msub><mo>+</mo><mi>t</mi><mo stretchy="false">(</mo><msub><mi>θ</mi><mn>1</mn></msub><mo>−</mo><msub><mi>θ</mi><mn>0</mn></msub><mo stretchy="false">)</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>θ</mi><mn>0</mn></msub></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>θ</mi><mn>1</mn></msub></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi><mo>=</mo><mn>0.5</mn></mrow></math></span> is the midpoint. Plotting the loss along this path shows whether there is a "hill" between the two solutions.

</div>

```
theta0 = np.array([0.0, 2.0]); theta1 = np.array([4.0, 0.0])
for t in (0, 0.25, 0.5, 1):
    print(t, theta0 + t * (theta1 - theta0))
```

## Line search

Optimisers move along parametric lines. From the current point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>0</mn></msub></mrow></math></span>, choose a direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> (for gradient descent, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi><mo>=</mo><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mn>0</mn></msub><mo stretchy="false">)</mo></mrow></math></span>, downhill) and try several step sizes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span>. That is a **line search**.

<div class="take">
<div class="lab">Key takeaway</div>

a line is a start point plus <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> times a direction. It works in any dimension, and it is how optimisers move.

</div>

---
id: "03.03"
course: "ml-maths-course"
module: "03"
slug: "normal-vectors-and-hyperplanes"
title: "Normal Vectors and Hyperplanes"
keyIdeas: ["In ⟦679⟧, the vector ⟦378⟧ is the normal: it is perpendicular to every direction inside the line or plane.", "In ⟦2⟧ the same equation gives a hyperplane, a flat slice of dimension ⟦711⟧.", "⟦378⟧ points toward the side where ⟦712⟧."]
question: "For the plane ⟦708⟧ in ⟦26⟧, give the normal vector and check that it is perpendicular to the direction between the points ⟦709⟧ and ⟦710⟧."
answer: "Step 1, the normal is the coefficient vector ⟦725⟧. Step 2, both points are on the plane: ⟦726⟧ and ⟦726⟧. Step 3, the direction between them is ⟦727⟧. Step 4, ⟦728⟧, so they are perpendicular."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

A line's equation hides a useful arrow: the coefficient vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>. It points straight out of the line, at right angles. That arrow tells you which way a classifier "faces", and it is the same idea in 2, 3 or 1,000 dimensions.

## The normal vector

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> is called the **normal** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>. "Normal" here means "at right angles".

Why? Take any two points <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></mrow></math></span> on the line. Both satisfy the equation, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><mi>b</mi></mrow></math></span>. Subtract: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span>. The vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>−</mo><msup><mi>x</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></mrow></math></span> is a direction lying along the line, and its dot product with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> is zero. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> is perpendicular to every direction inside the line.

~~~dg-raw
%%viz f_plane x=-1,4 y=-1,4 ln=3,4,5,3x₁+4x₂=5,a,0.85 pt=0.6,0.8,,k va=0.6,0.8,1.8,2.4,w,f | The normal w = (3, 4) drawn from a point on the line 3x₁ + 4x₂ = 5. It meets the line at a right angle.
~~~

## Hyperplanes

The same equation works in any dimension:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup><mo>:</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi><mo stretchy="false">}</mo><mo separator="true">,</mo><mspace width="2em"/><mi>w</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span></span>

- In 2D it is a line.
- In 3D it is a flat plane.
- In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> dimensions it is a flat slice of dimension <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>−</mo><mn>1</mn></mrow></math></span>, called a **hyperplane**. You cannot picture it, but the algebra is identical.

<div class="ex">
<div class="lab">Worked example</div>

**a plane in 3D**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mn>2</mn><msub><mi>x</mi><mn>2</mn></msub><mo>−</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mn>4</mn></mrow></math></span> has normal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Points on it include <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>.

The direction from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><mo stretchy="false">(</mo><mo>−</mo><mn>4</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>4</mn><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mn>4</mn><mo>+</mo><mn>0</mn><mo>+</mo><mn>4</mn><mo>=</mo><mn>0</mn></mrow></math></span>.

</div>

```
w = np.array([1, 2, -1])
P = np.array([[4, 0, 0], [0, 2, 0], [0, 0, -4]])
print(P @ w)  # [4 4 4]
print(w @ (P[1] - P[0]), w @ (P[2] - P[0]))  # 0 0
```

## In ML

Every unit of a linear layer computes <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span> for its own <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>. The set where that is zero is a hyperplane in input space. A ReLU unit switches on and off exactly across that hyperplane.

<div class="take">
<div class="lab">Key takeaway</div>

the coefficient vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> is perpendicular to the hyperplane <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span> in every dimension.

</div>

---
id: "03.07"
course: "ml-maths-course"
module: "03"
slug: "planes-in-three-dimensions"
title: "Planes in Three Dimensions"
keyIdeas: ["A plane in ⟦26⟧ is ⟦679⟧, or a point plus two independent directions: ⟦791⟧.", "The normal can be found from two directions with the cross product: ⟦792⟧.", "Two non-parallel planes meet in a line; three planes usually meet in a point."]
question: "Find the equation of the plane through ⟦788⟧ spanned by ⟦789⟧ and ⟦790⟧."
answer: "Step 1, the normal is ⟦805⟧. Step 2, the plane passes through the origin, so ⟦730⟧: ⟦806⟧, or ⟦807⟧. Step 3, check: ⟦280⟧ gives ⟦808⟧ and ⟦776⟧ gives ⟦809⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

Moving from the plane to three dimensions is the first step toward the high-dimensional spaces of ML. Most of what you need is the same as for lines, with one new tool: the cross product, which turns two directions into a normal.

## Two descriptions of a plane

- **Equation form:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><msub><mi>w</mi><mn>3</mn></msub><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mi>b</mi></mrow></math></span>, with normal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>.
- **Parametric form:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><msub><mi>x</mi><mn>0</mn></msub><mo>+</mo><mi>s</mi><mi>u</mi><mo>+</mo><mi>t</mi><mi>v</mi><mo>:</mo><mi>s</mi><mo separator="true">,</mo><mi>t</mi><mo>∈</mo><mi mathvariant="normal">ℝ</mi><mo stretchy="false">}</mo></mrow></math></span>: a point and two directions that don't point the same way.

## From directions to a normal

In 3D only, the **cross product** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span> is a vector perpendicular to both:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>u</mi><mo>×</mo><mi>v</mi><mo>=</mo><mo stretchy="false">(</mo><msub><mi>u</mi><mn>2</mn></msub><msub><mi>v</mi><mn>3</mn></msub><mo>−</mo><msub><mi>u</mi><mn>3</mn></msub><msub><mi>v</mi><mn>2</mn></msub><mo separator="true">,</mo><mtext> </mtext><msub><mi>u</mi><mn>3</mn></msub><msub><mi>v</mi><mn>1</mn></msub><mo>−</mo><msub><mi>u</mi><mn>1</mn></msub><msub><mi>v</mi><mn>3</mn></msub><mo separator="true">,</mo><mtext> </mtext><msub><mi>u</mi><mn>1</mn></msub><msub><mi>v</mi><mn>2</mn></msub><mo>−</mo><msub><mi>u</mi><mn>2</mn></msub><msub><mi>v</mi><mn>1</mn></msub><mo stretchy="false">)</mo></mrow></math></span></span>

```
u = np.array([1, 0, 1]); v = np.array([0, 1, 1])
w = np.cross(u, v)
print(w, w @ u, w @ v)  # [-1 -1  1] 0 0
```

<div class="ex">
<div class="lab">Worked example</div>

**a plane from three points**

Points <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>=</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Directions: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi><mo>−</mo><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi><mo>−</mo><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Normal: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>⋅</mo><mn>1</mn><mo>−</mo><mn>0</mn><mo separator="true">,</mo><mtext> </mtext><mn>0</mn><mo>−</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mtext> </mtext><mn>0</mn><mo>−</mo><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>p</mi><mo>=</mo><mn>1</mn></mrow></math></span>. The plane is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><msub><mi>x</mi><mn>3</mn></msub><mo>=</mo><mn>1</mn></mrow></math></span>.

</div>

## Intersections

- Two planes that are not parallel meet in a line.
- Three planes usually meet in a single point. That is a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>×</mo><mn>3</mn></mrow></math></span> linear system with one solution.
- They can also share a line, or have no common point at all. Lesson 03.09 sorts these cases out.

## Beyond 3D

The cross product only exists in 3D. In higher dimensions, normals are found by solving linear equations instead (module 04's null space).

<div class="take">
<div class="lab">Key takeaway</div>

a plane is a normal plus an offset, or a point plus two directions. In 3D the cross product links the two.

</div>

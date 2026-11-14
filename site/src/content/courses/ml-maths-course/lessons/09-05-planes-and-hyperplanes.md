---
id: "09.05"
course: "ml-maths-course"
module: "09"
slug: "planes-and-hyperplanes"
title: "Planes and Hyperplanes"
keyIdeas: ["A hyperplane ⟦2490⟧ is a level set of the linear function ⟦724⟧; ⟦378⟧ is perpendicular to it.", "It splits space into two half-spaces; a linear classifier predicts by which side ⟦7⟧ is on.", "The signed distance from ⟦570⟧ to the hyperplane is ⟦2491⟧."]
question: "For ⟦655⟧, ⟦749⟧, find the signed distance from ⟦2489⟧ to the boundary, and which side it is on."
answer: "Step 1, ⟦2496⟧. Step 2, ⟦756⟧, so the distance is ⟦2497⟧. Step 3, positive, so ⟦570⟧ is on the side ⟦378⟧ points to."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Geometry of multivariable surfaces"]
tools: ["NumPy"]
topic: {"id": "09.01", "title": "Geometry of multivariable surfaces"}
---

## The big picture

The simplest functions of several variables are linear: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span>. Their level sets are flat. They are the decision boundaries of linear classifiers and the building blocks of every neural network layer.

## Level sets of a linear function

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span> has level sets <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo>=</mo><mi>c</mi></mrow></math></span>: parallel lines in 2D, planes in 3D, **hyperplanes** in general. The vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> is perpendicular to all of them, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> increases fastest in direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-1,5 y=-1,4 tk=1 ln=3,4,5,w·x_+_b_=_0,a,0.1 v=0.6,0.8,w,f pt=3,2,x₀,k sg=3,2,1.56,0.08,k,d | The boundary 3x + 4y = 5 (green), its normal w (orange) and the point x₀ = (3, 2). The dashed segment is the perpendicular distance, 12/5 = 2.4.
~~~

## Distance to a hyperplane

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mtext>dist</mtext><mo stretchy="false">(</mo><msub><mi>x</mi><mn>0</mn></msub><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><msub><mi>x</mi><mn>0</mn></msub><mo>+</mo><mi>b</mi></mrow><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></mfrac></mrow></math></span></span>
The sign tells you the side. An SVM's margin is this quantity for the closest training points.

```
w = np.array([3.0, 4]); b = -5.0; x0 = np.array([3.0, 2])
print((w @ x0 + b) / np.linalg.norm(w))  # 2.4
foot = x0 - (w @ x0 + b) / (w @ w) * w
print(np.round(foot, 2), round(w @ foot + b, 10))  # [1.56 0.08] 0.0
```

The foot of the perpendicular, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1.56</mn><mo separator="true">,</mo><mn>0.08</mn><mo stretchy="false">)</mo></mrow></math></span>, lies on the boundary.

<div class="take">
<div class="lab">Key takeaway</div>

linear functions have flat, parallel level sets; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> is perpendicular to them, and dividing by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></math></span> turns a score into a distance.

</div>

---
id: "03.24"
course: "ml-maths-course"
module: "03"
slug: "systems-with-infinitely-many-solutions"
title: "Systems with Infinitely Many Solutions"
keyIdeas: ["A consistent system with fewer pivots than unknowns has infinitely many solutions.", "Every solution is one particular solution plus a solution of ⟦889⟧: ⟦1060⟧.", "Libraries return one of them, usually the one with the smallest length."]
question: "Write all solutions of ⟦977⟧ as a particular solution plus a homogeneous part."
answer: "Step 1, one particular solution: ⟦1074⟧ gives ⟦558⟧, so ⟦1075⟧. Step 2, solve ⟦1076⟧: ⟦1077⟧ gives ⟦1078⟧, so ⟦1079⟧. Step 3, all solutions: ⟦1080⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Solving linear systems by row reduction"]
tools: ["NumPy"]
topic: {"id": "03.03", "title": "Solving linear systems by row reduction"}
---

## The big picture

When equations are redundant, a whole line or plane of answers fits. That is not a failure: it means the data does not pin down the answer, and some extra rule (like "smallest weights") must pick one. Over-parameterised neural networks live in exactly this situation.

## Particular plus homogeneous

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>p</mi></msub></mrow></math></span> is any one solution of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>, then every solution has the form
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>x</mi><mo>=</mo><msub><mi>x</mi><mi>p</mi></msub><mo>+</mo><msub><mi>x</mi><mi>h</mi></msub><mo separator="true">,</mo><mspace width="2em"/><mi>A</mi><msub><mi>x</mi><mi>h</mi></msub><mo>=</mo><mn>0</mn></mrow></math></span></span>
Why? <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>p</mi></msub><mo>+</mo><msub><mi>x</mi><mi>h</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mi>b</mi><mo>+</mo><mn>0</mn><mo>=</mo><mi>b</mi></mrow></math></span>. And if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is any solution, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><msub><mi>x</mi><mi>p</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mi>b</mi><mo>−</mo><mi>b</mi><mo>=</mo><mn>0</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>−</mo><msub><mi>x</mi><mi>p</mi></msub></mrow></math></span> is a homogeneous solution.

~~~dg-raw
%%viz f_plane x=-3,5 y=-1,4 ln=1,2,4,x+2y=4,a,0.15 pt=4,0,xₚ;2,1,xₚ+xₕ,o va=4,0,2,1,xₕ,f | Starting from one solution xₚ = (4, 0), adding any multiple of xₕ = (−2, 1) stays on the line x + 2y = 4.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a plane of solutions**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>+</mo><mi>y</mi><mo>+</mo><mi>z</mi><mo>=</mo><mn>1</mn></mrow></math></span> has one equation and three unknowns, so two free variables.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>x</mi><mi>p</mi></msub><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>. Homogeneous solutions: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>+</mo><mi>t</mi><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>s</mi><mo>=</mo><mi>t</mi><mo>=</mo><mn>1</mn></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mn>1</mn><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mtext> </mtext><mn>1</mn><mo separator="true">,</mo><mtext> </mtext><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>, and indeed <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>+</mo><mn>1</mn><mo>=</mo><mn>1</mn></mrow></math></span>.

</div>

## Which one does code return?

`np.linalg.lstsq` returns the solution with the smallest length <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi></mrow></math></span>:

```
A = np.array([[1.0, 2]]); b = np.array([4.0])
x, *_ = np.linalg.lstsq(A, b, rcond=None)
print(np.round(x, 6))  # [0.8 1.6]
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.8</mn><mo separator="true">,</mo><mn>1.6</mn><mo stretchy="false">)</mo></mrow></math></span> is on the line (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.8</mn><mo>+</mo><mn>3.2</mn><mo>=</mo><mn>4</mn></mrow></math></span>) and is the point of the line closest to the origin.

## In ML

Gradient descent started from zero on an under-determined least-squares problem also converges to the smallest-norm solution. This "implicit bias" is one reason over-parameterised models can still generalise.

<div class="take">
<div class="lab">Key takeaway</div>

all solutions = one solution + anything <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> sends to zero. Code picks the shortest one.

</div>

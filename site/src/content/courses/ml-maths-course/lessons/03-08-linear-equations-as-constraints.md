---
id: "03.08"
course: "ml-maths-course"
module: "03"
slug: "linear-equations-as-constraints"
title: "Linear Equations as Constraints"
keyIdeas: ["Each linear equation ⟦814⟧ restricts ⟦7⟧ to a hyperplane; each inequality ⟦815⟧ restricts it to a half-space.", "Several constraints together carve out their intersection.", "In matrix form, ⟦816⟧ stacks equations as rows and ⟦810⟧ stacks inequalities."]
question: "Write as ⟦810⟧: ⟦811⟧, ⟦812⟧, ⟦813⟧. Describe the region."
answer: "Step 1, rewrite each as \"⟦135⟧\": ⟦820⟧, ⟦821⟧, ⟦813⟧. Step 2, rows of ⟦8⟧: ⟦822⟧, ⟦113⟧, ⟦111⟧, and ⟦823⟧. Step 3, the region is the triangle with corners ⟦685⟧, ⟦310⟧, ⟦5⟧."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

Linear equations and inequalities are rules a point must obey. Budgets, probabilities that must sum to 1, weights that must stay non-negative: all are linear constraints. Seeing each one as a cut through space makes systems of them easy to picture.

## One constraint, one cut

- An equation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>c</mi></mrow></math></span> keeps only the points on a hyperplane.
- An inequality <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>a</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>≤</mo><mi>c</mi></mrow></math></span> keeps a whole half-space.

## Many constraints, one region

Several constraints keep the points that satisfy all of them: the intersection.

~~~dg-raw
%%viz f_plane x=-0.5,1.5 y=-0.5,1.5 tk=0.5 pg=0:0_1:0_0:1,a ln=1,1,1,x₁+x₂=1,f;1,0,0,,m;0,1,0,,m | The triangle where x₁ ≥ 0, x₂ ≥ 0 and x₁ + x₂ ≤ 1. Each inequality removes a half-plane.
~~~

## Matrix form

Stack the left sides as rows of a matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and the right sides in a vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span>:

```
A = np.array([[-1, 0], [0, -1], [1, 1]]); b = np.array([0, 0, 1])
pts = np.array([[0.2, 0.3], [0.8, 0.5], [-0.1, 0.5]])
print(np.all(pts @ A.T <= b, axis=1))  # [ True False False]
```

Each row of `pts @ A.T` holds the three left sides for one point.

<div class="ex">
<div class="lab">Worked example</div>

**the probability simplex**

Three class probabilities <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>=</mo><mo stretchy="false">(</mo><msub><mi>p</mi><mn>1</mn></msub><mo separator="true">,</mo><msub><mi>p</mi><mn>2</mn></msub><mo separator="true">,</mo><msub><mi>p</mi><mn>3</mn></msub><mo stretchy="false">)</mo></mrow></math></span> must satisfy <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>≥</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mn>1</mn></msub><mo>+</mo><msub><mi>p</mi><mn>2</mn></msub><mo>+</mo><msub><mi>p</mi><mn>3</mn></msub><mo>=</mo><mn>1</mn></mrow></math></span>.

That is one equation (a plane) and three inequalities (half-spaces). Together they give a triangle in 3D called the simplex. Every softmax output lies on it.

</div>

## Where you will see this

- **Linear programming** optimises a linear objective over such a region.
- **Constrained optimisation** (module 09) adds constraints to a loss.
- **ReLU networks** split input space into regions bounded by hyperplanes.

<div class="take">
<div class="lab">Key takeaway</div>

each linear constraint is a cut. A system of them is the region that survives every cut.

</div>

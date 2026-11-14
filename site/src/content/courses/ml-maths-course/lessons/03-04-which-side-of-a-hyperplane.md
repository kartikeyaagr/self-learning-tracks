---
id: "03.04"
course: "ml-maths-course"
module: "03"
slug: "which-side-of-a-hyperplane"
title: "Which Side of a Hyperplane"
keyIdeas: ["The sign of ⟦734⟧ says which side of the hyperplane ⟦7⟧ is on: positive on the side ⟦378⟧ points to, negative on the other.", "A linear classifier predicts with ⟦735⟧; its decision boundary is a hyperplane.", "Evaluating a batch of points is one matrix product ⟦736⟧."]
question: "A classifier has ⟦729⟧, ⟦730⟧ and predicts class 1 when ⟦353⟧. Classify ⟦731⟧, ⟦732⟧ and ⟦733⟧."
answer: "Step 1, ⟦731⟧: ⟦746⟧, class 1. Step 2, ⟦732⟧: ⟦747⟧, class 0. Step 3, ⟦733⟧: ⟦328⟧, exactly on the boundary; the rule ⟦748⟧ gives class 0, but a tiny change would flip it."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Lines, planes and geometric constraints"]
tools: ["NumPy"]
topic: {"id": "03.01", "title": "Lines, planes and geometric constraints"}
---

## The big picture

A hyperplane cuts space into two halves. A linear classifier is exactly that: compute one number, look at its sign, and you know which half (which class) a point is in. Logistic regression, linear SVMs and the last layer of most neural networks all work this way.

## The sign test

Moving from a point on the hyperplane in the direction of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> increases <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi></mrow></math></span>. So:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>&gt;</mo><mi>b</mi></mrow></math></span>: the side <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> points toward.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>&lt;</mo><mi>b</mi></mrow></math></span>: the other side.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>=</mo><mi>b</mi></mrow></math></span>: on the hyperplane.

~~~dg-raw
%%viz f_region kind=half w=3,4 c0=-5 x=-1,4 y=-1,4 edge=closed pts=2,2,(2~_2);0,0,(0~_0),f lab=3x₁+4x₂>5 | The shaded half-plane is where 3x₁ + 4x₂ > 5. The point (2, 2) is inside it; the origin is on the other side.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**which side is a point on?**

Line <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mn>4</mn><msub><mi>x</mi><mn>2</mn></msub><mo>=</mo><mn>5</mn></mrow></math></span>, normal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">)</mo></mrow></math></span>.

Point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>6</mn><mo>+</mo><mn>8</mn><mo>=</mo><mn>14</mn><mo>&gt;</mo><mn>5</mn></mrow></math></span>, positive side.

Point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mn>5</mn></mrow></math></span>, negative side.

Point <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0.5</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>5</mn></mrow></math></span>, exactly on the line.

</div>

## A batch at once

Stack the points as rows of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mi>w</mi></mrow></math></span> computes every dot product in one go, which is what a linear classifier does:

```
w = np.array([3, 4]); b = 5
X = np.array([[2, 2], [0, 0], [1, 0.5]])
print(X @ w - b)  # [ 9. -5.  0.]
print(np.sign(X @ w - b))  # [ 1. -1.  0.]
```

## Classifier notation

ML usually writes the boundary as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo>=</mo><mn>0</mn></mrow></math></span> (with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>+</mo><mi>b</mi></mrow></math></span>), so the bias here is minus the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> above. The predicted class is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mrow><mi mathvariant="normal">s</mi><mi mathvariant="normal">i</mi><mi mathvariant="normal">g</mi><mi mathvariant="normal">n</mi></mrow><mo stretchy="false">(</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span>, or with probabilities, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo stretchy="false">(</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>+</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span> in logistic regression. Either way, the boundary is the hyperplane where the score is zero.

## Where this breaks in practice

- **Points near the boundary.** A score of 0.001 and a score of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>0.001</mn></mrow></math></span> get different classes. Tiny input noise can flip them, so report scores, not just labels.

<div class="take">
<div class="lab">Key takeaway</div>

the sign of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>x</mi><mo>−</mo><mi>b</mi></mrow></math></span> tells you the side. A linear classifier is that sign.

</div>

---
id: "07.05"
course: "ml-maths-course"
module: "07"
slug: "classifying-critical-points"
title: "Classifying Critical Points"
keyIdeas: ["At a critical point ⟦2189⟧, so ⟦2190⟧.", "⟦1785⟧ positive definite: local minimum. Negative definite: local maximum. Mixed signs: saddle. A zero eigenvalue: the test is inconclusive.", "In high dimensions most critical points of a random-like loss are saddles."]
question: "Classify the critical point at the origin of ⟦2188⟧."
answer: "Step 1, the gradient ⟦2198⟧ is zero at the origin. Step 2, ⟦2199⟧, trace 4, determinant ⟦2200⟧. Step 3, ⟦2201⟧, so ⟦2202⟧. Step 4, mixed signs: a saddle. Along ⟦2203⟧, ⟦2204⟧; along ⟦2205⟧, ⟦2206⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §4.5-4.6"}, {"title": "MIT 18.065 Matrix Methods in Data Analysis (Strang)", "url": "https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/", "note": "Read: lectures 6-9"}]
tags: ["Curvature and quadratic objectives"]
tools: ["NumPy"]
topic: {"id": "07.01", "title": "Curvature and quadratic objectives"}
---

## The big picture

Setting the gradient to zero finds candidates. The Hessian's eigenvalues tell you which candidates are minima, which are maxima, and which are saddles.

## The test

| Hessian eigenvalues | Type |
| --- | --- |
| all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&gt;</mo><mn>0</mn></mrow></math></span> | local minimum |
| all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo><mn>0</mn></mrow></math></span> | local maximum |
| some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&gt;</mo><mn>0</mn></mrow></math></span>, some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo><mn>0</mn></mrow></math></span> | saddle point |
| some <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mn>0</mn></mrow></math></span>, others one sign | inconclusive: look at higher-order terms |

In 2D a shortcut: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>H</mi><mo>&lt;</mo><mn>0</mn></mrow></math></span> means a saddle; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>H</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>H</mi><mn>11</mn></msub><mo>&gt;</mo><mn>0</mn></mrow></math></span> means a minimum.

```
H = np.array([[2.0, 4], [4, 2]])
print(np.linalg.eigvalsh(H))  # [-2.  6.]
f = lambda x, y: x ** 2 + 4 * x * y + y ** 2
print(round(f(0.1, 0.1), 4), round(f(0.1, -0.1), 4))  # 0.06 -0.02
```

## An inconclusive case

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><msup><mi>y</mi><mn>4</mn></msup></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>−</mo><msup><mi>y</mi><mn>4</mn></msup></mrow></math></span> both have <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi><mo>=</mo><mi mathvariant="normal">diag</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></math></span> at the origin. The first has a minimum there, the second a saddle. A zero eigenvalue means the quadratic picture says nothing about that direction.

## Saddles in deep learning

For a random symmetric <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>×</mo><mi>n</mi></mrow></math></span> matrix, the chance that all eigenvalues are positive falls extremely fast with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>. So among critical points of a large network's loss, saddles vastly outnumber minima. Gradient noise in SGD helps push training off them.

<div class="take">
<div class="lab">Key takeaway</div>

at a critical point, the signs of the Hessian's eigenvalues decide minimum, maximum or saddle; a zero eigenvalue leaves the question open.

</div>

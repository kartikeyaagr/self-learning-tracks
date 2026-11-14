---
id: "10.13"
course: "ml-maths-course"
module: "10"
slug: "transformations-in-several-dimensions-and-normalising-flows"
title: "Transformations in Several Dimensions and Normalising Flows"
keyIdeas: ["For an invertible ⟦3374⟧, ⟦3375⟧.", "Normalising flows chain invertible layers and add up log-determinants: ⟦3376⟧.", "Flow layers are designed so the determinant is cheap, usually a triangular Jacobian."]
question: "⟦3369⟧ and ⟦3370⟧ with ⟦3371⟧. How does ⟦3372⟧ compare with ⟦3373⟧?"
answer: "Step 1, ⟦914⟧, so the map stretches area by 6. Step 2, ⟦3380⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Transforming random variables"]
tools: ["NumPy"]
topic: {"id": "10.03", "title": "Transforming random variables"}
---

## The big picture

The one-dimensional derivative factor becomes a Jacobian determinant, the volume-scaling factor from lessons 03.02 and 09.50. Normalising flows are generative models built entirely on this.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><msup><mi>g</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mtext> </mtext><mo stretchy="false">∣</mo><mi>det</mi><mo>⁡</mo><msub><mi>J</mi><msup><mi>g</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo stretchy="false">∣</mo></mrow></math></span></span>

~~~dg-flow Normalising flow: sample simple noise, push it through invertible layers, track the log-determinant.
z ~ N(0, I) -(invertible g_1)-> h_1 -(invertible g_2)-> *x
~~~

```
from scipy import stats
A = np.array([[2.0, 0], [0, 3]])
pz = stats.multivariate_normal(np.zeros(2), np.eye(2)).pdf([0, 0])
px = stats.multivariate_normal(np.zeros(2), A @ A.T).pdf([0, 0])
print(round(pz / px, 6), np.linalg.det(A))  # 6.0 6.0
```

## Cheap determinants

A general <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mo>×</mo><mi>d</mi></mrow></math></span> determinant costs about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>d</mi><mn>3</mn></msup></mrow></math></span> operations. Coupling layers keep half the coordinates fixed and transform the rest using them, which makes the Jacobian triangular. Its determinant is the product of the diagonal: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> operations.

<div class="take">
<div class="lab">Key takeaway</div>

in many dimensions the density picks up <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">∣</mo><mi>det</mi><mo>⁡</mo><mi>J</mi><mo stretchy="false">∣</mo></mrow></math></span>; flows choose layers that make it cheap.

</div>

## Where this breaks in practice

- Flows must be invertible, so the latent has the same dimension as the data. That makes them memory-hungry for images.

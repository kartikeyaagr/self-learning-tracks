---
id: "06.06"
course: "ml-maths-course"
module: "06"
slug: "other-inner-products-and-mahalanobis-distance"
title: "Other Inner Products and Mahalanobis Distance"
keyIdeas: ["Any symmetric positive definite ⟦607⟧ defines an inner product ⟦1909⟧.", "With ⟦1910⟧ you get the Mahalanobis distance ⟦1911⟧, measured in units of the data's spread.", "It is the quantity in the exponent of a Gaussian density and a standard outlier score."]
question: "Data has mean 0 and covariance ⟦1906⟧. Compare the Euclidean and Mahalanobis distances of ⟦1907⟧ and ⟦1908⟧ from the mean."
answer: "Step 1, both points have Euclidean distance 5. Step 2, ⟦1922⟧. Step 3, ⟦1907⟧: ⟦1923⟧. Step 4, ⟦1908⟧: ⟦1924⟧. Step 5, the first is half a standard deviation out (normal); the second is five standard deviations out (very unusual)."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
---

## The big picture

The ordinary dot product treats every direction the same. Real data does not: some features vary a lot, others barely. A point 10 units away along a high-variance direction can be less surprising than one 2 units away along a low-variance direction. A weighted inner product fixes this.

## Weighted inner products

For a symmetric positive definite matrix <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span> (module 05),
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">⟨</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><msub><mo stretchy="false">⟩</mo><mi>M</mi></msub><mo>=</mo><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>M</mi><mi>y</mi></mrow></math></span></span>
keeps the three properties from lesson 06.01: linear, symmetric, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">⟨</mo><mi>x</mi><mo separator="true">,</mo><mi>x</mi><msub><mo stretchy="false">⟩</mo><mi>M</mi></msub><mo>&gt;</mo><mn>0</mn></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>. So it gives its own lengths, angles and distances.

<div class="ex">
<div class="lab">Worked example</div>

**a diagonal weight**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi><mo>=</mo><mi mathvariant="normal">diag</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>4</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><msub><mi mathvariant="normal">∥</mi><mi>M</mi></msub><mo>=</mo><msqrt><mn>4</mn></msqrt><mo>=</mo><mn>2</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><msub><mi mathvariant="normal">∥</mi><mi>M</mi></msub><mo>=</mo><mn>1</mn></mrow></math></span>.

The first axis now "counts double". Circles of constant <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span>-length become ellipses squashed along that axis.

</div>

## Mahalanobis distance

With the data's covariance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> and mean <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span>, use <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi><mo>=</mo><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>d</mi><mi>M</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msqrt><mrow><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mi mathvariant="normal">⊤</mi></msup><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>μ</mi><mo stretchy="false">)</mo></mrow></msqrt></mrow></math></span></span>
Dividing by the covariance measures distance in standard deviations, direction by direction.

~~~dg-raw
%%viz f_plane x=-12,12 y=-7,7 tk=5 pt=5,0,(5~_0):_0.5_sd,a;0,5,(0~_5):_5_sd,f sg=-10,0,10,0,m,d;0,-1,0,1,m,d | Data spread: about ±10 along x₁ and ±1 along x₂ (dashed). Both points are 5 units from the centre, but (0, 5) is far outside the data's spread.
~~~

```
Si = np.linalg.inv(np.diag([100.0, 1]))
maha = lambda v: np.sqrt(v @ Si @ v)
print(maha(np.array([5.0, 0])), maha(np.array([0.0, 5])))  # 0.5 5.0
```

## Where it appears

- The Gaussian density is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∝</mo><mi>exp</mi><mo>⁡</mo><mo stretchy="false">(</mo><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><msub><mi>d</mi><mi>M</mi></msub><mo stretchy="false">(</mo><mi>x</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></math></span>: points with equal Mahalanobis distance have equal density.
- Outlier detection: flag points with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>d</mi><mi>M</mi></msub></mrow></math></span> above a threshold.
- Out-of-distribution detection for neural networks often fits a Gaussian to features and scores new inputs by Mahalanobis distance.

<div class="take">
<div class="lab">Key takeaway</div>

a positive definite matrix gives a new inner product; with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> it measures distance in units of the data's own spread.

</div>

## Where this breaks in practice

- With many features and few samples, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span> is singular or close to it and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">Σ</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> explodes. Use a shrinkage estimate such as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi><mo>+</mo><mi>ϵ</mi><mi>I</mi></mrow></math></span>.

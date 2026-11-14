---
id: "06.05"
course: "ml-maths-course"
module: "06"
slug: "dot-products-in-high-dimensions"
title: "Dot Products in High Dimensions"
keyIdeas: ["For random vectors with independent standard normal entries in ⟦2⟧, ⟦1891⟧ has mean 0 and variance ⟦3⟧.", "Their lengths are about ⟦1890⟧, so their cosine is about ⟦1892⟧: random high-dimensional vectors are nearly perpendicular.", "Attention divides ⟦1889⟧ by ⟦1890⟧ to bring the scores' spread back to about 1."]
question: "Queries and keys have i.i.d. entries with mean 0 and variance 1, in ⟦1888⟧. What are the mean and standard deviation of ⟦1889⟧? Why divide by ⟦1890⟧ before the softmax?"
answer: "Step 1, ⟦1889⟧ is a sum of 128 terms ⟦1902⟧. Step 2, each term has mean ⟦328⟧ and variance ⟦1903⟧. Step 3, the terms are independent, so the sum has mean 0 and variance 128, standard deviation ⟦1899⟧. Step 4, softmax of scores spread over about ⟦1904⟧ puts almost all weight on the largest; its gradients become tiny. Dividing by ⟦1905⟧ brings the spread back to 1."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Geometry from inner products"]
tools: ["NumPy"]
topic: {"id": "06.01", "title": "Geometry from inner products"}
---

## The big picture

Intuition from 2D and 3D fails in 768 dimensions. Two random directions there are almost always nearly perpendicular. This fact explains the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi>d</mi></msqrt></mrow></math></span> in attention and why cosine similarities between unrelated embeddings cluster near 0.

## The size of a random dot product

Take <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> with independent standard normal entries.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>x</mi><mi mathvariant="normal">⊤</mi></msup><mi>y</mi><mo>=</mo><msub><mo>∑</mo><mi>i</mi></msub><msub><mi>x</mi><mi>i</mi></msub><msub><mi>y</mi><mi>i</mi></msub></mrow></math></span> is a sum of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> independent terms, each with mean 0 and variance 1. So it has mean 0 and standard deviation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi>d</mi></msqrt></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> is a sum of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> terms with mean 1, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>x</mi><mi mathvariant="normal">∥</mi><mo>≈</mo><msqrt><mi>d</mi></msqrt></mrow></math></span>.
- So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>cos</mi><mo>⁡</mo><mi>θ</mi><mo>≈</mo><mstyle displaystyle="true" scriptlevel="0"><mfrac><mrow><msqrt><mi>d</mi></msqrt><mtext>-sized</mtext></mrow><mi>d</mi></mfrac></mstyle><mo>=</mo><mstyle displaystyle="true" scriptlevel="0"><mfrac><mn>1</mn><msqrt><mi>d</mi></msqrt></mfrac></mstyle></mrow></math></span>.

```
rng = np.random.default_rng(0)
for d in (3, 100, 10000):
    X = rng.normal(size=(500, d)); Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    c = (Xn[:250] * Xn[250:]).sum(1)
    print(d, round(c.std(), 3), round(1 / np.sqrt(d), 3))
```

The measured spread of cosines matches <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mi>d</mi></msqrt></mrow></math></span> closely once <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> is moderately large.

~~~dg-raw
%%viz f_bars v=0.577,0.1,0.036,0.01 l=d_=_3,d_=_100,d_=_768,d_=_10000 hi=none yl=typical_cosine_size_(1/√d) | As the dimension grows, two random directions become nearly perpendicular. At d = 768 (a common embedding size) the typical cosine is about 0.036.
~~~

## Why attention divides by √d

An unscaled attention score <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>q</mi><mi mathvariant="normal">⊤</mi></msup><mi>k</mi></mrow></math></span> has standard deviation about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi>d</mi></msqrt></mrow></math></span>. Softmax of widely spread numbers picks one winner and ignores the rest.

~~~dg-flow Scaled dot-product attention.
q, k -(q.k)-> score, sd ~ √(d) -(divide by √(d))-> *score, sd ~ 1 -(softmax)-> weights
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what saturation looks like**

Scores <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>11</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>9</mn><mo stretchy="false">)</mo></mrow></math></span>: softmax gives about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.99998</mn><mo separator="true">,</mo><mn>0.00002</mn><mo separator="true">,</mo><mn>0.0</mn><mo stretchy="false">)</mo></mrow></math></span>. One winner takes everything.

Divided by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>128</mn></msqrt><mo>≈</mo><mn>11.3</mn></mrow></math></span>: scores <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.97</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>0.80</mn><mo stretchy="false">)</mo></mrow></math></span>, softmax about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.65</mn><mo separator="true">,</mo><mn>0.24</mn><mo separator="true">,</mo><mn>0.11</mn><mo stretchy="false">)</mo></mrow></math></span>.

</div>

```
sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
s = np.array([11.0, 0, -9])
print(np.round(sm(s), 5).tolist())  # [0.99998, 2e-05, 0.0]
print(np.round(sm(s / np.sqrt(128)), 2))  # [0.65 0.24 0.11]
```

<div class="take">
<div class="lab">Key takeaway</div>

in high dimensions random vectors are nearly perpendicular, and raw dot products grow like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mi>d</mi></msqrt></mrow></math></span>; scale by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mi>d</mi></msqrt></mrow></math></span> when their size matters.

</div>

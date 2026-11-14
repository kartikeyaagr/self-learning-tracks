---
id: "08.06"
course: "ml-maths-course"
module: "08"
slug: "centring-and-the-covariance-matrix"
title: "Centring and the Covariance Matrix"
keyIdeas: ["Centring subtracts each column's mean so every feature has mean 0.", "The sample covariance is ⟦2401⟧; ⟦2402⟧ is a variance and ⟦2403⟧ a covariance.", "The variance of the projection onto a unit vector ⟦378⟧ is ⟦2404⟧."]
question: "Data points ⟦2399⟧. Centre them, compute ⟦385⟧, and find the variance of the projection onto ⟦2400⟧."
answer: "Step 1, means ⟦2036⟧; centred points ⟦2412⟧. Step 2, ⟦2413⟧, ⟦2414⟧, ⟦2415⟧. Step 3, ⟦2416⟧, and ⟦2417⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 9-10"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 12-13"}]
tags: ["Principal component analysis"]
tools: ["NumPy"]
topic: {"id": "08.03", "title": "Principal component analysis"}
---

## The big picture

A dataset with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> features is a cloud of points in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> dimensions. The covariance matrix summarises the cloud's spread in every direction at once. PCA, Gaussian models and whitening all start from it.

## Centring

Subtract each column's mean: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>X</mi><mi>c</mi></msub><mo>=</mo><mi>X</mi><mo>−</mo><mn mathvariant="normal">𝟏</mn><msup><mover accent="true"><mi>x</mi><mo>ˉ</mo></mover><mi mathvariant="normal">⊤</mi></msup></mrow></math></span>. By lesson 06.13 this removes each column's component along the all-ones vector.

~~~dg-raw
%%viz f_plane x=-3,3 y=-3,3 tk=1 pt=-1,-2,,k;0,0,,k;1,2,,k ln=2,-1,0,all_points_lie_on_y_=_2x,a,0.08 | The three centred points (−1, −2), (0, 0), (1, 2) lie on one line, so their covariance matrix has rank 1: there is spread along (1, 2) and none across it.
~~~

## The covariance matrix

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>S</mi><mo>=</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></mfrac></mstyle><msubsup><mi>X</mi><mi>c</mi><mi mathvariant="normal">⊤</mi></msubsup><msub><mi>X</mi><mi>c</mi></msub></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>S</mi><mrow><mi>j</mi><mi>j</mi></mrow></msub></mrow></math></span>: variance of feature <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>S</mi><mrow><mi>j</mi><mi>k</mi></mrow></msub></mrow></math></span>: covariance of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>j</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi></mrow></math></span>; positive when they rise together.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> is symmetric and positive semidefinite.

```
X = np.array([[1.0, 2], [2, 4], [3, 6]])
Xc = X - X.mean(axis=0)
print(Xc.T @ Xc / (len(X) - 1))
print(np.allclose(np.cov(X, rowvar=False), Xc.T @ Xc / 2))  # True
```

## Variance in any direction

Project every point onto a unit vector <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>: the scores are <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>X</mi><mi>c</mi></msub><mi>w</mi></mrow></math></span>, with variance
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mo stretchy="false">(</mo><msub><mi>X</mi><mi>c</mi></msub><mi>w</mi><mo stretchy="false">)</mo><mo>=</mo><mstyle displaystyle="false" scriptlevel="0"><mfrac><mn>1</mn><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></mfrac></mstyle><mi mathvariant="normal">∥</mi><msub><mi>X</mi><mi>c</mi></msub><mi>w</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup><mo>=</mo><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>S</mi><mi>w</mi></mrow></math></span></span>
So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> answers "how spread out is the data along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span>?" for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> at once. For the example, along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><msqrt><mn>5</mn></msqrt></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>S</mi><mi>w</mi><mo>=</mo><mo stretchy="false">(</mo><mn>1</mn><mo>+</mo><mn>8</mn><mo>+</mo><mn>16</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>5</mn><mo>=</mo><mn>5</mn></mrow></math></span>, all of the total variance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>+</mo><mn>4</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

centre first; the covariance matrix then gives the spread along any direction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>w</mi><mi mathvariant="normal">⊤</mi></msup><mi>S</mi><mi>w</mi></mrow></math></span>.

</div>

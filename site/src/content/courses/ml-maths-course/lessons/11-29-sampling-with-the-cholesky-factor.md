---
id: "11.29"
course: "ml-maths-course"
module: "11"
slug: "sampling-with-the-cholesky-factor"
title: "Sampling With the Cholesky Factor"
keyIdeas: ["Factor ⟦3828⟧, draw ⟦3829⟧, and return ⟦3830⟧.", "It works because ⟦3831⟧.", "It is the reparameterisation trick for a full-covariance Gaussian."]
question: "Give the Cholesky factor of ⟦3827⟧ and the sampling formula."
answer: "Step 1, ⟦3832⟧, since ⟦3833⟧. Step 2, ⟦3834⟧, ⟦3835⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Gaussian combinations"]
tools: ["NumPy"]
topic: {"id": "11.05", "title": "Gaussian combinations"}
---

## The big picture

Libraries give independent standard normals. To get correlated ones, mix them with a matrix whose square is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span>.

## The recipe

~~~dg-flow Sampling a correlated Gaussian.
z ~ N(0, I) -(multiply by L, the Cholesky factor of Σ)-> L z -(add μ)-> *x ~ N(μ, Σ)
~~~

```
rng = np.random.default_rng(0)
L = np.linalg.cholesky(np.array([[1.0, 0.8], [0.8, 1]]))
print(L.round(2).tolist())  # [[1.0, 0.0], [0.8, 0.6]]
x = rng.normal(size=(500000, 2)) @ L.T
print(round(np.corrcoef(x.T)[0, 1], 2))  # 0.8
```

~~~dg-raw
%%viz f_plane x=-1.2,1.5 y=-1.2,1.5 tk=1 v=1,0,e₁,m;0,1,e₂,m;1,0.8,Le₁,a;0,0.6,Le₂,f | L maps the standard axes (grey) to (1, 0.8) and (0, 0.6). Independent noise along e₁ and e₂ becomes correlated noise.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>μ</mi><mo>+</mo><mi>L</mi><mi>z</mi></mrow></math></span> with the Cholesky factor turns independent normals into any Gaussian.

</div>

## Where this breaks in practice

- Cholesky fails on a covariance that is only positive semidefinite (or slightly indefinite from rounding). Add a small multiple of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi></mrow></math></span>.

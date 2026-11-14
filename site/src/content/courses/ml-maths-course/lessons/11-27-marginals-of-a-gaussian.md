---
id: "11.27"
course: "ml-maths-course"
module: "11"
slug: "marginals-of-a-gaussian"
title: "Marginals of a Gaussian"
keyIdeas: ["The marginal of a block of coordinates is Gaussian with that block's entries of ⟦1238⟧ and ⟦930⟧.", "No integral is needed: delete the other rows and columns.", "For jointly Gaussian variables, a zero covariance means independence."]
question: "⟦3811⟧, ⟦3812⟧. Find the marginal of ⟦3813⟧."
answer: "Step 1, keep entries 1 and 3: ⟦3814⟧. Step 2, ⟦3815⟧. Step 3, the zero off-diagonal means ⟦3816⟧ and ⟦3817⟧ are independent."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Gaussian combinations"]
tools: ["NumPy"]
topic: {"id": "11.05", "title": "Gaussian combinations"}
---

## The big picture

For most distributions, marginalising needs an integral. For Gaussians it is bookkeeping.

## Deleting rows and columns

```
mu = np.array([1.0, 2, 3])
S = np.array([[4.0, 1, 0], [1, 2, 0.5], [0, 0.5, 1]])
keep = [0, 2]
print(mu[keep], S[np.ix_(keep, keep)].tolist())  # [1. 3.] [[4.0, 0.0], [0.0, 1.0]]
rng = np.random.default_rng(0)
X = rng.multivariate_normal(mu, S, size=500000)[:, keep]
print(np.round(np.cov(X, rowvar=False), 1).tolist())  # [[4.0, 0.0], [0.0, 1.0]]
```

~~~dg-raw
%%viz f_curve f=Math.exp(-(x-1)*(x-1)/8)/(2*Math.sqrt(2*Math.PI)) x=-6,8 y=0,0.22 tk=2 | The marginal of X₁ alone: N(1, 4), read straight off the first entry of μ and Σ.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a Gaussian's marginals are read off by keeping the relevant entries of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Σ</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- This only works for jointly Gaussian vectors; a mixture of Gaussians has mixture marginals.

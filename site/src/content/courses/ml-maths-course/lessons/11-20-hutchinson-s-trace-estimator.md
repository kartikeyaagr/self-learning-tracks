---
id: "11.20"
course: "ml-maths-course"
module: "11"
slug: "hutchinson-s-trace-estimator"
title: "Hutchinson's Trace Estimator"
keyIdeas: ["If ⟦313⟧ has mean 0 and covariance ⟦1128⟧, then ⟦3752⟧.", "Averaging ⟦3753⟧ over random ⟦313⟧ estimates a trace using only products ⟦3754⟧.", "Large models use it for traces of Hessians and Jacobians that are too big to build."]
question: "Why do random ⟦1373⟧ vectors work for ⟦313⟧?"
answer: "Step 1, each entry has mean 0 and variance 1, and the entries are independent. Step 2, so ⟦3756⟧, and the quadratic-form formula gives ⟦3757⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Moments of joint distributions"]
tools: ["NumPy"]
topic: {"id": "11.03", "title": "Moments of joint distributions"}
---

## The big picture

You can compute <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi><mi>z</mi></mrow></math></span> for a huge matrix (a Hessian-vector product) but never store <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span>. This trick still gets its trace.

## The estimator

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">tr</mi><mo>⁡</mo><mi>M</mi><mo>≈</mo><mfrac><mn>1</mn><mi>K</mi></mfrac><munderover><mo>∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msubsup><mi>z</mi><mi>k</mi><mi mathvariant="normal">⊤</mi></msubsup><mi>M</mi><msub><mi>z</mi><mi>k</mi></msub></mrow></math></span></span>

```
rng = np.random.default_rng(0)
M = rng.normal(size=(50, 50))
z = rng.choice([-1.0, 1.0], size=(20000, 50))
est = np.einsum('ni,ij,nj->n', z, M, z).mean()
print(abs(est - np.trace(M)) < 1.0)  # True
```

<div class="take">
<div class="lab">Key takeaway</div>

random probe vectors turn matrix-vector products into a trace estimate.

</div>

## Where this breaks in practice

- The estimate's variance grows with the off-diagonal size of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span>; use more probes, or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>±</mo><mn>1</mn></mrow></math></span> entries, which have lower variance than Gaussian ones.

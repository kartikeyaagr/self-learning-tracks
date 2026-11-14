---
id: "04.16"
course: "ml-maths-course"
module: "04"
slug: "choosing-bases-in-ml"
title: "Choosing Bases in ML"
keyIdeas: ["Whitening changes to a basis where the data covariance is the identity.", "Fourier and wavelet bases make signals sparse: few large coordinates.", "A hidden layer is only defined up to an invertible change of basis, so single neurons need not mean anything."]
question: "A hidden layer ⟦1506⟧ feeds ⟦1507⟧. Show that replacing ⟦1259⟧ by ⟦1508⟧ and ⟦1204⟧ by ⟦1509⟧ (for invertible ⟦41⟧) gives the same ⟦509⟧. What does this say about interpreting single neurons?"
answer: "Step 1, new hidden values ⟦1512⟧. Step 2, new output ⟦1513⟧. Step 3, so the network computes exactly the same function with hidden coordinates ⟦1514⟧ instead of ⟦547⟧. Step 4, the training objective can't prefer one basis over another (unless something like a ReLU or regulariser breaks the symmetry), so an individual coordinate of ⟦547⟧ need not be meaningful."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 2"}, {"title": "Axler: Linear Algebra Done Right", "url": "https://linear.axler.net/", "note": "Read: ch. 1-3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 5-10"}]
tags: ["Bases as coordinate systems"]
tools: ["NumPy"]
topic: {"id": "04.03", "title": "Bases as coordinate systems"}
---

## The big picture

Choosing a basis is a modelling decision. The same data can look tangled in one basis and simple in another. This lesson tours three places in ML where the basis is chosen on purpose, and one where it is not determined at all.

## Whitening

Find a basis where the data has covariance <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi></mrow></math></span>: every direction has variance 1 and no two directions are correlated. PCA followed by rescaling each component does this.

```
rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 2)) @ np.array([[2.0, 0], [1.5, 0.5]])
Xc = X - X.mean(0)
vals, vecs = np.linalg.eigh(np.cov(Xc.T))
Z = Xc @ vecs / np.sqrt(vals)
print(np.round(np.cov(Z.T), 2))
```

The printed covariance is the identity, up to rounding.

## Sparse bases

Audio is a mix of a few frequencies, so in the Fourier basis most coordinates are near zero. JPEG uses a cosine basis on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>8</mn><mo>×</mo><mn>8</mn></mrow></math></span> blocks for the same reason: a few coordinates carry almost everything.

<div class="ex">
<div class="lab">Worked example</div>

**a signal with two frequencies**

A signal of 64 samples made of two sine waves has 62 non-zero values in the standard basis (it happens to cross zero at two samples).

In the Fourier basis it has just 4 non-zero coordinates (each real sine shows up at a positive and a negative frequency).

</div>

```
t = np.arange(64)
x = np.sin(2 * np.pi * 3 * t / 64) + 0.5 * np.sin(2 * np.pi * 10 * t / 64)
print((np.abs(x) > 1e-9).sum(), (np.abs(np.fft.fft(x)) > 1e-9).sum())  # 62 4
```

## Hidden layers have no preferred basis

For a linear map between layers, any invertible <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> can be inserted: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>V</mi><mo stretchy="false">(</mo><mi>B</mi><mi>W</mi><mo stretchy="false">)</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>B</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup></mrow></math></span> absorbed by the next layer leaves the function unchanged. So "neuron 17 detects cats" is a claim about one arbitrary basis. Interpretability methods such as sparse autoencoders look for a basis where single directions do mean something.

## Where this breaks in practice

- **Comparing models unit by unit.** Two trained networks use different bases for their hidden layers. Compare them with methods that ignore rotations, such as CKA or Procrustes alignment.

<div class="take">
<div class="lab">Key takeaway</div>

pick the basis that makes the problem simple: uncorrelated (whitening), sparse (Fourier), or interpretable. Hidden layers don't come with a basis of their own.

</div>

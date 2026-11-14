---
id: "06.12"
course: "ml-maths-course"
module: "06"
slug: "orthogonal-initialisation"
title: "Orthogonal Initialisation"
keyIdeas: ["A product of orthogonal matrices is orthogonal, so a deep stack of them keeps every input's length exactly.", "Random Gaussian layers keep length only on average; over many layers the singular values of the product spread out.", "Orthogonal initialisation (Saxe et al., 2014) keeps signals and gradients well-scaled at depth."]
question: "A 50-layer linear network uses random orthogonal ⟦2000⟧ matrices. What is ⟦2001⟧ for a unit vector ⟦7⟧? What changes with scaled Gaussian matrices?"
answer: "Step 1, each ⟦2002⟧ keeps length, so the product does: the output has length exactly 1, for every input. Step 2, Gaussian layers stretch some directions and squeeze others. Step 3, over 50 layers these effects multiply, so a few directions are amplified and many are crushed; the output length depends strongly on the input."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 3"}, {"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 14-17"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 5"}]
tags: ["Orthogonal structure"]
tools: ["NumPy"]
topic: {"id": "06.02", "title": "Orthogonal structure"}
---

## The big picture

A deep network is a long product of layer matrices. If each one slightly distorts lengths, the distortions compound. Starting the weights as orthogonal matrices removes the distortion at initialisation, which helps very deep and recurrent networks train.

## The experiment

```
rng = np.random.default_rng(0)
d, L = 64, 50
x = rng.normal(size=d); x /= np.linalg.norm(x)
o = x.copy()
for _ in range(L):
    o = np.linalg.qr(rng.normal(size=(d, d)))[0] @ o
print(round(np.linalg.norm(o), 6))  # 1.0
```

`np.linalg.qr` of a Gaussian matrix gives a random orthogonal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span>. After 50 layers the length is still exactly 1.

## The Gaussian product spreads out

```
rng = np.random.default_rng(1)
P = np.eye(64)
for _ in range(50):
    P = rng.normal(size=(64, 64)) / np.sqrt(64) @ P
s = np.linalg.svd(P, compute_uv=False)
print(s.max() / s.min() > 1e6)  # True
```

Each Gaussian layer keeps length only on average. The product's largest and smallest stretch factors differ by over a million times: a few directions dominate and the rest are lost. Gradients flowing backwards suffer the same.

~~~dg-raw
%%viz f_bars v=1,1000000 l=orthogonal,Gaussian hi=none yl=max/min_stretch_after_50_layers_(lower_bound) | After 50 layers the orthogonal product stretches every direction equally (ratio 1). The Gaussian product's ratio exceeds a million.
~~~

## In practice

- PyTorch has `torch.nn.init.orthogonal_`.
- Orthogonal init matters most for RNNs (the same matrix applied at every step) and very deep networks without normalisation.
- With nonlinearities the guarantee is only approximate, but the benefit largely remains.

<div class="take">
<div class="lab">Key takeaway</div>

orthogonal matrices keep lengths exactly, so a deep stack of them neither explodes nor vanishes at initialisation.

</div>

## Where this breaks in practice

- Orthogonality is lost as training updates the weights. Methods that force it during training limit what the network can express; most models only initialise this way.

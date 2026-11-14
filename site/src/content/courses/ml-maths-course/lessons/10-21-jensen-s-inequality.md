---
id: "10.21"
course: "ml-maths-course"
module: "10"
slug: "jensen-s-inequality"
title: "Jensen's Inequality"
keyIdeas: ["For convex ⟦3444⟧: ⟦3445⟧. For concave ⟦3444⟧ the inequality flips.", "⟦3446⟧ is how the ELBO in variational inference is derived.", "Equality holds only when ⟦513⟧ is constant (or ⟦3444⟧ is linear where ⟦513⟧ lives)."]
question: "⟦513⟧ is 1 or 3 with equal probability. Check Jensen for ⟦2328⟧ and for ⟦2559⟧."
answer: "Step 1, ⟦3447⟧. Step 2, ⟦3448⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
---

## The big picture

Expectation does not pass through nonlinear functions. Jensen says which way the gap goes, and that one inequality underlies the ELBO, the non-negativity of KL divergence, and many bounds.

## The picture

A convex function curves up. The chord between two points on its graph lies above the curve. The average of the function values is a point on the chord; the function at the average is on the curve below it.

~~~dg-raw
%%viz f_curve f=x*x g=4*x-3 x=0,3.3 y=0,14 tk=1 ytk=2 pts=1;2;3 fl=φ(x)_=_x² gl=chord_from_x_=_1_to_3 | φ(x) = x² (green) and the chord between x = 1 and x = 3 (orange). At the mean x = 2 the chord is at 5 = Eφ(X), above the curve's value 4 = φ(EX).
~~~

```
x = np.array([1.0, 3.0])
print(x.mean() ** 2, (x ** 2).mean(), round(np.log(x).mean(), 3), round(np.log(x.mean()), 3))  # 4.0 5.0 0.549 0.693
```

<div class="take">
<div class="lab">Key takeaway</div>

for convex functions, the function of the mean is at most the mean of the function.

</div>

## Where this breaks in practice

- Averaging predicted log-probabilities is not the same as the log of averaged probabilities; ensembles should average probabilities.

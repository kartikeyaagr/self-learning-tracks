---
id: "09.33"
course: "ml-maths-course"
module: "09"
slug: "the-hessian-as-the-jacobian-of-the-gradient"
title: "The Hessian as the Jacobian of the Gradient"
keyIdeas: ["The Hessian ⟦2847⟧ is the Jacobian of the map ⟦2848⟧.", "Newton's method linearises the gradient: solve ⟦2849⟧, so ⟦2222⟧.", "Hessian-vector products ⟦2850⟧ cost about two gradient evaluations and never form ⟦1785⟧."]
question: "For ⟦2846⟧, compute ⟦2698⟧, its Jacobian ⟦1785⟧, and one Newton step from ⟦111⟧."
answer: "Step 1, ⟦2855⟧ at ⟦111⟧. Step 2, ⟦2856⟧. Step 3, ⟦2857⟧; since ⟦2858⟧ for this quadratic, ⟦2859⟧ and the step lands on the minimum ⟦685⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Jacobians and local approximation"]
tools: ["NumPy"]
topic: {"id": "09.04", "title": "Jacobians and local approximation"}
---

## The big picture

The gradient is itself a vector-valued map. Its Jacobian is the Hessian, the matrix of second derivatives from lesson 09.14. Seeing it this way explains Newton's method and how frameworks compute curvature cheaply.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>H</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>J</mi><mrow><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><msub><mi>H</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mfrac><mrow><msup><mi mathvariant="normal">∂</mi><mn>2</mn></msup><mi>f</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>i</mi></msub><mi mathvariant="normal">∂</mi><msub><mi>x</mi><mi>j</mi></msub></mrow></mfrac></mrow></math></span></span>

## Newton's method

Linearise the gradient: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>d</mi><mo stretchy="false">)</mo><mo>≈</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>+</mo><mi>H</mi><mi>d</mi></mrow></math></span>. Set this to zero:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>d</mi><mo>=</mo><mo>−</mo><msup><mi>H</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span></span>

```
H = np.array([[2.0, 3], [3, 10]]); x = np.array([1.0, 1])
grad = H @ x
print(x - np.linalg.solve(H, grad))  # [0. 0.]
```

## Hessian-vector products

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi><mi>v</mi></mrow></math></span> is the directional derivative of the gradient along <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>H</mi><mi>v</mi><mo>≈</mo><mfrac><mrow><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>+</mo><mi>h</mi><mi>v</mi><mo stretchy="false">)</mo><mo>−</mo><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mi>h</mi><mi>v</mi><mo stretchy="false">)</mo></mrow><mrow><mn>2</mn><mi>h</mi></mrow></mfrac></mrow></math></span></span>
Autodiff computes it exactly for about the cost of two gradients. Power iteration with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi><mi>v</mi></mrow></math></span> estimates <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>λ</mi><mi>max</mi><mo>⁡</mo></msub></mrow></math></span> of a huge network's Hessian.

```
H = np.array([[2.0, 3], [3, 10]])
gradf = lambda p: np.array([2 * p[0] + 3 * p[1], 3 * p[0] + 10 * p[1]])
v = np.array([1.0, -1]); h = 1e-5; x = np.array([0.3, 0.7])
print(np.round((gradf(x + h * v) - gradf(x - h * v)) / (2 * h), 6), H @ v)  # [-1. -7.] [-1. -7.]
```

<div class="take">
<div class="lab">Key takeaway</div>

the Hessian is the Jacobian of the gradient; Newton's method solves its linearisation, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi><mi>v</mi></mrow></math></span> is cheap even when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>H</mi></mrow></math></span> is huge.

</div>

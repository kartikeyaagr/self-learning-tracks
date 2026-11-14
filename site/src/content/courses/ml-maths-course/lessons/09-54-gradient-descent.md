---
id: "09.54"
course: "ml-maths-course"
module: "09"
slug: "gradient-descent"
title: "Gradient Descent"
keyIdeas: ["Gradient descent repeats ⟦2714⟧ with step size ⟦324⟧.", "On a quadratic with largest Hessian eigenvalue ⟦491⟧, it converges only if ⟦490⟧.", "The condition number ⟦3149⟧ sets the speed: long narrow valleys are slow."]
question: "For ⟦3148⟧, what is the largest stable step size?"
answer: "Step 1, the Hessian is ⟦3157⟧, so ⟦3158⟧. Step 2, stability needs ⟦3159⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Constrained and unconstrained optimization"]
tools: ["NumPy", "PyTorch"]
topic: {"id": "09.08", "title": "Constrained and unconstrained optimization"}
---

## The big picture

Almost every model is trained by some form of gradient descent. This lesson shows why the step size matters and why badly scaled problems crawl.

## The update

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>x</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>x</mi><mi>k</mi></msub><mo>−</mo><mi>η</mi><mtext> </mtext><mi mathvariant="normal">∇</mi><mi>f</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>k</mi></msub><mo stretchy="false">)</mo></mrow></math></span></span>
On a quadratic with Hessian eigenvalues <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>≤</mo><mo>⋯</mo><mo>≤</mo><mi>L</mi></mrow></math></span>, each eigen-direction shrinks by a factor <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">∣</mo><mn>1</mn><mo>−</mo><mi>η</mi><mi>λ</mi><mo stretchy="false">∣</mo></mrow></math></span> per step. Too large an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi></mrow></math></span> and the steepest direction blows up; too small and the flattest direction barely moves.

~~~dg-raw
%%viz gdzoo live | Gradient descent with a step size and momentum on four landscapes. Push the step size up until the path starts to zig-zag, then diverge.
~~~

```
A = np.diag([1.0, 25.0])
for eta in (0.04, 0.077, 0.085):
    x = np.array([1.0, 1.0])
    for k in range(200):
        x = x - eta * (A @ x)
    print(eta, f"{np.linalg.norm(x):.1e}")
# 0.04 2.8e-04
# 0.077 2.0e-07
# 0.085 1.7e+10
```

At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mo>=</mo><mn>0.085</mn><mo>&gt;</mo><mn>0.08</mn></mrow></math></span> the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> coordinate grows every step.

<div class="take">
<div class="lab">Key takeaway</div>

gradient descent steps downhill by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi><mi mathvariant="normal">∇</mi><mi>f</mi></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>η</mi></mrow></math></span> must stay below <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi mathvariant="normal">/</mi><mi>L</mi></mrow></math></span>, and a large <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mi mathvariant="normal">/</mi><mi>μ</mi></mrow></math></span> makes it slow.

</div>

## Where this breaks in practice

- In deep learning the step size is the single most important hyperparameter. Learning-rate warm-up and decay schedules exist because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi></mrow></math></span> changes during training.

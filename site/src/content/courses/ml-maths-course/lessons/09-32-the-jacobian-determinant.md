---
id: "09.32"
course: "ml-maths-course"
module: "09"
slug: "the-jacobian-determinant"
title: "The Jacobian Determinant"
keyIdeas: ["For square ⟦2733⟧, ⟦2839⟧ is how much ⟦2801⟧ scales small volumes near ⟦7⟧.", "⟦2840⟧ means ⟦2801⟧ crushes some direction there and is not locally invertible.", "It is the factor in the change-of-variables formulas for integrals and densities; normalising flows are built to make it cheap."]
question: "For polar coordinates ⟦2837⟧, compute ⟦2838⟧."
answer: "Step 1, ⟦2844⟧. Step 2, ⟦2845⟧. Step 3, small polar boxes far from the origin cover more area."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Jacobians and local approximation"]
tools: ["NumPy"]
topic: {"id": "09.04", "title": "Jacobians and local approximation"}
---

## The big picture

Lesson 03.02 showed that a matrix's determinant scales areas. A curved map does the same locally, with its Jacobian's determinant. This number appears every time you change variables in an integral or a probability density.

## Local area scaling

A tiny square of area <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>ϵ</mi><mn>2</mn></msup></mrow></math></span> near <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> maps to an almost-parallelogram of area about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><msub><mi>J</mi><mi>F</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mi mathvariant="normal">∣</mi><mtext> </mtext><msup><mi>ϵ</mi><mn>2</mn></msup></mrow></math></span>.

~~~dg-raw
%%viz f_plane x=-0.2,2.4 y=-0.2,2.4 tk=1 pg=1:0_1.2:0_1.18:0.21_0.98:0.17,m;1.92:0.83_2.2:0.94_2.12:1.14_1.84:1.03,a | Polar coordinates. A small box with Δr = 0.2, Δθ = 0.17 near r = 1 (grey) and the same box near r = 2 (green, rotated to θ = 0.4). The far one covers about twice the area: det J = r.
~~~

```
def J(r, t):
    return np.array([[np.cos(t), -r * np.sin(t)], [np.sin(t), r * np.cos(t)]])
for r in (0.5, 1, 2):
    print(r, round(np.linalg.det(J(r, 0.3)), 6))
```

## Where it appears

- Change of variables in integrals: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi><mi>x</mi><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>J</mi><mi mathvariant="normal">∣</mi><mtext> </mtext><mi>d</mi><mi>u</mi><mtext> </mtext><mi>d</mi><mi>v</mi></mrow></math></span> (lesson 09.50).
- Densities of transformed random variables (module 10).
- Normalising flows use layers whose Jacobian is triangular, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>det</mi><mo>⁡</mo><mi>J</mi></mrow></math></span> is just the product of the diagonal.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>det</mi><mo>⁡</mo><mi>J</mi><mi mathvariant="normal">∣</mi></mrow></math></span> is the local volume scaling of a map; it is the correction factor whenever you change variables.

</div>

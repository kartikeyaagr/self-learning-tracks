---
id: "09.30"
course: "ml-maths-course"
module: "09"
slug: "the-chain-rule-as-jacobian-products"
title: "The Chain Rule as Jacobian Products"
keyIdeas: ["For ⟦2813⟧: ⟦2814⟧.", "A network's input-to-output Jacobian is the product of its layers' Jacobians.", "For a scalar loss, ⟦2815⟧."]
question: "⟦2809⟧ with ⟦2810⟧ and ⟦2811⟧. Find ⟦2812⟧ at ⟦2129⟧."
answer: "Step 1, ⟦2822⟧. Step 2, ⟦2823⟧. Step 3, ⟦2824⟧. Step 4, ⟦2825⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Jacobians and local approximation"]
tools: ["NumPy"]
topic: {"id": "09.04", "title": "Jacobians and local approximation"}
---

## The big picture

The one-variable chain rule multiplies derivatives. For maps between vector spaces it multiplies Jacobian matrices. A deep network's gradient is one long product of these.

## The rule

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>J</mi><mrow><mi>G</mi><mo>∘</mo><mi>F</mi></mrow></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>J</mi><mi>G</mi></msub><mo stretchy="false">(</mo><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mtext>  </mtext><msub><mi>J</mi><mi>F</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Shapes check it: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>J</mi><mi>G</mi></msub></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>×</mo><mi>m</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>J</mi><mi>F</mi></msub></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>×</mo><mi>n</mi></mrow></math></span>, the product is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>k</mi><mo>×</mo><mi>n</mi></mrow></math></span>.

```
W = np.array([[1.0, 2], [0, 1]]); x = np.array([1.0, 1])
y = W @ x
print(2 * y @ W)  # [ 6. 14.]
L = lambda v: ((W @ v) ** 2).sum(); h = 1e-6
print(np.round([(L(x + h * e) - L(x - h * e)) / (2 * h) for e in np.eye(2)], 4))  # [ 6. 14.]
```

## Deep networks

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi mathvariant="normal">∇</mi><mi>x</mi></msub><msup><mi>L</mi><mi mathvariant="normal">⊤</mi></msup><mo>=</mo><mfrac><mrow><mi mathvariant="normal">∂</mi><mi mathvariant="normal">ℓ</mi></mrow><mrow><mi mathvariant="normal">∂</mi><msub><mi>h</mi><mi>K</mi></msub></mrow></mfrac><mtext> </mtext><msub><mi>J</mi><msub><mi>f</mi><mi>K</mi></msub></msub><mtext> </mtext><msub><mi>J</mi><msub><mi>f</mi><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow></msub></msub><mo>⋯</mo><msub><mi>J</mi><msub><mi>f</mi><mn>1</mn></msub></msub></mrow></math></span></span>
If the Jacobians tend to stretch, the product explodes; if they shrink, it vanishes. Residual connections give each block a Jacobian <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>I</mi><mo>+</mo><msub><mi>J</mi><mtext>block</mtext></msub></mrow></math></span>, which keeps the product better behaved.

<div class="take">
<div class="lab">Key takeaway</div>

compose maps, multiply Jacobians; a network's gradient is a long product of layer Jacobians.

</div>

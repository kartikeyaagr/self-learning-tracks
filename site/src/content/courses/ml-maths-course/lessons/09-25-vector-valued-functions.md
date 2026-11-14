---
id: "09.25"
course: "ml-maths-course"
module: "09"
slug: "vector-valued-functions"
title: "Vector-Valued Functions"
keyIdeas: ["A map ⟦2742⟧ returns a vector; it is ⟦958⟧ ordinary functions ⟦2743⟧ stacked.", "A neural network layer ⟦2455⟧ is such a map.", "Its derivative is a matrix, the Jacobian (lesson 09.28), not a single vector."]
question: "Write the layer ⟦2739⟧ with ⟦2740⟧ as two component functions and evaluate at ⟦2741⟧."
answer: "Step 1, ⟦2749⟧, ⟦2750⟧. Step 2, at ⟦732⟧: ⟦2751⟧, ⟦2752⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Gradients and vector-valued maps"]
tools: ["NumPy"]
topic: {"id": "09.03", "title": "Gradients and vector-valued maps"}
---

## The big picture

Losses return one number, but most of a network's work is done by maps that return vectors: each layer turns a vector into another vector. This lesson sets up notation for them.

## Components

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true">(</mo><mtable rowspacing="0.16em" columnalign="center" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><msub><mi>F</mi><mn>1</mn></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mi><mi mathvariant="normal">⋮</mi><mpadded height="0em" voffset="0em"><mspace mathbackground="black" width="0em" height="1.5em"></mspace></mpadded></mi></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><msub><mi>F</mi><mi>m</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mstyle></mtd></mtr></mtable><mo fence="true">)</mo></mrow></mrow></math></span></span>
Each <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>F</mi><mi>i</mi></msub><mo>:</mo><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup><mo>→</mo><mi mathvariant="normal">ℝ</mi></mrow></math></span> is an ordinary function with its own gradient.

```
W = np.array([[1.0, -1], [2, 0]])
layer = lambda x: np.maximum(W @ x, 0)
print(layer(np.array([1.0, 3])))  # [0. 2.]
```

## Examples in ML

| Map | From | To |
| --- | --- | --- |
| linear layer <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>n</mi></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>m</mi></msup></mrow></math></span> |
| softmax | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>K</mi></msup></mrow></math></span> logits | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>K</mi></msup></mrow></math></span> probabilities |
| encoder of an autoencoder | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>784</mn></msup></mrow></math></span> image | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>32</mn></msup></mrow></math></span> code |
| a whole network | inputs | outputs |

~~~dg-flow A network is a composition of vector-valued maps.
x in R^n -(layer 1)-> h_1 in R^(d1) -(layer 2)-> h_2 in R^(d2) -(loss)-> *L in R
~~~

<div class="take">
<div class="lab">Key takeaway</div>

a vector-valued map is several ordinary functions stacked; layers are vector-valued maps.

</div>

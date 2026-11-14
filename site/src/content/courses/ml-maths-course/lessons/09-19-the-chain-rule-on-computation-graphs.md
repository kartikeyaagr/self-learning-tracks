---
id: "09.19"
course: "ml-maths-course"
module: "09"
slug: "the-chain-rule-on-computation-graphs"
title: "The Chain Rule on Computation Graphs"
keyIdeas: ["A computation graph writes a function as a sequence of simple operations with known local derivatives.", "The derivative of the output with respect to any node is the sum over paths of the product of local derivatives.", "Reverse mode (backprop) computes this for every node in one backward sweep."]
question: "For ⟦2589⟧ with ⟦391⟧, ⟦2585⟧, ⟦2586⟧, ⟦730⟧, run the forward and backward passes through the nodes ⟦2667⟧, ⟦2668⟧, ⟦2669⟧, ⟦2670⟧."
answer: "Step 1, forward: ⟦2678⟧, ⟦2679⟧, ⟦2680⟧, ⟦2597⟧. Step 2, backward: ⟦2681⟧. Step 3, ⟦2674⟧, ⟦2676⟧, ⟦2675⟧. Step 4, ⟦2682⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["Local change in many variables"]
tools: ["NumPy"]
topic: {"id": "09.02", "title": "Local change in many variables"}
---

## The big picture

Any formula, however big, is built from small steps. Record the steps as a graph, know each step's local derivative, and the chain rule gives every derivative mechanically. This is what PyTorch and JAX do.

## A graph for a tiny loss

~~~dg-flow Forward pass (values) for L = (w x + b - y)^2.
w = 1, x = 2 -(multiply)-> u = 2 -(add b = 0)-> v = 2 -(subtract y = 5)-> e = -3 -(square)-> *L = 9
~~~

## The backward pass

Start with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>L</mi><mo>=</mo><mn>1</mn></mrow></math></span> and walk back, multiplying by each local derivative:

| Node | Local derivative | Gradient arriving |
| --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>=</mo><msup><mi>e</mi><mn>2</mn></msup></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>e</mi><mo>=</mo><mo>−</mo><mn>6</mn></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>e</mi><mo>=</mo><mo>−</mo><mn>6</mn></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>e</mi><mo>=</mo><mi>v</mi><mo>−</mo><mi>y</mi></mrow></math></span> | 1 | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>v</mi><mo>=</mo><mo>−</mo><mn>6</mn></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>v</mi><mo>=</mo><mi>u</mi><mo>+</mo><mi>b</mi></mrow></math></span> | 1 and 1 | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>u</mi><mo>=</mo><mo>−</mo><mn>6</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>b</mi><mo>=</mo><mo>−</mo><mn>6</mn></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi><mo>=</mo><mi>w</mi><mi>x</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∂</mi><mi>L</mi><mi mathvariant="normal">/</mi><mi mathvariant="normal">∂</mi><mi>w</mi><mo>=</mo><mo>−</mo><mn>12</mn></mrow></math></span> |

```
w, x, b, y = 1.0, 2.0, 0.0, 5.0
u = w * x; v = u + b; e = v - y; L = e ** 2
gL = 1.0; ge = gL * 2 * e; gv = ge; gu = gv; gb = gv; gw = gu * x
print(L, gw, gb)  # 9.0 -12.0 -6.0
```

## When paths merge

If a node feeds several later nodes, the gradients from all of them add. In code this is why frameworks **accumulate** into `.grad`.

<div class="take">
<div class="lab">Key takeaway</div>

backprop walks the computation graph from the output back, multiplying local derivatives and adding where paths merge.

</div>

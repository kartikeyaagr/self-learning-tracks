---
id: "03.35"
course: "ml-maths-course"
module: "03"
slug: "composing-and-inverting-affine-maps"
title: "Composing and Inverting Affine Maps"
keyIdeas: ["⟦1268⟧: a composition of affine maps is affine.", "So a stack of linear layers with no activation in between collapses to one layer.", "The inverse of ⟦1226⟧ is ⟦1269⟧."]
question: "Two layers ⟦1264⟧, ⟦1265⟧ with no activation, ⟦1266⟧, ⟦1267⟧. Write the single equivalent layer and compare parameter counts."
answer: "Step 1, substitute: ⟦1291⟧. Step 2, single layer: ⟦1292⟧ parameters. Step 3, two layers: ⟦1293⟧. Step 4, 52 times more parameters, and nothing new computed. A ReLU between them breaks the collapse."
sources: [{"title": "MIT 18.06SC Linear Algebra (Strang)", "url": "https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/", "note": "Read: lectures 1-8, 18-20"}, {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/?topic=linear-algebra", "note": "Read: ch. 1-7"}, {"title": "Boyd, Vandenberghe: Introduction to Applied Linear Algebra (VMLS)", "url": "https://web.stanford.edu/~boyd/vmls/", "note": "Read: ch. 1-11"}]
tags: ["Affine transformations"]
tools: ["NumPy"]
topic: {"id": "03.05", "title": "Affine transformations"}
---

## The big picture

Two facts about affine maps explain a lot about neural networks: stacking them gives nothing new, and they can be merged and undone with simple algebra. The first is why networks need nonlinear activations; the second is why some layers can be folded away after training.

## Compositions collapse

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>g</mi><mo stretchy="false">(</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><msub><mi>A</mi><mn>2</mn></msub><mo stretchy="false">(</mo><msub><mi>A</mi><mn>1</mn></msub><mi>x</mi><mo>+</mo><msub><mi>b</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mo>+</mo><msub><mi>b</mi><mn>2</mn></msub><mo>=</mo><mo stretchy="false">(</mo><msub><mi>A</mi><mn>2</mn></msub><msub><mi>A</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mi>x</mi><mo>+</mo><mo stretchy="false">(</mo><msub><mi>A</mi><mn>2</mn></msub><msub><mi>b</mi><mn>1</mn></msub><mo>+</mo><msub><mi>b</mi><mn>2</mn></msub><mo stretchy="false">)</mo></mrow></math></span></span>
The result is again "matrix times <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> plus a vector".

<div class="ex">
<div class="lab">Worked example</div>

**two layers become one**

Layer 1: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>=</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn></mrow></math></span>. Layer 2: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>3</mn><mi>h</mi><mo>−</mo><mn>4</mn></mrow></math></span>.

Substitute: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>3</mn><mo stretchy="false">(</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mo>−</mo><mn>4</mn><mo>=</mo><mn>6</mn><mi>x</mi><mo>−</mo><mn>1</mn></mrow></math></span>.

At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span>: two layers give <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>=</mo><mn>5</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>11</mn></mrow></math></span>; one layer gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>12</mn><mo>−</mo><mn>1</mn><mo>=</mo><mn>11</mn></mrow></math></span>.

</div>

```
rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(5, 3)), rng.normal(size=5)
W2, b2 = rng.normal(size=(2, 5)), rng.normal(size=2)
x = rng.normal(size=3)
print(np.allclose(W2 @ (W1 @ x + b1) + b2, (W2 @ W1) @ x + (W2 @ b1 + b2)))  # True
```

That is why networks need **nonlinearities** (ReLU, GELU, tanh) between layers. Without them, depth adds nothing.

## Folding layers

~~~dg-flow Batch norm at inference is affine, so it folds into the layer before it.
x -(W x + c)-> conv out -(γ(z - μ)/σ + β)-> y
x -(W' x + c')-> *merged layer, same y
~~~

<div class="ex">
<div class="lab">Worked example</div>

**folding batch norm with numbers**

Layer: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi><mo>=</mo><mn>2</mn><mi>x</mi><mo>+</mo><mn>1</mn></mrow></math></span>. Batch norm: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>3</mn><mfrac><mrow><mi>z</mi><mo>−</mo><mn>5</mn></mrow><mn>2</mn></mfrac><mo>+</mo><mn>0.5</mn></mrow></math></span>.

Substitute: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mn>1.5</mn><mo stretchy="false">(</mo><mn>2</mn><mi>x</mi><mo>−</mo><mn>4</mn><mo stretchy="false">)</mo><mo>+</mo><mn>0.5</mn><mo>=</mo><mn>3</mn><mi>x</mi><mo>−</mo><mn>5.5</mn></mrow></math></span>. One layer, weight 3, bias <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>5.5</mn></mrow></math></span>.

</div>

LoRA's low-rank correction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>W</mi><mo>+</mo><mi>B</mi><mi>A</mi></mrow></math></span> is merged the same way after training.

## Inverting

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>=</mo><mi>A</mi><mi>x</mi><mo>+</mo><mi>b</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is invertible, subtract <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi></mrow></math></span> first, then undo <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msup><mi>f</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>A</mi><mrow><mo>−</mo><mn>1</mn></mrow></msup><mo stretchy="false">(</mo><mi>y</mi><mo>−</mo><mi>b</mi><mo stretchy="false">)</mo></mrow></math></span></span>
The shift was applied last, so it is undone first.

<div class="ex">
<div class="lab">Worked example</div>

**back to dollars**

A model predicts standardised prices <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>z</mi><mo>^</mo></mover></mrow></math></span>, with training <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi><mo>=</mo><mn>410</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo>=</mo><mn>95</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span>.

Invert: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>σ</mi><mover accent="true"><mi>z</mi><mo>^</mo></mover><mo>+</mo><mi>μ</mi></mrow></math></span>. A prediction <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>z</mi><mo>^</mo></mover><mo>=</mo><mn>1.2</mn></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>524</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span> dollars.

An error of 0.2 in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>z</mi></mrow></math></span> units is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.2</mn><mo>×</mo><mn>95</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn><mo>=</mo><mn>19</mn><mo separator="true" lspace="0em" rspace="0em">,</mo><mn>000</mn></mrow></math></span> dollars; the shift cancels in differences.

</div>

## Where this breaks in practice

- **Augmentation.** Rotating or shearing an image is an affine map; apply the same map to bounding boxes and keypoints, or the labels no longer match.

<div class="take">
<div class="lab">Key takeaway</div>

affine maps compose to affine maps, so depth needs nonlinearities. Invert by removing the shift first.

</div>

---
id: "02.12"
course: "ml-maths-course"
module: "02"
slug: "converse-inverse-and-contrapositive"
title: "Converse, Inverse and Contrapositive"
keyIdeas: ["From ⟦554⟧ you can build the contrapositive ⟦562⟧, the converse ⟦563⟧, and the inverse ⟦564⟧.", "Only the contrapositive means the same as the original.", "The converse and the inverse mean the same as each other."]
question: "\"If a model is overfitting, its validation loss rises while training loss falls.\" Your validation loss is falling. What can you conclude? Write the contrapositive and the converse, and say which is valid."
answer: "Step 1, ⟦102⟧ = \"the model is overfitting\", ⟦404⟧ = \"validation loss rises while training loss falls\". The rule is ⟦554⟧. Step 2, contrapositive: ⟦562⟧, \"if validation loss is not rising while training loss falls, the model is not overfitting\". Valid whenever the rule is. Step 3, converse: ⟦563⟧, \"if validation loss rises while training loss falls, the model is overfitting\". Not valid: a distribution shift or a learning-rate spike can give the same pattern. Step 4, your validation loss is falling, so ⟦404⟧ is false. The contrapositive gives ⟦449⟧: no overfitting of this kind yet. Step 5, that is only as good as the rule, which is a rule of thumb."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

From one if-then statement you can build three others by swapping and negating its parts. Exactly one of them is guaranteed to be true when the original is. Knowing which one is the difference between sound debugging and wishful thinking.

## The four forms

Start from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span>.

- **Contrapositive** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>q</mi><mo>⇒</mo><mi mathvariant="normal">¬</mi><mi>p</mi></mrow></math></span>: "if not q, then not p". Always means the same as the original.
- **Converse** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi><mo>⇒</mo><mi>p</mi></mrow></math></span>: "if q, then p". Does **not** follow from the original.
- **Inverse** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>p</mi><mo>⇒</mo><mi mathvariant="normal">¬</mi><mi>q</mi></mrow></math></span>: "if not p, then not q". Does not follow either. It is the contrapositive of the converse, so it means the same as the converse.

~~~dg-flow Only the contrapositive carries the truth of an implication.
p implies q -(equivalent)-> *not q implies not p
q implies p -(equivalent)-> not p implies not q
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an everyday one first**

Original, true: "If an animal is a dog, it has four legs" (take this as true for healthy dogs).

Contrapositive, true: "If an animal does not have four legs, it is not a dog."

Converse, false: "If an animal has four legs, it is a dog." A cat breaks it.

Inverse, false: "If an animal is not a dog, it does not have four legs." The cat breaks it again.

</div>

## Checking with a truth table

Only the contrapositive matches the original on every row:

```
imp = lambda a, b: (not a) or b
rows = [(p, q) for p in (True, False) for q in (True, False)]
orig = [imp(p, q) for p, q in rows]
contra = [imp(not q, not p) for p, q in rows]
conv = [imp(q, p) for p, q in rows]
print(orig == contra, orig == conv)  # True False
```

<div class="ex">
<div class="lab">Worked example</div>

**convexity**

True: "if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is convex and differentiable, every stationary point is a global minimum". (A stationary point is where the derivative is zero.)

Contrapositive, also true: "if some stationary point is not a global minimum, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is not convex". So finding a bad local minimum proves the loss is not convex.

Converse, false: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mn>3</mn><msup><mrow><mi>sin</mi><mo>⁡</mo></mrow><mn>2</mn></msup><mi>x</mi></mrow></math></span> has only one stationary point, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>, the global minimum. But <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>f</mi><mrow><mo mathvariant="normal">′</mo><mo mathvariant="normal">′</mo></mrow></msup><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>2</mn><mo>+</mo><mn>6</mn><mi>cos</mi><mo>⁡</mo><mn>2</mn><mi>x</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>4</mn></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mi>π</mi><mi mathvariant="normal">/</mi><mn>2</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is not convex.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

"if A then B" lets you reason from not-B to not-A. Its converse and inverse are separate claims.

</div>

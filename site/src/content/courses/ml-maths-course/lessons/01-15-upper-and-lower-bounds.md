---
id: "01.15"
course: "ml-maths-course"
module: "01"
slug: "upper-and-lower-bounds"
title: "Upper and Lower Bounds"
keyIdeas: ["An upper bound of a set ⟦8⟧ is any number at least as big as everything in ⟦8⟧. A lower bound is at most everything.", "A set is bounded if it has both an upper and a lower bound.", "Bounds are not unique: if 3 is an upper bound, so is every bigger number."]
question: "Give three upper bounds and one lower bound for ⟦278⟧. Is ⟦279⟧ bounded above? Bounded below? Is the set of possible sigmoid outputs bounded?"
answer: "Step 1, any number ⟦291⟧ is an upper bound: for example 0.9, 1 and 100. Step 2, any number ⟦292⟧ is a lower bound, for example 0. Step 3, ⟦279⟧ has no upper bound (for any ⟦280⟧, the number ⟦293⟧ is in the set and bigger), but 0 is a lower bound. So it is bounded below, not above. Step 4, sigmoid outputs lie in ⟦5⟧, so 1 is an upper bound and 0 a lower bound: the set is bounded."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

"The loss can't go below 0." "Probabilities never exceed 1." Statements like these are bounds. Bounds come before the questions that matter in optimisation: is there a best value, and is it ever reached?

## Upper and lower bounds

For a set of numbers <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>u</mi></mrow></math></span> is an **upper bound** if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>≤</mo><mi>u</mi></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∈</mo><mi>A</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℓ</mi></mrow></math></span> is a **lower bound** if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℓ</mi><mo>≤</mo><mi>a</mi></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∈</mo><mi>A</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is **bounded** if it has both.

<div class="ex">
<div class="lab">Worked example</div>

**bounds of a small set**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span>. Upper bounds: 3, 3.5, 10, 1000, and so on. Lower bounds: 1, 0, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>7</mn></mrow></math></span>, and so on.

Bounds are not unique: once you have one upper bound, every bigger number is another.

</div>

~~~dg-raw
%%viz f_line x=-1,3 iv=0,1,co,S;1,inf,cc,upper_bounds | S = [0, 1). Every number from 1 upward is an upper bound of S, and 1 is the smallest of them.
~~~

## Unbounded sets

<div class="ex">
<div class="lab">Worked example</div>

**three sets**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℕ</mi><mo>=</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mo>…</mo><mtext> </mtext><mo stretchy="false">}</mo></mrow></math></span>: bounded below by 0, not bounded above.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: bounded, by 0 below and 1 above, even though neither bound is in the set.

Cross-entropy loss <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mi>log</mi><mo>⁡</mo><mi>p</mi></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∈</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>: bounded below by 0, but not above, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mi>log</mi><mo>⁡</mo><mi>p</mi></mrow></math></span> grows without limit as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>→</mo><mn>0</mn></mrow></math></span>.

</div>

```
p = np.array([0.5, 0.1, 1e-3, 1e-9])
print(np.round(-np.log(p), 2))  # [ 0.69  2.3   6.91 20.72]
```

A confident wrong prediction can make the loss as large as you like.

## In ML

- Losses are usually bounded below by 0, which is why "loss went negative" signals a bug.
- Gradient clipping forces the gradient norm into a bounded set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>g</mi><mo>:</mo><mi mathvariant="normal">∥</mi><mi>g</mi><mi mathvariant="normal">∥</mi><mo>≤</mo><mi>c</mi><mo stretchy="false">}</mo></mrow></math></span> to stop a single bad batch from blowing up training.

<div class="take">
<div class="lab">Key takeaway</div>

an upper bound is anything at least as big as every element. A set may have no upper bound at all, and many upper bounds when it has one.

</div>

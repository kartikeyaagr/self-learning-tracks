---
id: "01.22"
course: "ml-maths-course"
module: "01"
slug: "the-extreme-value-theorem"
title: "The Extreme Value Theorem"
keyIdeas: ["If ⟦90⟧ is continuous and ⟦385⟧ is closed and bounded, then ⟦90⟧ has a maximum and a minimum on ⟦385⟧.", "Drop \"closed\" or \"bounded\" and the guarantee fails: ⟦7⟧ on ⟦5⟧, ⟦386⟧ on ⟦301⟧.", "Unregularised logistic regression on separable data has no minimiser: weight space is unbounded."]
question: "For each, say whether a minimum is guaranteed and why: (a) ⟦380⟧ on ⟦381⟧, (b) ⟦382⟧ on ⟦355⟧, (c) ⟦383⟧ on ⟦301⟧, (d) a continuous loss on ⟦384⟧."
answer: "Step 1, (a) continuous on a closed, bounded interval: guaranteed. The min is 0 at ⟦387⟧. Step 2, (b) the set is not closed (0 missing), and indeed there is no min: values approach 0 but never reach it. Step 3, (c) the set is not bounded, and there is no min: ⟦386⟧ approaches 0 as ⟦7⟧ grows. Step 4, (d) a closed ball is closed and bounded and the loss is continuous, so the theorem guarantees a minimum."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

Before running an optimiser, it helps to know that there is something to find. The extreme value theorem is the basic guarantee: on a compact set, a continuous function always attains its best value.

## The theorem

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is **continuous** (no jumps) and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> is **closed and bounded**, then there are points in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> reaches its maximum and its minimum.

<div class="ex">
<div class="lab">Worked example</div>

**the theorem at work**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>x</mi></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>: closed and bounded, max 1 at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow></math></span>, min 0 at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup><mo>−</mo><mi>x</mi></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></math></span>: min <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>0.25</mn></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0.5</mn></mrow></math></span>, max 2 at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>2</mn></mrow></math></span>.

</div>

```
x = np.linspace(0, 2, 2001)
f = x ** 2 - x
print(round(float(f.min()), 4), float(x[f.argmin()]))  # -0.25 0.5
print(float(f.max()), float(x[f.argmax()]))  # 2.0 2.0
```

## Breaking each condition

- **Not closed.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>x</mi></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> approaches 1 and 0 but reaches neither: no max and no min.
- **Not bounded.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>x</mi></mrow></msup></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span> approaches 0 as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> grows but never gets there: no min.
- **Not continuous.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>x</mi></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></mrow></math></span>, on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>: values approach 1 but the function jumps down at 1, so no max.

~~~dg-flow Before you optimise, ask two questions.
Loss function -(does a best point exist?)-> argmin is non-empty -(is it unique?)-> *one answer, the optimiser can converge to it
~~~

~~~dg-raw
%%viz f_curve f=x*x x=-1.5,1.5 y=0,2.25 dom=-1,1,cc mark=max | f(x) = x² on the closed interval [−1, 1]. A maximum exists and is reached at the edge.
~~~

~~~dg-raw
%%viz f_curve f=x*x x=-1.5,1.5 y=0,2.25 dom=-1,1,oo mark=max | The same function on the open interval (−1, 1). The values climb toward 1 but never reach it, so there is no maximum.
~~~

## Where this breaks in practice

- **Separable data.** Unregularised logistic regression on perfectly separable data keeps improving as the weights grow. Weight space is not bounded, so no minimiser exists and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi></mrow></math></span> grows forever. L2 regularisation or early stopping fixes it.

<div class="take">
<div class="lab">Key takeaway</div>

continuous function plus closed, bounded set means a best point exists. If training never settles, check which condition is missing.

</div>

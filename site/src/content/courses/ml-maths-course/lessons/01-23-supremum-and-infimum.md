---
id: "01.23"
course: "ml-maths-course"
module: "01"
slug: "supremum-and-infimum"
title: "Supremum and Infimum"
keyIdeas: ["The supremum ⟦397⟧ is the smallest upper bound of ⟦8⟧. The infimum ⟦398⟧ is the largest lower bound.", "The supremum need not be in the set. If it is, it is the maximum.", "Every non-empty set of reals with an upper bound has a supremum (completeness of ⟦19⟧)."]
question: "Logistic regression on linearly separable data, no regularisation. What are ⟦395⟧ and ⟦396⟧? What do you see in training, and what fixes it?"
answer: "Step 1, separable means some ⟦378⟧ puts every point on the right side. Scaling that ⟦378⟧ by 2, 10, 100 keeps every point right but pushes the predicted probabilities toward 0 and 1, so the loss keeps falling toward 0. So ⟦410⟧. Step 2, no finite ⟦378⟧ reaches loss 0, because a sigmoid never outputs exactly 0 or 1. So ⟦411⟧. Step 3, in training: ⟦394⟧ grows without limit, the loss creeps down forever, and probabilities pile up at 0 and 1. Step 4, fix: add ⟦412⟧ (L2), which makes big weights costly and gives a unique finite minimiser; early stopping also works in practice."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

Many losses get closer and closer to a best value without ever reaching it. "Minimum" is the wrong word there, because nothing in the set is the minimum. The supremum and infimum name the value being approached, whether or not it is reached. That is why optimisation papers write inf and sup.

## The definitions

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sup</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> (read "soup of A") is the **smallest upper bound** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>inf</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> (read "inf of A") is the **largest lower bound** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**three sets and their sups**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sup</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>1</mn></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>∈</mo><mi>A</mi></mrow></math></span>, so the maximum exists and equals 1.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sup</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>1</mn></mrow></math></span>, but <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo mathvariant="normal">∉</mo><mi>A</mi></mrow></math></span>, so there is no maximum.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo>−</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><mo>:</mo><mi>n</mi><mo>=</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mo>…</mo><mtext> </mtext><mo stretchy="false">}</mo></mrow></math></span>: the values creep toward 1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sup</mi><mo>⁡</mo><mi>A</mi><mo>=</mo><mn>1</mn></mrow></math></span>, no maximum.

</div>

```
n = np.arange(1, 6)
print(np.round(1 - 1 / n, 3))  # [0.    0.5   0.667 0.75  0.8  ]
```

Why is 1 the *smallest* upper bound? Any smaller candidate, like 0.99, is beaten by some element (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>−</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>200</mn><mo>=</mo><mn>0.995</mn></mrow></math></span>).

~~~dg-raw
%%viz f_line x=-1,2 iv=0,1,oo,S pt=0,inf;1,sup | S = (0, 1) has no maximum or minimum, but its supremum is 1 and its infimum is 0. Neither is in S.
~~~

## Completeness of the reals

Every non-empty set of real numbers with an upper bound has a supremum in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span>. The rationals lack this: the fractions <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>q</mi><mn>2</mn></msup><mo>&lt;</mo><mn>2</mn></mrow></math></span> creep toward <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>2</mn></msqrt></mrow></math></span>, which is not a fraction, so inside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℚ</mi></mrow></math></span> there is no supremum. This property is one reason calculus is built on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span>.

## Sup and inf of a function

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mrow><mi>sup</mi><mo>⁡</mo></mrow><mrow><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></msub><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> means the supremum of the set of values <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>:</mo><mi>x</mi><mo>∈</mo><mi>S</mi><mo stretchy="false">}</mo></mrow></math></span>. For <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>x</mi></mrow></msup></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>sup</mi><mo>⁡</mo><mi>f</mi><mo>=</mo><mn>1</mn></mrow></math></span> (reached at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>0</mn></mrow></math></span>) and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>inf</mi><mo>⁡</mo><mi>f</mi><mo>=</mo><mn>0</mn></mrow></math></span> (never reached).

<div class="take">
<div class="lab">Key takeaway</div>

a max is a sup that is actually reached. Write inf and sup when the best value might only be approached.

</div>

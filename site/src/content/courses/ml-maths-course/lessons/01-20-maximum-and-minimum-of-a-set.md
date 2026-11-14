---
id: "01.20"
course: "ml-maths-course"
module: "01"
slug: "maximum-and-minimum-of-a-set"
title: "Maximum and Minimum of a Set"
keyIdeas: ["The maximum of ⟦8⟧ is an element of ⟦8⟧ that is ⟦346⟧ every element of ⟦8⟧. It is the one upper bound that belongs to ⟦8⟧.", "Finite non-empty sets always have a max and a min. Infinite sets might not.", "⟦5⟧ has no maximum: for any ⟦357⟧, the number ⟦358⟧ is bigger and still inside."]
question: "Find the maximum and minimum, if they exist, of ⟦354⟧, ⟦4⟧, ⟦355⟧ and ⟦356⟧."
answer: "Step 1, ⟦368⟧: max 7, min 1. Step 2, ⟦4⟧: max 1, min 0, both in the set. Step 3, ⟦355⟧: max 1; no min, because 0 is not in the set and any ⟦369⟧ in it has ⟦370⟧ smaller and still in it. Step 4, ⟦371⟧: min 0 (at ⟦372⟧); no max, because the values climb toward 1 without reaching it."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

"The best validation accuracy." "The smallest loss." These are maxima and minima. The surprise is that a set of numbers does not always have one, and that is exactly the situation of an optimiser that keeps improving forever.

## The definitions

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>max</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> is an element <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>∈</mo><mi>A</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>≤</mo><mi>m</mi></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∈</mo><mi>A</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>min</mi><mo>⁡</mo><mi>A</mi></mrow></math></span> is an element <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>∈</mo><mi>A</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>m</mi><mo>≤</mo><mi>a</mi></mrow></math></span> for every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>∈</mo><mi>A</mi></mrow></math></span>.

The key words are "in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>". A maximum is an upper bound that the set actually contains.

```
acc = np.array([0.81, 0.86, 0.88, 0.88, 0.87])
print(acc.max(), acc.min())  # 0.88 0.81
```

Every finite, non-empty set has a max and a min: just compare all the elements.

## When there is no maximum

<div class="ex">
<div class="lab">Worked example</div>

**the open interval**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> has no maximum. Take any <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span> in it. The midpoint <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">/</mi><mn>2</mn></mrow></math></span> is still less than 1, so it is in the set, and it is bigger than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi></mrow></math></span>. Whatever you pick, something beats it.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a loss curve that never arrives**

Losses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo separator="true">,</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0.333</mn><mo separator="true">,</mo><mn>0.25</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span>. The set has max 1 but no min: each value is beaten by the next, and 0 itself is never reached.

</div>

```
n = np.arange(1, 6)
print(np.round(1 / n, 3))  # [1.    0.5   0.333 0.25  0.2  ]
```

The next lessons cover when a best value is guaranteed (closed and bounded sets) and what to call the "best value" when it is never reached (supremum and infimum).

<div class="take">
<div class="lab">Key takeaway</div>

a maximum must belong to the set. Finite sets always have one; open intervals and endless improvements do not.

</div>

~~~dg-raw
%%viz f_line x=-1,2 iv=0,1,cc,[0,1]_has_max_1;0,1,co,[0,1)_has_no_max | Top: 1 is in the set, so it is the maximum. Bottom: 1 is left out, and any number below 1 has a bigger one still in the set.
~~~

---
id: "02.05"
course: "ml-maths-course"
module: "02"
slug: "universal-and-existential-quantifiers"
title: "Universal and Existential Quantifiers"
keyIdeas: ["⟦479⟧ reads \"for every x in S, P(x)\": every element passes.", "⟦480⟧ reads \"there is at least one x in S with P(x)\".", "Python's `all(...)` and `any(...)` are these two quantifiers over a finite collection."]
question: "Batch losses are ⟦475⟧. Decide each claim: (a) ⟦476⟧; (b) ⟦477⟧; (c) ⟦478⟧."
answer: "Step 1, (a) check every loss: 0.3, 0.0, 1.2, 0.4 are all at least 0. True. Step 2, (b) we need one zero: ⟦488⟧. True. Step 3, (c) every loss must be below 1, but ⟦489⟧ is not. One failure is enough, so (c) is false."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

A predicate with a blank can be turned into a statement in a second way: instead of filling in the blank, say how many values make it true. "Every loss is non-negative." "Some example is mislabelled." These two kinds of claim are called quantifiers, and nearly every theorem in ML is built from them.

## The two quantifiers

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>x</mi><mo>∈</mo><mi>S</mi><mo separator="true">,</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, read "for all x in S, P of x". It means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> holds for **every** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi><mi>x</mi><mo>∈</mo><mi>S</mi><mo separator="true">,</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, read "there exists an x in S such that P of x". It means **at least one** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> has property <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span>.

The symbol <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> is an upside-down A (for "All"). The symbol <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> is a backwards E (for "Exists").

<div class="ex">
<div class="lab">Worked example</div>

**claims about a set of numbers**

Let <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi><mo>=</mo><mo stretchy="false">{</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>6</mn><mo separator="true">,</mo><mn>7</mn><mo stretchy="false">}</mo></mrow></math></span>.

"<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is even": false, because of 7.

"<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is odd": true, 7 is odd.

"<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>": true.

</div>

## In code

`all` and `any` are exactly these two quantifiers:

```
S = [2, 4, 6, 7]
print(all(x % 2 == 0 for x in S))  # False
print(any(x % 2 == 1 for x in S))  # True
print(all(x > 0 for x in S))  # True
```

## How much evidence each needs

- To prove a "for all" claim you must cover **every** element. To disprove it, one counterexample is enough.
- To prove an "exists" claim, one example is enough. To disprove it, you must rule out every element.

## The empty set

Over an empty set, every "for all" claim is true (there is nothing to break it) and every "exists" claim is false (there is nothing to find).

```
print(all([]), any([]))  # True False
```

## Where this breaks in practice

- **"It worked on 3 seeds."** That is an "exists" statement about those 3 seeds, not "for all seeds".
- **Empty validation sets.** "Every validation example is correct" is true when the set is empty. Check the size before celebrating.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> means every element passes, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> means at least one does. One counterexample kills a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span>.

</div>

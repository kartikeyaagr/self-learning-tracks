---
id: "02.02"
course: "ml-maths-course"
module: "02"
slug: "negation-conjunction-and-disjunction"
title: "Negation, Conjunction and Disjunction"
keyIdeas: ["⟦449⟧ (not p) is true exactly when ⟦102⟧ is false.", "⟦448⟧ (p and q) is true only when both are true. ⟦447⟧ (p or q) is true when at least one is true.", "Python's `not`, `and`, `or` are the same three connectives."]
question: "Early-stopping rule: stop when ⟦447⟧, with ⟦102⟧ = \"no improvement for 5 epochs\" and ⟦404⟧ = \"epoch is 100\". At epoch 40 there has been no improvement for 6 epochs. Do you stop? What if the rule were ⟦448⟧?"
answer: "Step 1, at epoch 40, ⟦102⟧ is true (6 epochs without improvement is at least 5) and ⟦404⟧ is false (40 is not 100). Step 2, ⟦447⟧ needs at least one true part. ⟦102⟧ is true, so ⟦447⟧ is true: stop. Step 3, ⟦448⟧ needs both. ⟦404⟧ is false, so ⟦448⟧ is false: keep training. Step 4, with ⟦450⟧ the run only stops at epoch 100, and only if it has also stalled. A one-symbol change gives a very different rule."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

Single statements are rarely enough. "Stop if the loss stalls or we run out of budget." "Keep the token if it is not padding and not masked." This lesson defines the three basic ways of joining statements, and the symbols used for them.

## The three connectives

For two statements <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>p</mi></mrow></math></span>, read "not p", is true exactly when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is false. This is **negation**.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∧</mo><mi>q</mi></mrow></math></span>, read "p and q", is true only when both are true. This is **conjunction**.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∨</mo><mi>q</mi></mrow></math></span>, read "p or q", is true when at least one is true. This is **disjunction**.

A way to remember the symbols: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∧</mo></mrow></math></span> looks like the A in "And", and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∨</mo></mrow></math></span> is the other way up.

<div class="ex">
<div class="lab">Worked example</div>

**with numbers**

Let <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> = "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4</mn><mo>&gt;</mo><mn>3</mn></mrow></math></span>" (true) and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> = "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4</mn></mrow></math></span> is odd" (false).

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>q</mi></mrow></math></span>: "4 is not odd", true.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∧</mo><mi>q</mi></mrow></math></span>: false, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> is false.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∨</mo><mi>q</mi></mrow></math></span>: true, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is true.

</div>

## In code

Python spells the connectives as words:

```
p, q = 4 > 3, 4 % 2 == 1
print(not q, p and q, p or q)  # True False True
```

## Combining several connectives

Brackets say what goes first, exactly as in arithmetic. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mo stretchy="false">(</mo><mi>p</mi><mo>∧</mo><mi>q</mi><mo stretchy="false">)</mo></mrow></math></span> is "not both", while <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>p</mi><mo>∧</mo><mi>q</mi></mrow></math></span> is "p is false and q is true".

<div class="ex">
<div class="lab">Worked example</div>

**a data filter**

Keep an example if it is labelled and (long or high quality).

In symbols: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>lab</mtext><mo>∧</mo><mo stretchy="false">(</mo><mtext>long</mtext><mo>∨</mo><mtext>good</mtext><mo stretchy="false">)</mo></mrow></math></span>.

A labelled, short, high-quality example: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>T</mi><mo>∧</mo><mo stretchy="false">(</mo><mi>F</mi><mo>∨</mo><mi>T</mi><mo stretchy="false">)</mo><mo>=</mo><mi>T</mi><mo>∧</mo><mi>T</mi><mo>=</mo><mi>T</mi></mrow></math></span>. Kept.

An unlabelled, long example: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>F</mi><mo>∧</mo><mo stretchy="false">(</mo><mo>…</mo><mo stretchy="false">)</mo><mo>=</mo><mi>F</mi></mrow></math></span>. Dropped, whatever the bracket says.

</div>

## Where this breaks in practice

- **Short-circuiting.** Python stops evaluating `p and q` as soon as `p` is false. That is fine for logic, but if `q` has a side effect (logging, a counter), it silently does not run.
- **English "and".** "Show me the cats and dogs" usually means the union (cats or dogs). Translate the meaning, not the word.

<div class="take">
<div class="lab">Key takeaway</div>

NOT flips, AND needs both, OR needs at least one. Use brackets to say what goes first.

</div>

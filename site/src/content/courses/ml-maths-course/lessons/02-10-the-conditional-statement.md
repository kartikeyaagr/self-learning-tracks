---
id: "02.10"
course: "ml-maths-course"
module: "02"
slug: "the-conditional-statement"
title: "The Conditional Statement"
keyIdeas: ["⟦554⟧ (\"if p then q\") is false in exactly one case: ⟦102⟧ true and ⟦404⟧ false.", "⟦102⟧ is the premise (hypothesis), ⟦404⟧ is the conclusion.", "⟦554⟧ has the same truth table as ⟦555⟧."]
question: "Rule: \"if the gradient check passes, the backward pass is correct\". On which of the four rows of (passes, correct) is the rule broken?"
answer: "Step 1, ⟦102⟧ = \"the gradient check passes\", ⟦404⟧ = \"the backward pass is correct\". Step 2, the rule ⟦554⟧ is false only when ⟦102⟧ is true and ⟦404⟧ is false. Step 3, so it is broken only on the row where the check passes but the backward pass is wrong. Every other row is consistent with the rule."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

"If the data is normalised, training is stable." "If the code is correct, the test passes." ML reasoning is full of if-then sentences. This lesson pins down exactly when such a sentence is true, which is less obvious than it looks.

## The truth table

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span> is read "p implies q" or "if p then q". Its truth table:

| p | q | p implies q |
| --- | --- | --- |
| T | T | T |
| T | F | F |
| F | T | T |
| F | F | T |

Think of it as a **promise**. "If it rains, I'll bring an umbrella."

- It rains and I bring the umbrella: promise kept.
- It rains and I don't: promise broken. This is the only false row.
- It doesn't rain: I made no promise about that day, so whatever I do, I didn't break it.

## The same as "not p or q"

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span> is false only when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is true and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> is false, so it is true when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is false or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> is true: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi><mo>≡</mo><mi mathvariant="normal">¬</mi><mi>p</mi><mo>∨</mo><mi>q</mi></mrow></math></span>.

```
implies = lambda p, q: (not p) or q
print([implies(p, q) for p in (True, False) for q in (True, False)])  # [True, False, True, True]
```

<div class="ex">
<div class="lab">Worked example</div>

**reading a training rule**

"If the learning rate is above 1, the loss diverges." Here <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> = "lr > 1" and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> = "loss diverges".

A run with lr = 0.5 whose loss diverged does not break the rule. The rule says nothing about lr ≤ 1.

Only a run with lr > 1 that did not diverge breaks it.

</div>

## Where this breaks in practice

- **Reading "if" as "only if".** "If the lr is above 1, the loss diverges" does not say small learning rates are safe.

<div class="take">
<div class="lab">Key takeaway</div>

an if-then claim is broken only when the premise holds and the conclusion fails.

</div>

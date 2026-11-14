---
id: "02.13"
course: "ml-maths-course"
module: "02"
slug: "affirming-the-consequent"
title: "Affirming the Consequent"
keyIdeas: ["Affirming the consequent is concluding ⟦102⟧ from ⟦554⟧ and ⟦404⟧. It is invalid.", "The valid moves are modus ponens (from ⟦102⟧, conclude ⟦404⟧) and modus tollens (from ⟦454⟧, conclude ⟦449⟧).", "In experiments, the fix is an ablation: compare runs with and without the suspected cause."]
question: "\"If my fix works, the loss drops. The loss dropped, so my fix works.\" Name the error and describe the experiment that would actually test the fix."
answer: "Step 1, the rule is ⟦554⟧ with ⟦102⟧ = \"the fix works\" and ⟦404⟧ = \"the loss drops\". Step 2, the argument goes from ⟦404⟧ to ⟦102⟧: that is affirming the consequent. Other causes (a new learning-rate schedule, a different seed, a data change) can also make the loss drop. Step 3, to test the fix, run an ablation: the same setup and seeds with and without the fix, changing nothing else. If the loss drops only with the fix, across several seeds, you have real evidence."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

The most common reasoning mistake in debugging and in reading papers is running an implication backwards. It feels natural, which is why it needs a name.

## Two valid moves, two invalid ones

Given the rule <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span>:

- **Modus ponens** (valid): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> holds, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> holds.
- **Modus tollens** (valid): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> fails, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> fails. This is the contrapositive in action.
- **Affirming the consequent** (invalid): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> holds, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> holds.
- **Denying the antecedent** (invalid): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> fails, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> fails.

A brute-force check shows the invalid one has a counterexample row: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> false and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> true satisfies both the rule and the observation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>, yet <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is false.

```
rows = [(p, q) for p in (True, False) for q in (True, False)]
print([(p, q) for p, q in rows if ((not p) or q) and q and not p])  # [(False, True)]
```

<div class="ex">
<div class="lab">Worked example</div>

**the mistake in an ML setting**

Rule: "If my code has the bug, the loss goes to NaN." Observation: the loss went to NaN. Conclusion: "the bug is there."

Not justified. A too-large learning rate, a log of zero or bad input data also send the loss to NaN.

The valid direction is modus tollens: "the loss did not go to NaN, so this bug (which always causes NaN) is not present."

</div>

## Where this breaks in practice

- **Unit tests.** "If the code is correct, the test passes." A passing test does not prove the code correct. A failing test does prove it wrong for that input.
- **Benchmarks.** "A model that reasons well scores high on this benchmark." A high score does not show reasoning; memorised test data also scores high.

<div class="take">
<div class="lab">Key takeaway</div>

from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span> you may go forward from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> or backward from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>q</mi></mrow></math></span>. Never backward from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>.

</div>

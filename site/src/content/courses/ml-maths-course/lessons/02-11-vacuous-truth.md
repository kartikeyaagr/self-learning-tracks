---
id: "02.11"
course: "ml-maths-course"
module: "02"
slug: "vacuous-truth"
title: "Vacuous Truth"
keyIdeas: ["When the premise ⟦102⟧ is false, ⟦554⟧ is true whatever ⟦404⟧ is. This is called vacuous truth.", "It makes \"for all\" claims with conditions work correctly on the cases the condition excludes.", "`all(...)` over an empty collection is `True` for the same reason."]
question: "Is \"every negative eigenvalue of this covariance matrix is larger than 5\" true, given that covariance matrices have no negative eigenvalues?"
answer: "Step 1, rewrite it: \"for every eigenvalue ⟦559⟧, if ⟦560⟧ then ⟦561⟧\". Step 2, a covariance matrix has no negative eigenvalues, so the premise ⟦560⟧ is false for every ⟦559⟧. Step 3, an implication with a false premise is true, so the claim is true for every eigenvalue. It is vacuously true, and it tells you nothing useful."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

The two rows of the truth table where the premise is false often feel wrong: why should "if pigs fly, then 2 + 2 = 5" be true? This lesson shows why that choice is the only one that makes ordinary maths work, and where it can fool you.

## Why false premises give true

Take the claim "for every whole number <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>2</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is prime, then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is odd". It is a true theorem, so it must be true at every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, including <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>4</mn></mrow></math></span>.

At <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>4</mn></mrow></math></span> the premise ("4 > 2 and 4 is prime") is false. If a false premise made the implication false, the theorem would fail at 4. So an implication with a false premise has to count as true.

```
implies = lambda p, q: (not p) or q
is_prime = lambda n: n > 1 and all(n % k for k in range(2, n))
print(all(implies(x > 2 and is_prime(x), x % 2 == 1) for x in range(50)))  # True
```

<div class="ex">
<div class="lab">Worked example</div>

**vacuous claims in practice**

"All the failed runs used lr = 10." If no run failed, this is true, because there is no failed run to break it.

"Every example the model got wrong was mislabelled." If the model got nothing wrong, true again, and meaningless.

</div>

## Empty collections

Python follows the same rule. A "for all" over nothing is true:

```
print(all(x > 100 for x in []))  # True
```

## Where this breaks in practice

- **Empty filters.** A test like `assert all(loss < 1 for loss in val_losses)` passes when `val_losses` is empty, for example after a data-loading bug. Also check the length.
- **Impressive-sounding vacuous results.** A guarantee whose conditions are never met in practice is technically true and practically empty. Check that the premise can actually happen.

<div class="take">
<div class="lab">Key takeaway</div>

a false premise makes an implication true. That keeps theorems consistent, but a vacuously true claim tells you nothing.

</div>

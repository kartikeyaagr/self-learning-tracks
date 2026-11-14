---
id: "02.06"
course: "ml-maths-course"
module: "02"
slug: "negating-quantified-statements"
title: "Negating Quantified Statements"
keyIdeas: ["To negate, flip each quantifier and negate the inside: ⟦492⟧.", "Likewise ⟦493⟧.", "Negating \"and\" gives \"or\" of the negations, and the other way round (De Morgan)."]
question: "Negate: \"for every learning rate ⟦490⟧, gradient descent converges on every ⟦491⟧-smooth convex function\". What would a counterexample look like?"
answer: "Step 1, write the structure: \"for all ⟦490⟧, for all ⟦491⟧-smooth convex ⟦90⟧, GD converges on ⟦90⟧\". Step 2, flip each \"for all\" into \"there exists\" and negate the inside: \"there exist a learning rate ⟦490⟧ and an ⟦491⟧-smooth convex ⟦90⟧ such that GD does not converge on ⟦90⟧\". Step 3, so a counterexample is one concrete pair ⟦505⟧, with a starting point, where the iterates fail to converge. One is enough. Step 4, note: the original is in fact true (for the function values), so no counterexample exists. The exercise is about the form of the negation."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

Being able to write the exact opposite of a claim is a superpower. It tells you what a counterexample would look like, what a proof by contradiction should assume, and what a failing test is really testing. There is one rule to learn, and it is mechanical.

## The flipping rule

To negate, **flip each quantifier and negate the inside**:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">¬</mi><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">(</mo><mi mathvariant="normal">∀</mi><mi>x</mi><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">)</mo><mo>≡</mo><mi mathvariant="normal">∃</mi><mi>x</mi><mtext> </mtext><mi mathvariant="normal">¬</mi><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><mi mathvariant="normal">¬</mi><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">(</mo><mi mathvariant="normal">∃</mi><mi>x</mi><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo fence="false" stretchy="true" minsize="1.2em" maxsize="1.2em">)</mo><mo>≡</mo><mi mathvariant="normal">∀</mi><mi>x</mi><mtext> </mtext><mi mathvariant="normal">¬</mi><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span></span>

Read the first one aloud: "it is not true that every x has P" means the same as "some x does not have P".

Read the second one aloud: "there is no x with P" means "every x fails P".

## Checking the rule in code

On a finite set, both rules can be checked directly:

```
S = range(-3, 4)
P = lambda x: x * x > 4
print((not all(P(x) for x in S)) == any(not P(x) for x in S))  # True
print((not any(P(x) for x in S)) == all(not P(x) for x in S))  # True
```

<div class="ex">
<div class="lab">Worked example</div>

**negating step by step**

Claim: "every image in the dataset has a label". In symbols: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>x</mi><mo>∈</mo><mi>D</mi><mo separator="true">,</mo><mtext> labelled</mtext><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>.

Negation: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi><mi>x</mi><mo>∈</mo><mi>D</mi><mo separator="true">,</mo><mtext> </mtext><mi mathvariant="normal">¬</mi><mtext>labelled</mtext><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>. In words: "at least one image has no label".

To prove the claim false, you only need to find one unlabelled image.

</div>

## Negating the inside

The inside is often an "and" or an "or". Negate it with De Morgan's laws: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mo stretchy="false">(</mo><mi>a</mi><mo>∧</mo><mi>b</mi><mo stretchy="false">)</mo><mo>≡</mo><mi mathvariant="normal">¬</mi><mi>a</mi><mo>∨</mo><mi mathvariant="normal">¬</mi><mi>b</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mo stretchy="false">(</mo><mi>a</mi><mo>∨</mo><mi>b</mi><mo stretchy="false">)</mo><mo>≡</mo><mi mathvariant="normal">¬</mi><mi>a</mi><mo>∧</mo><mi mathvariant="normal">¬</mi><mi>b</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**a claim with "and" inside**

Claim: "every weight is between <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn></mrow></math></span>": <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>i</mi><mo separator="true">,</mo><mtext> </mtext><mo stretchy="false">(</mo><msub><mi>w</mi><mi>i</mi></msub><mo>≥</mo><mo>−</mo><mn>1</mn><mo>∧</mo><msub><mi>w</mi><mi>i</mi></msub><mo>≤</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

Negation: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi><mi>i</mi><mo separator="true">,</mo><mtext> </mtext><mo stretchy="false">(</mo><msub><mi>w</mi><mi>i</mi></msub><mo>&lt;</mo><mo>−</mo><mn>1</mn><mo>∨</mo><msub><mi>w</mi><mi>i</mi></msub><mo>&gt;</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. "Some weight is below <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> or above <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn></mrow></math></span>."

Note how <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≥</mo></mrow></math></span> became <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>&lt;</mo></mrow></math></span> and "and" became "or".

</div>

## Where this breaks in practice

- **Negating only the first quantifier.** "Not every model beats the baseline" is not "no model beats the baseline". It means "some model does not beat it".
- **Forgetting to flip the inequality.** The negation of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>≤</mo><mn>1</mn></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>1</mn></mrow></math></span>, not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>≥</mo><mn>1</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

flip every quantifier, then negate the condition at the end, using De Morgan for "and" and "or".

</div>

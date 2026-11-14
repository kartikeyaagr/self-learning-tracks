---
id: "02.16"
course: "ml-maths-course"
module: "02"
slug: "direct-proof-and-proof-by-contrapositive"
title: "Direct Proof and Proof by Contrapositive"
keyIdeas: ["A direct proof assumes ⟦102⟧ and reasons step by step to ⟦404⟧.", "A proof by contrapositive assumes ⟦454⟧ and reaches ⟦449⟧.", "Checking many examples in code is evidence, not proof."]
question: "Prove by contrapositive: if ⟦583⟧ is even then ⟦233⟧ is even."
answer: "Step 1, the contrapositive is \"if ⟦233⟧ is odd, then ⟦583⟧ is odd\". Step 2, write an odd ⟦233⟧ as ⟦588⟧. Step 3, ⟦603⟧, which is odd. Step 4, the contrapositive is proved, so the original claim holds."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

Reading proofs is part of reading ML papers, and a few patterns cover most of them. This lesson covers the two most common: going straight from premise to conclusion, and going backward through the contrapositive.

## Direct proof

Assume <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span>. Use definitions and known facts until you reach <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**a direct proof**

Claim: the sum of two even numbers is even.

Assume <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>=</mo><mn>2</mn><mi>j</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>=</mo><mn>2</mn><mi>k</mi></mrow></math></span>. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>a</mi><mo>+</mo><mi>b</mi><mo>=</mo><mn>2</mn><mi>j</mi><mo>+</mo><mn>2</mn><mi>k</mi><mo>=</mo><mn>2</mn><mo stretchy="false">(</mo><mi>j</mi><mo>+</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></math></span>, which is even.

</div>

## Proof by contrapositive

Since <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span> means the same as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>q</mi><mo>⇒</mo><mi mathvariant="normal">¬</mi><mi>p</mi></mrow></math></span>, you may prove the second instead. This is easier when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>q</mi></mrow></math></span> gives you something concrete to work with.

<div class="ex">
<div class="lab">Worked example</div>

**why the contrapositive is easier here**

Claim: if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup></mrow></math></span> is even, then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> is even.

Direct is awkward: knowing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup></mrow></math></span> is even does not obviously tell you about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>.

Contrapositive: if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>2</mn><mi>k</mi><mo>+</mo><mn>1</mn></mrow></math></span> then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup><mo>=</mo><mn>2</mn><mo stretchy="false">(</mo><mn>2</mn><msup><mi>k</mi><mn>2</mn></msup><mo>+</mo><mn>2</mn><mi>k</mi><mo stretchy="false">)</mo><mo>+</mo><mn>1</mn></mrow></math></span> is odd. Done.

</div>

## Testing is not proving

Code can check a claim on many cases. It cannot cover the infinitely many others:

```
print(all(n % 2 == 0 for n in range(100) if (n * n) % 2 == 0))  # True
```

This makes the claim believable. The algebra above is what makes it certain.

## Where this breaks in practice

- **Proofs that check a few cases.** "We verified the bound on 5 datasets" is evidence, not a proof.

<div class="take">
<div class="lab">Key takeaway</div>

prove <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span> directly, or prove <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">¬</mi><mi>q</mi><mo>⇒</mo><mi mathvariant="normal">¬</mi><mi>p</mi></mrow></math></span> when that is easier. Checks in code support a claim but do not prove it.

</div>

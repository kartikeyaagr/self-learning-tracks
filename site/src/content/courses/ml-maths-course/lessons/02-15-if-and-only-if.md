---
id: "02.15"
course: "ml-maths-course"
module: "02"
slug: "if-and-only-if"
title: "If and Only If"
keyIdeas: ["⟦581⟧ (\"p if and only if q\") means both ⟦554⟧ and ⟦563⟧.", "It is true exactly when ⟦102⟧ and ⟦404⟧ have the same truth value.", "To prove an \"iff\", prove both directions separately."]
question: "Prove: for a real number ⟦7⟧, ⟦580⟧."
answer: "Step 1, direction ⟦594⟧: ⟦595⟧. Step 2, direction ⟦596⟧: if ⟦597⟧, then ⟦598⟧, so ⟦599⟧. That is the contrapositive, so the direction holds. Step 3, both directions hold, so the \"iff\" holds."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

Definitions and the best theorems are two-way: a condition that is both necessary and sufficient. "A matrix is invertible if and only if its determinant is non-zero." Such a result lets you swap one condition for the other freely.

## The definition

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext><mi>q</mi></mrow></math></span> means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>p</mi><mo>⇒</mo><mi>q</mi><mo stretchy="false">)</mo><mo>∧</mo><mo stretchy="false">(</mo><mi>q</mi><mo>⇒</mo><mi>p</mi><mo stretchy="false">)</mo></mrow></math></span>. It is true when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> are both true or both false:

| p | q | p iff q |
| --- | --- | --- |
| T | T | T |
| T | F | F |
| F | T | F |
| F | F | T |

It is the opposite of XOR: "iff" says the two agree, XOR says they differ.

```
iff = lambda p, q: p == q
print([iff(p, q) for p in (True, False) for q in (True, False)])  # [True, False, False, True]
```

## Proving an iff

Prove the two directions separately. The two halves often need different ideas.

<div class="ex">
<div class="lab">Worked example</div>

**both directions with numbers**

Claim: a whole number <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> is even if and only if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup></mrow></math></span> is even.

(<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>⇒</mo></mrow></math></span>) If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>2</mn><mi>k</mi></mrow></math></span>, then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup><mo>=</mo><mn>4</mn><msup><mi>k</mi><mn>2</mn></msup><mo>=</mo><mn>2</mn><mo stretchy="false">(</mo><mn>2</mn><msup><mi>k</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></math></span>, even.

(<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>⇐</mo></mrow></math></span>) By contrapositive: if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>=</mo><mn>2</mn><mi>k</mi><mo>+</mo><mn>1</mn></mrow></math></span> is odd, then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>n</mi><mn>2</mn></msup><mo>=</mo><mn>2</mn><mo stretchy="false">(</mo><mn>2</mn><msup><mi>k</mi><mn>2</mn></msup><mo>+</mo><mn>2</mn><mi>k</mi><mo stretchy="false">)</mo><mo>+</mo><mn>1</mn></mrow></math></span> is odd.

</div>

## Where you will see it

- "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is invertible <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext><mi>det</mi><mo>⁡</mo><mi>A</mi><mo mathvariant="normal">≠</mo><mn>0</mn></mrow></math></span>."
- "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> is convex <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext><msup><mi>f</mi><mrow><mo mathvariant="normal">′</mo><mo mathvariant="normal">′</mo></mrow></msup><mo>≥</mo><mn>0</mn></mrow></math></span> everywhere" (for twice-differentiable <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> on an interval).
- "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> are independent <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span> for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo separator="true">,</mo><mi>y</mi></mrow></math></span>."

## Where this breaks in practice

- **One direction quoted as both.** "Uncorrelated variables are independent" is false in general. Independence implies zero correlation, not the other way round.

<div class="take">
<div class="lab">Key takeaway</div>

"iff" is two implications. Prove each direction, and only swap conditions freely when both hold.

</div>

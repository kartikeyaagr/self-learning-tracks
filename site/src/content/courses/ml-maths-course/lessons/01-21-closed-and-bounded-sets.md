---
id: "01.21"
course: "ml-maths-course"
module: "01"
slug: "closed-and-bounded-sets"
title: "Closed and Bounded Sets"
keyIdeas: ["A set in ⟦2⟧ that is both closed and bounded is called compact.", "Compact sets have no \"escape routes\": you cannot run off to infinity or creep toward a missing edge.", "Intervals ⟦294⟧, closed discs, boxes ⟦374⟧ and the probability simplex are compact."]
question: "Which are compact: ⟦4⟧, ⟦5⟧, ⟦301⟧, the closed unit disc, ⟦373⟧, ⟦2⟧? For each non-compact one, say which property fails."
answer: "Step 1, ⟦4⟧: closed and bounded, compact. Step 2, ⟦5⟧: bounded but not closed (edges missing), not compact. Step 3, ⟦301⟧: closed but not bounded, not compact. Step 4, the closed unit disc: compact. Step 5, the ball ⟦379⟧: closed (non-strict inequality) and bounded (radius 10), compact. Step 6, ⟦2⟧: closed but not bounded, not compact."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

## The big picture

There are two ways a search for the best point can fail: the best is approached at a missing edge, or it lies "at infinity". A set that rules out both is called **compact**, and on compact sets the best point is guaranteed to exist (next lesson).

## Closed plus bounded

In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span>, a set is **compact** when it is

- **closed**: it contains its whole boundary, so no missing edge, and
- **bounded**: it fits inside some ball, so it does not run off to infinity.

<div class="ex">
<div class="lab">Worked example</div>

**compact or not?**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span> and the box <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><msup><mo stretchy="false">]</mo><mi>d</mi></msup></mrow></math></span>: compact.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>: bounded but missing its ends.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></math></span>: closed but unbounded.

The probability simplex <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>p</mi><mo>:</mo><msub><mi>p</mi><mi>i</mi></msub><mo>≥</mo><mn>0</mn><mo separator="true">,</mo><mo>∑</mo><msub><mi>p</mi><mi>i</mi></msub><mo>=</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>: closed and bounded, so compact.

</div>

## Making a search space compact

Weight space <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> is not compact. Two common tricks make it effectively compact:

- **Constraints.** Max-norm constraints keep <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∥</mi><mi>w</mi><mi mathvariant="normal">∥</mi><mo>≤</mo><mi>c</mi></mrow></math></span>, a closed ball.
- **Penalties.** Adding <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>λ</mi><mi mathvariant="normal">∥</mi><mi>w</mi><msup><mi mathvariant="normal">∥</mi><mn>2</mn></msup></mrow></math></span> to the loss makes huge weights so costly that the search never goes there.

```
w = np.array([3.0, 4.0])
c = 2.0
w_proj = w * min(1.0, c / np.linalg.norm(w))
print(w_proj, np.linalg.norm(w_proj))  # [1.2 1.6] 2.0
```

This projects <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>w</mi></mrow></math></span> back onto the closed ball of radius 2, a compact set.

<div class="take">
<div class="lab">Key takeaway</div>

compact means closed and bounded. It removes both ways a best point can escape.

</div>

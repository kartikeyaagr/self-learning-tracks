---
id: "02.17"
course: "ml-maths-course"
module: "02"
slug: "proof-by-contradiction-and-counterexample"
title: "Proof by Contradiction and Counterexample"
keyIdeas: ["Proof by contradiction assumes the claim is false and derives something impossible.", "A single counterexample disproves a \"for all\" claim.", "Searching for counterexamples in code is a fast way to test a conjecture before trying to prove it."]
question: "Disprove: \"for all real ⟦604⟧, ⟦605⟧\"."
answer: "Step 1, a \"for all\" claim needs one counterexample. Step 2, try ⟦622⟧, ⟦623⟧: ⟦624⟧ but ⟦625⟧. Step 3, ⟦626⟧, so the claim is false."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python", "NumPy"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

Two more proof patterns finish the toolkit. Contradiction is the standard way to prove that something is impossible. A counterexample is the fastest way to kill a claim that is too strong.

## Proof by contradiction

Assume the statement is false, and show that this leads to something impossible, like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>=</mo><mn>0</mn></mrow></math></span> or a number that is both even and odd. Then the statement must be true.

<div class="ex">
<div class="lab">Worked example</div>

**there is no largest whole number**

Assume there is a largest whole number <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span>.

Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi><mo>+</mo><mn>1</mn></mrow></math></span> is a whole number and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi><mo>+</mo><mn>1</mn><mo>&gt;</mo><mi>M</mi></mrow></math></span>.

That contradicts "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>M</mi></mrow></math></span> is the largest", so no largest whole number exists.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**XOR needs a hidden layer**

Claim: no single threshold unit <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><msub><mi>w</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><msub><mi>x</mi><mn>2</mn></msub><mo>+</mo><mi>b</mi><mo>&gt;</mo><mn>0</mn><mo stretchy="false">]</mo></mrow></math></span> computes XOR.

Assume one does. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>↦</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>b</mi><mo>≤</mo><mn>0</mn></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>↦</mo><mn>1</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>2</mn></msub><mo>+</mo><mi>b</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>↦</mo><mn>1</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>1</mn></msub><mo>+</mo><mi>b</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span>; <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>↦</mo><mn>0</mn></mrow></math></span> gives <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><mo>+</mo><mi>b</mi><mo>≤</mo><mn>0</mn></mrow></math></span>.

Adding the middle two: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>w</mi><mn>1</mn></msub><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><mo>+</mo><mi>b</mi><mo>&gt;</mo><mo>−</mo><mi>b</mi><mo>≥</mo><mn>0</mn></mrow></math></span>, contradicting the last one. So no such unit exists.

</div>

## Counterexamples

To knock down "for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>", show one <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> fails. Code is good at finding them:

```
cands = [(a, b) for a in range(-3, 4) for b in range(-3, 4)]
bad = [(a, b) for a, b in cands if (a + b) ** 2 != a * a + b * b]
print(bad[:2])  # [(-3, -3), (-3, -2)]
```

So "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>a</mi><mo>+</mo><mi>b</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo>=</mo><msup><mi>a</mi><mn>2</mn></msup><mo>+</mo><msup><mi>b</mi><mn>2</mn></msup></mrow></math></span>" is false; the missing term is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mi>a</mi><mi>b</mi></mrow></math></span>.

A search can also suggest a claim is true. Try many weights for a single threshold unit on XOR and count the best number of correct rows:

```
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
grid = np.linspace(-2, 2, 21)
best = max(((X @ np.array([w1, w2]) + b > 0) == y).sum() for w1 in grid for w2 in grid for b in grid)
print(best)  # 3
```

The best is 3 out of 4, which matches the proof.

## Where this breaks in practice

- **A failed search is not a proof.** Not finding a counterexample in a grid means only that the grid had none.

<div class="take">
<div class="lab">Key takeaway</div>

contradiction proves impossibility; one counterexample disproves a "for all". Search in code first, then prove.

</div>

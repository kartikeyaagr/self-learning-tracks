---
id: "01.05"
course: "ml-maths-course"
module: "01"
slug: "set-builder-notation"
title: "Set-Builder Notation"
keyIdeas: ["⟦88⟧ is read \"the set of all x in A such that P(x)\". Before the colon is where to look, after it is the test.", "⟦89⟧ is the image of ⟦8⟧: apply ⟦90⟧ to every element and collect the results.", "Set-builder notation translates directly into a comprehension or a masked array."]
question: "Write in set-builder notation, then in Python: (a) the hard negatives, items ⟦82⟧ other than the positive ⟦83⟧ with score ⟦84⟧, for ⟦85⟧; (b) all pairs of indices ⟦86⟧ from ⟦87⟧. Give each result."
answer: "(a) Step 1, where to look: indices ⟦104⟧. Step 2, the test: ⟦105⟧ and ⟦84⟧. So the set is ⟦106⟧. Step 3, check: index 0 is the positive, 2 and 3 have scores 0.7 and 0.6, so the result is ⟦107⟧. Python: `[j for j in range(5) if j != 0 and s[j] > 0.5]`. (b) The set is ⟦108⟧. Listing: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), six pairs, which is ⟦109⟧. Python: `[(i, j) for i in range(4) for j in range(4) if i < j]`."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["Python", "NumPy"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

Set-builder notation is the compact way mathematicians write "keep the elements that pass a test". Papers use it constantly: for the set of misclassified points, for hard negatives, for the argmax. Once you can read it aloud and turn it into code, a lot of notation stops being scary.

## The notation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><mi>A</mi><mo>:</mo><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span></span>
Read it aloud as "the set of all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> such that <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> is true".

- Before the colon: **where to look**, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>A</mi></mrow></math></span>.
- After the colon: **the test**, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>.

Some books write a vertical bar instead of the colon; it means the same.

<div class="ex">
<div class="lab">Worked example</div>

**three sets built by rules**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>n</mi><mo>∈</mo><mi mathvariant="normal">ℕ</mi><mo>:</mo><mi>n</mi><mo>&lt;</mo><mn>5</mn><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>n</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>20</mn><mo stretchy="false">}</mo><mo>:</mo><mi>n</mi><mtext> is divisible by </mtext><mn>6</mn><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mn>6</mn><mo separator="true">,</mo><mn>12</mn><mo separator="true">,</mo><mn>18</mn><mo stretchy="false">}</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><mi mathvariant="normal">ℝ</mi><mo>:</mo><msup><mi>x</mi><mn>2</mn></msup><mo>&lt;</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span> is the interval <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>: infinitely many elements, written in one line.

</div>

~~~dg-raw
%%viz setb | Pick a rule and a universe. The highlighted numbers are the set the rule builds.
~~~

## In code: comprehensions

A Python set comprehension even looks like the maths:

```
evens = {n for n in range(10) if n % 2 == 0}
print(sorted(evens))  # [0, 2, 4, 6, 8]
```

Maths: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>n</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>9</mn><mo stretchy="false">}</mo><mo>:</mo><mi>n</mi><mtext> is even</mtext><mo stretchy="false">}</mo></mrow></math></span>.

## Images: apply a function to every element

A second form puts a formula before the colon:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>:</mo><mi>x</mi><mo>∈</mo><mi>A</mi><mo stretchy="false">}</mo></mrow></math></span></span>
"every value <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>". This is the **image** of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> under <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span>.

```
words = ["the", "cat", "sat", "on", "the", "mat"]
print(sorted({len(w) for w in words}))  # [2, 3]
```

Maths: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mtext>len</mtext><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo><mo>:</mo><mi>w</mi><mo>∈</mo><mtext>words</mtext><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span>. Repeats vanish.

## Pairs that satisfy a rule

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>i</mi><mo separator="true">,</mo><mi>j</mi><mo stretchy="false">)</mo><mo>:</mo><mn>0</mn><mo>≤</mo><mi>i</mi><mo>&lt;</mo><mi>j</mi><mo>&lt;</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span> is every pair of different indices, counted once.

```
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
print(pairs)  # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
```

Pairwise losses in contrastive learning and ranking loop over exactly this set.

## A rule from a paper

"Hard negatives" for an anchor are the items that are not the positive <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> but still score above a threshold: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>j</mi><mo mathvariant="normal">≠</mo><mi>p</mi><mo>:</mo><msub><mi>s</mi><mi>j</mi></msub><mo>&gt;</mo><mn>0.5</mn><mo stretchy="false">}</mo></mrow></math></span>.

```
s = np.array([0.9, 0.3, 0.7, 0.6, 0.1])
idx = np.arange(len(s))
print(idx[(idx != 0) & (s > 0.5)])  # [2 3]
```

~~~dg-flow Every set-builder expression is "where to look", then "the test".
Where to look: x in A -(apply the test P to each x)-> True / False mask -(keep the Trues)-> *the set {x in A : P(x)}
~~~

<div class="take">
<div class="lab">Key takeaway</div>

read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><mi>A</mi><mo>:</mo><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span> as "x in A such that P". In code it is a comprehension or a mask.

</div>

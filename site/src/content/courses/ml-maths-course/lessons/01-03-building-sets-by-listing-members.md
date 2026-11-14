---
id: "01.03"
course: "ml-maths-course"
module: "01"
slug: "building-sets-by-listing-members"
title: "Building Sets by Listing Members"
keyIdeas: ["Roster notation lists the elements inside braces: ⟦53⟧.", "Order and repeats do not matter: ⟦54⟧.", "⟦55⟧ is the empty set, ⟦56⟧ is the number of elements, and ⟦42⟧ means every element of ⟦8⟧ is in ⟦41⟧."]
question: "The labels in a batch are ⟦51⟧. Write the set of labels that appear, its size, and say whether ⟦52⟧ it."
answer: "Step 1, collect the distinct values: 2, 0 and 1. As a set, ⟦68⟧; order and repeats are dropped. Step 2, its size is ⟦69⟧. Step 3, every element of ⟦37⟧ (that is, 0 and 1) is in ⟦68⟧, so yes, ⟦70⟧. In code: `set(labels)` gives the set and `len(...)` its size."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["Python"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

The simplest way to describe a small set is to write its elements down. This is called **roster notation**, and it is what you use for label sets, small vocabularies and lists of settings.

## Roster notation

Write the elements inside curly braces, separated by commas:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mtext>cat</mtext><mo separator="true">,</mo><mtext>dog</mtext><mo separator="true">,</mo><mtext>bird</mtext><mo stretchy="false">}</mo><mo separator="true">,</mo><mspace width="2em"/><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>9</mn><mo stretchy="false">}</mo></mrow></math></span></span>
The three dots mean "continue the obvious pattern". Two rules always hold:

- **No order.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span> are the same set.
- **No repeats.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span> is just <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span>.

Python's `set` type follows exactly these rules:

```
A = {3, 1, 2, 2}
print(A == {1, 2, 3})  # True
print(len(A))  # 3
print(2 in A, 5 in A)  # True False
```

## Notation you will see on every page

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>A</mi></mrow></math></span>: "x is in A". <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo mathvariant="normal">∉</mo><mi>A</mi></mrow></math></span>: "x is not in A".
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>⊆</mo><mi>B</mi></mrow></math></span>: "A is a subset of B", every element of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is also in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∅</mi></mrow></math></span>: the empty set, the set with nothing in it.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span>: the size of a finite set.

<div class="ex">
<div class="lab">Worked example</div>

**a vocabulary**

The sentence "the cat sat on the mat" has 6 words but only 5 distinct ones.

Its vocabulary is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mtext>cat</mtext><mo separator="true">,</mo><mtext>mat</mtext><mo separator="true">,</mo><mtext>on</mtext><mo separator="true">,</mo><mtext>sat</mtext><mo separator="true">,</mo><mtext>the</mtext><mo stretchy="false">}</mo></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>V</mi><mi mathvariant="normal">∣</mi><mo>=</mo><mn>5</mn></mrow></math></span>. The repeated "the" disappears.

</div>

```
words = ["the", "cat", "sat", "on", "the", "mat"]
print(sorted(set(words)))  # ['cat', 'mat', 'on', 'sat', 'the']
print(len(set(words)))  # 5
```

<div class="ex">
<div class="lab">Worked example</div>

**the empty set is a subset of everything**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∅</mi><mo>⊆</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span> is true, because there is no element of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∅</mi></mrow></math></span> that fails to be in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi mathvariant="normal">∅</mi><mi mathvariant="normal">∣</mi><mo>=</mo><mn>0</mn></mrow></math></span>. A batch with no examples of class 7 gives the empty set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>i</mi><mo>:</mo><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mn>7</mn><mo stretchy="false">}</mo><mo>=</mo><mi mathvariant="normal">∅</mi></mrow></math></span>.

</div>

## Where this breaks in practice

- **Python's `{}` is not the empty set.** `{}` makes an empty dictionary; use `set()`.
- **Sets of floats.** `{0.1 + 0.2, 0.3}` has two elements, because the floats differ in the last digit.

<div class="take">
<div class="lab">Key takeaway</div>

list small sets in braces. Order and repeats never matter, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∈</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>⊆</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∅</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>A</mi><mi mathvariant="normal">∣</mi></mrow></math></span> are the four symbols to read fluently.

</div>

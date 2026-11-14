---
id: "01.02"
course: "ml-maths-course"
module: "01"
slug: "equivalent-sets"
title: "Equivalent Sets"
keyIdeas: ["Two sets are equal when they have exactly the same elements. Prove it by showing each is a subset of the other.", "Two sets are equivalent (same size) when their elements can be paired off one to one. That pairing is a bijection.", "⟦11⟧ and ⟦13⟧ are the same size; ⟦19⟧ is strictly bigger than both."]
question: "A tokenizer maps 50,000 strings to ids. Is the vocabulary equal to ⟦39⟧, equivalent to it, or neither? Why does the answer matter for a model?"
answer: "Step 1, equal would mean the same elements. The vocabulary contains strings like \"cat\" and the other set contains numbers, so they are not equal. Step 2, equivalent means a one-to-one pairing with nothing left over. The tokenizer's lookup table gives each string exactly one id and each id exactly one string, so it is a bijection, and the sets are equivalent. Step 3, why it matters: the model can work entirely with ids (numbers that index rows of an embedding table) and the bijection lets you translate back to text without losing anything."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["Python"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

"Same" can mean two different things for sets. Sometimes we mean "the very same elements". Sometimes we only mean "the same number of elements". Mixing them up causes confusion, so mathematics gives them separate names.

## Equal sets

Two sets are **equal**, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>B</mi></mrow></math></span>, when they have exactly the same elements. Order and repeats do not matter.

The standard proof has two steps. First show every element of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>, written <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>⊆</mo><mi>B</mi></mrow></math></span> ("A is a subset of B"). Then show every element of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> is in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, written <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>⊆</mo><mi>A</mi></mrow></math></span>. Both together give <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mi>B</mi></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**equal sets**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>: same elements, different order.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><mi mathvariant="normal">ℤ</mi><mo>:</mo><msup><mi>x</mi><mn>2</mn></msup><mo>=</mo><mn>4</mn><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span>: the rule and the list describe the same two numbers.

</div>

## Equivalent sets

Two sets are **equivalent**, or have the same **cardinality** (size), when you can pair their elements off one to one with nothing left over on either side. Such a pairing is called a **bijection**.

<div class="ex">
<div class="lab">Worked example</div>

**equal versus equivalent**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mtext>a</mtext><mo separator="true">,</mo><mtext>b</mtext><mo separator="true">,</mo><mtext>c</mtext><mo stretchy="false">}</mo></mrow></math></span> are not equal, but they are equivalent: pair 1 with a, 2 with b, 3 with c.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mtext>a</mtext><mo separator="true">,</mo><mtext>b</mtext><mo stretchy="false">}</mo></mrow></math></span> are not equivalent: whatever you do, one number is left without a partner.

</div>

```
vocab = ["the", "cat", "sat"]
to_id = {w: i for i, w in enumerate(vocab)}
to_word = {i: w for w, i in to_id.items()}
print(to_id)  # {'the': 0, 'cat': 1, 'sat': 2}
print(to_word[1])  # cat
```

The two dictionaries are the bijection and its reverse.

~~~dg-raw
%%viz f_map A=a,b,c B=1,2,3 f=a:1,b:2,c:3 an=A bn=B | A one-to-one pairing. Every element of A gets its own partner in B, none left over, so A and B are equivalent.
~~~

## Infinite sets

For infinite sets, "same size" gives surprises.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℕ</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℤ</mi></mrow></math></span> are the same size. List the integers as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>3</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span> and pair them with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span>. Every integer gets a partner.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> is strictly bigger than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℕ</mi></mrow></math></span>. No list, however long, can contain every real number (Cantor's diagonal argument).

The practical lesson: a function on a finite set, like token id to embedding, can be stored as a table. A function on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> cannot, so we describe it with a formula and parameters instead.

<div class="take">
<div class="lab">Key takeaway</div>

equal means same elements; equivalent means same size. A lookup table that pairs things one to one is a bijection.

</div>

---
id: "01.09"
course: "ml-maths-course"
module: "01"
slug: "union-and-intersection"
title: "Union and Intersection"
keyIdeas: ["Union ⟦187⟧: in ⟦8⟧ or in ⟦41⟧ (or both). Intersection ⟦188⟧: in both.", "Union and intersection are \"or\" and \"and\" for sets. In Python they are `|` and `&`.", "Data leakage means the training and test sets have a non-empty intersection."]
question: "Training ids are ⟦185⟧ and test ids are ⟦186⟧. Compute the union and intersection, say what the intersection means, and give the cleaned test set."
answer: "Step 1, union: everything in either, ⟦200⟧. Step 2, intersection: what is in both, ⟦199⟧. Step 3, a non-empty intersection means examples 13 and 14 were seen in training, so the test score would be too optimistic (leakage). Step 4, remove them: the cleaned test set is ⟦201⟧, which is the test set minus the training set."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Set operations"]
tools: ["Python", "NumPy"]
topic: {"id": "01.02", "title": "Set operations"}
---

## The big picture

Once you can describe single sets, you want to combine them: "users who clicked **or** bought", "users who clicked **and** bought". Union and intersection are those two words, made precise.

## The definitions

| Operation | Symbol | Read aloud | Elements |
| --- | --- | --- | --- |
| Union | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∪</mo><mi>B</mi></mrow></math></span> | "A union B" | in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> or in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> (or both) |
| Intersection | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∩</mo><mi>B</mi></mrow></math></span> | "A intersect B" | in both <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> |

Memory trick: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∪</mo></mrow></math></span> looks like a cup that holds everything, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∩</mo></mrow></math></span> like a cap over the overlap only.

<div class="ex">
<div class="lab">Worked example</div>

**by hand**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">}</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∪</mo><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">}</mo></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∩</mo><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span>.

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∩</mo><mi>B</mi><mo>=</mo><mi mathvariant="normal">∅</mi></mrow></math></span>, the sets are called **disjoint**: they share nothing.

</div>

```
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
print(A | B)  # {1, 2, 3, 4, 5, 6}
print(A & B)  # {3, 4}
```

~~~dg-raw
%%viz f_venn sh=u | A ∪ B, the union: everything in A, in B, or in both.
~~~

~~~dg-raw
%%viz f_venn sh=i | A ∩ B, the intersection: only the overlap.
~~~

## With arrays of ids

```
train_ids = np.array([10, 11, 12, 13, 14])
test_ids = np.array([13, 14, 15])
print(np.intersect1d(train_ids, test_ids))  # [13 14]
print(np.union1d(train_ids, test_ids))  # [10 11 12 13 14 15]
```

<div class="ex">
<div class="lab">Worked example</div>

**data leakage is a non-empty intersection**

Let <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>T</mi></mrow></math></span> be the training documents and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>E</mi></mrow></math></span> the test documents. If a document is in both, the model has seen the test answer, and the test score is too optimistic.

The check is whether <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>T</mi><mo>∩</mo><mi>E</mi><mo>=</mo><mi mathvariant="normal">∅</mi></mrow></math></span>. Here the intersection is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>13</mn><mo separator="true">,</mo><mn>14</mn><mo stretchy="false">}</mo></mrow></math></span>: two examples leaked.

</div>

## Where this breaks in practice

- **Near-duplicates.** Two copies of the same image with different file names are different ids, so an id intersection misses them. Real leakage checks also compare content (hashes or embeddings).

<div class="take">
<div class="lab">Key takeaway</div>

union is "or", intersection is "and". A non-empty intersection between train and test is leakage.

</div>

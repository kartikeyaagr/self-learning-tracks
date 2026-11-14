---
id: "01.13"
course: "ml-maths-course"
module: "01"
slug: "counting-with-inclusion-and-exclusion"
title: "Counting with Inclusion and Exclusion"
keyIdeas: ["⟦255⟧: add the sizes, subtract the overlap counted twice.", "For three sets, add the three sizes, subtract the three pairwise overlaps, add back the triple overlap.", "The same formula holds for probabilities: ⟦256⟧."]
question: "1,000 users clicked an ad, 300 bought something and 200 did both. How many did at least one? How many clicked but did not buy? If there are 5,000 users, how many did neither?"
answer: "Step 1, at least one is the union: ⟦262⟧. Step 2, clicked but not bought is ⟦263⟧. Step 3, neither is the complement of the union: ⟦264⟧."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Set operations"]
tools: ["Python"]
topic: {"id": "01.02", "title": "Set operations"}
---

## The big picture

When you add the sizes of two groups that overlap, people in both groups get counted twice. Inclusion and exclusion is the bookkeeping that fixes it. It is used in dashboards, in probability, and in counting arguments.

## Two sets

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∣</mi><mi>A</mi><mo>∪</mo><mi>B</mi><mi mathvariant="normal">∣</mi><mo>=</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mi mathvariant="normal">∣</mi><mo>+</mo><mi mathvariant="normal">∣</mi><mi>B</mi><mi mathvariant="normal">∣</mi><mo>−</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mo>∩</mo><mi>B</mi><mi mathvariant="normal">∣</mi></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**checking with small sets**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">}</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∩</mo><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo stretchy="false">}</mo></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>4</mn><mo>+</mo><mn>4</mn><mo>−</mo><mn>2</mn><mo>=</mo><mn>6</mn></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∪</mo><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">}</mo></mrow></math></span> has 6 elements.

</div>

```
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
print(len(A) + len(B) - len(A & B), len(A | B))  # 6 6
```

~~~dg-raw
%%viz f_venn sh=none a=Python b=SQL n=12,5,8,3 | Counts in each region. |Python ∪ SQL| = 17 + 13 − 5 = 25, because the 5 in the overlap would otherwise be counted twice.
~~~

## Three sets

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∣</mi><mi>A</mi><mo>∪</mo><mi>B</mi><mo>∪</mo><mi>C</mi><mi mathvariant="normal">∣</mi><mo>=</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mi mathvariant="normal">∣</mi><mo>+</mo><mi mathvariant="normal">∣</mi><mi>B</mi><mi mathvariant="normal">∣</mi><mo>+</mo><mi mathvariant="normal">∣</mi><mi>C</mi><mi mathvariant="normal">∣</mi><mo>−</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mo>∩</mo><mi>B</mi><mi mathvariant="normal">∣</mi><mo>−</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mo>∩</mo><mi>C</mi><mi mathvariant="normal">∣</mi><mo>−</mo><mi mathvariant="normal">∣</mi><mi>B</mi><mo>∩</mo><mi>C</mi><mi mathvariant="normal">∣</mi><mo>+</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mo>∩</mo><mi>B</mi><mo>∩</mo><mi>C</mi><mi mathvariant="normal">∣</mi></mrow></math></span></span>
Add the singles, subtract the pairs, add back the triple (which the pair step removed once too often).

<div class="ex">
<div class="lab">Worked example</div>

**three filters on a dataset**

50 images are blurry, 30 are mislabelled, 20 are duplicates. 10 are blurry and mislabelled, 5 blurry and duplicate, 4 mislabelled and duplicate, 2 all three.

Images with at least one problem: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>50</mn><mo>+</mo><mn>30</mn><mo>+</mo><mn>20</mn><mo>−</mo><mn>10</mn><mo>−</mo><mn>5</mn><mo>−</mo><mn>4</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>83</mn></mrow></math></span>.

</div>

## Probabilities

Divide every count by the total and the same rule holds for probabilities:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∪</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>+</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo><mo>−</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span></span>
You will use it in module 10.

## Where this breaks in practice

- **Adding percentages of overlapping groups.** "60% use the app on phone, 50% on laptop" does not mean 110% use it. Without the overlap you cannot get the union.

<div class="take">
<div class="lab">Key takeaway</div>

to count a union, add the sizes and subtract what you counted twice.

</div>

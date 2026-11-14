---
id: "01.10"
course: "ml-maths-course"
module: "01"
slug: "difference-complement-and-de-morgan-s-laws"
title: "Difference, Complement and De Morgan's Laws"
keyIdeas: ["Difference ⟦203⟧: in ⟦8⟧ but not in ⟦41⟧. Complement ⟦205⟧: in the universe ⟦207⟧ but not in ⟦8⟧.", "De Morgan: ⟦206⟧ and ⟦208⟧.", "In NumPy, `~(a | b)` equals `~a & ~b`."]
question: "With ⟦202⟧, ⟦191⟧, ⟦192⟧, compute ⟦203⟧, ⟦204⟧, ⟦205⟧, and check ⟦206⟧."
answer: "Step 1, ⟦203⟧ keeps elements of ⟦8⟧ not in ⟦41⟧: ⟦60⟧. Step 2, ⟦211⟧, so difference depends on order. Step 3, ⟦212⟧. Step 4, ⟦213⟧, so ⟦214⟧. Step 5, ⟦215⟧, so ⟦216⟧. The two sides match."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Set operations"]
tools: ["Python", "NumPy"]
topic: {"id": "01.02", "title": "Set operations"}
---

## The big picture

"Users who clicked but did not buy." "Everything that is not spam." These need two more operations: difference and complement. De Morgan's laws then tell you how "not" interacts with "and" and "or", which is the most common source of mask bugs.

## Difference and complement

Assume every set lives inside a bigger set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span>, the **universe** (say, all users).

| Operation | Symbol | Read aloud | Elements |
| --- | --- | --- | --- |
| Difference | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>∖</mo><mi>B</mi></mrow></math></span> | "A minus B" | in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> but not in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> |
| Complement | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi>c</mi></msup></mrow></math></span> | "A complement" | in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> but not in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> |

The complement is a difference from the universe: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>A</mi><mi>c</mi></msup><mo>=</mo><mi>U</mi><mo>∖</mo><mi>A</mi></mrow></math></span>.

```
U = set(range(1, 9))
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
print(sorted(A - B), sorted(B - A))  # [1, 2] [5, 6]
print(sorted(U - A))  # [5, 6, 7, 8]
```

~~~dg-raw
%%viz f_venn sh=d | A ∖ B: the part of A that is outside B.
~~~

## De Morgan's laws

Taking a complement flips "or" into "and" and back:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">(</mo><mi>A</mi><mo>∪</mo><mi>B</mi><msup><mo stretchy="false">)</mo><mi>c</mi></msup><mo>=</mo><msup><mi>A</mi><mi>c</mi></msup><mo>∩</mo><msup><mi>B</mi><mi>c</mi></msup><mo separator="true">,</mo><mspace width="2em"/><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><msup><mo stretchy="false">)</mo><mi>c</mi></msup><mo>=</mo><msup><mi>A</mi><mi>c</mi></msup><mo>∪</mo><msup><mi>B</mi><mi>c</mi></msup></mrow></math></span></span>
In words: "not (A or B)" is "not A and not B". "Not (A and B)" is "not A or not B".

<div class="ex">
<div class="lab">Worked example</div>

**in plain words**

"Users who neither clicked nor bought" = "users who did not click" and "users who did not buy".

"Users who did not do both" = "did not click" or "did not buy". This second group is much larger.

</div>

In NumPy, `~` is "not":

```
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])
print(np.array_equal(~(a | b), ~a & ~b))  # True
print(np.array_equal(~(a & b), ~a | ~b))  # True
```

~~~dg-raw
%%viz f_venn sh=n | (A ∪ B)ᶜ: the shaded part is outside both circles. That is exactly Aᶜ ∩ Bᶜ, which is De Morgan's first law.
~~~

## Where this breaks in practice

- **Negating a filter.** To drop rows that are "missing or invalid", the keep-mask is `~(missing | invalid)`, which is `~missing & ~invalid`. Writing `~missing | ~invalid` keeps almost everything.
- **The universe matters.** "Not spam" depends on what <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>U</mi></mrow></math></span> is. Complements are only defined relative to a universe.

<div class="take">
<div class="lab">Key takeaway</div>

difference is "but not", complement is "not". De Morgan swaps "and" and "or" whenever you push a "not" inside brackets.

</div>

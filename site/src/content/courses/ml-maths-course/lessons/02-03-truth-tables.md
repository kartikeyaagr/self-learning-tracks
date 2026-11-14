---
id: "02.03"
course: "ml-maths-course"
module: "02"
slug: "truth-tables"
title: "Truth Tables"
keyIdeas: ["A truth table lists every combination of input truth values, so a formula's meaning can be checked row by row.", "With ⟦233⟧ statements there are ⟦461⟧ rows.", "Two formulas are equivalent (⟦462⟧) when their columns agree on every row."]
question: "Build the truth table of ⟦455⟧ and of ⟦460⟧. Are they equivalent?"
answer: "Step 1, list the four rows ⟦465⟧: TT, TF, FT, FF. Step 2, ⟦448⟧ is T, F, F, F, so ⟦455⟧ is F, T, T, T. Step 3, ⟦449⟧ is F, F, T, T and ⟦454⟧ is F, T, F, T. Their OR is F, T, T, T. Step 4, the two columns match on every row, so ⟦466⟧. This is one of De Morgan's laws."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

Words like "and" and "or" are slippery in English. A truth table removes all doubt: it lists every possible input and the output for each. If two formulas give the same column, they mean the same thing, and no argument is needed.

## Building a table

List every combination of inputs. With two statements there are 4 rows, with three there are 8, and in general <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mn>2</mn><mi>n</mi></msup></mrow></math></span>.

| p | q | not p | p and q | p or q |
| --- | --- | --- | --- | --- |
| T | T | F | T | T |
| T | F | F | F | T |
| F | T | T | F | T |
| F | F | T | F | F |

Read one row at a time. Row 2 says: if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is true and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> is false, then "p and q" is false and "p or q" is true.

## Generating tables in code

`itertools.product` lists every row for you, however many inputs there are:

```
from itertools import product
for p, q in product([True, False], repeat=2):
    print(p, q, p and q)
```

Counting rows for three inputs:

```
from itertools import product
print(len(list(product([True, False], repeat=3))))  # 8
```

## Equivalence

Two formulas are **equivalent**, written <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≡</mo></mrow></math></span>, if they have the same column. You can test this with a one-line check:

```
from itertools import product
rows = list(product([True, False], repeat=2))
left = [not (p and q) for p, q in rows]
right = [(not p) or (not q) for p, q in rows]
print(left == right)  # True
```

<div class="ex">
<div class="lab">Worked example</div>

**a formula that is always true**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∨</mo><mi mathvariant="normal">¬</mi><mi>p</mi></mrow></math></span> is true on both rows (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> = T gives T, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> = F gives T). A formula true on every row is called a **tautology**.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∧</mo><mi mathvariant="normal">¬</mi><mi>p</mi></mrow></math></span> is false on both rows. A formula false on every row is a **contradiction**.

</div>

## Where this breaks in practice

- **Too many rows.** With 30 boolean inputs there are over a billion rows. Tables are a tool for small formulas; for big ones use algebra (lesson 02.18).

<div class="take">
<div class="lab">Key takeaway</div>

list all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mn>2</mn><mi>n</mi></msup></mrow></math></span> rows, fill each column, and compare columns to decide equivalence.

</div>

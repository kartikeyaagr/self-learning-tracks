---
id: "02.04"
course: "ml-maths-course"
module: "02"
slug: "inclusive-and-exclusive-or"
title: "Inclusive and Exclusive Or"
keyIdeas: ["Mathematical \"or\" is inclusive: ⟦447⟧ is true when both are true.", "Exclusive or (XOR, ⟦467⟧) is true when exactly one is true.", "On booleans, XOR is the same as \"not equal\": ⟦468⟧."]
question: "Two annotators label an image as \"cat\" (true) or \"not cat\". You want to flag images where they disagree. Which connective do you use, and what does it give for the label pairs (T, T), (T, F), (F, F)?"
answer: "Step 1, \"disagree\" means exactly one said cat. That is XOR. Step 2, (T, T): both agree, ⟦471⟧, not flagged. Step 3, (T, F): they disagree, ⟦472⟧, flagged. Step 4, (F, F): both agree, ⟦473⟧, not flagged. Step 5, inclusive OR would wrongly flag (T, T), because ⟦474⟧."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python", "NumPy"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

In everyday English "tea or coffee?" usually means one but not both. In maths, "or" means at least one, and both is allowed. Mixing the two up gives wrong filters and wrong probability calculations, so it pays to have a separate name for each.

## Two kinds of or

- **Inclusive or**, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>∨</mo><mi>q</mi></mrow></math></span>: true when at least one is true. This is what "or" means in maths and in Python.
- **Exclusive or**, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⊕</mo><mi>q</mi></mrow></math></span> (XOR): true when exactly one is true.

| p | q | p or q | p xor q |
| --- | --- | --- | --- |
| T | T | T | F |
| T | F | T | T |
| F | T | T | T |
| F | F | F | F |

The only row where they differ is the first.

~~~dg-raw
%%viz f_venn sh=u | Inclusive or: everything in A, in B, or in both.
~~~

~~~dg-raw
%%viz f_venn sh=x | Exclusive or: in exactly one of A and B. The overlap is left out.
~~~

## In code

Python has no `xor` keyword, but on booleans "exactly one is true" is the same as "they differ":

```
print([p != q for p in (True, False) for q in (True, False)])  # [False, True, True, False]
```

NumPy has it for arrays:

```
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])
print(np.logical_xor(a, b))  # [False  True  True False]
```

<div class="ex">
<div class="lab">Worked example</div>

**counting with inclusive or**

In a class of 30, 12 take Python, 9 take SQL, and 4 take both.

Inclusive or ("Python or SQL"): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>12</mn><mo>+</mo><mn>9</mn><mo>−</mo><mn>4</mn><mo>=</mo><mn>17</mn></mrow></math></span> students.

Exclusive or ("exactly one of them"): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>17</mn><mo>−</mo><mn>4</mn><mo>=</mo><mn>13</mn></mrow></math></span> students.

</div>

## Where this breaks in practice

- **Survey and spec wording.** "Users who clicked or bought" in a metric spec: ask whether buyers who also clicked count once, twice or not at all.
- **`^` on non-booleans.** In Python and NumPy `^` is bitwise XOR. On booleans it gives logical XOR, but `3 ^ 5` is `6`, not True.

<div class="take">
<div class="lab">Key takeaway</div>

maths "or" includes both. When you mean "exactly one", say XOR.

</div>

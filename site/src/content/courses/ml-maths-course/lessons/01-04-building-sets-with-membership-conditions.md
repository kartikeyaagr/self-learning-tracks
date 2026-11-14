---
id: "01.04"
course: "ml-maths-course"
module: "01"
slug: "building-sets-with-membership-conditions"
title: "Building Sets with Membership Conditions"
keyIdeas: ["Big or infinite sets are described by a test: keep the elements that pass.", "In NumPy the test is a boolean mask, and indexing with it keeps the elements that pass.", "Combine tests with `&` (and), `|` (or) and `~` (not), with brackets around each test."]
question: "Ages are ⟦71⟧. Describe the working-age set (18 or over, under 65) by a membership condition, list it, and write the NumPy line."
answer: "Step 1, the condition has two parts joined by \"and\": ⟦79⟧ and ⟦80⟧. Step 2, test each age: 15 fails the first part, 22, 37 and 64 pass both, 70 fails the second. Step 3, the set is ⟦81⟧. Step 4, in NumPy: `ages[(ages >= 18) & (ages < 65)]`, with brackets around each comparison because `&` binds tighter than `>=`."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["NumPy"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

You cannot list every real number between 0 and 1, or every image of a cat. Instead you describe a set by a **membership condition**: a test that each candidate either passes or fails. The set is everything that passes. This is the idea behind every filter you will ever write.

## Where to look, then a test

A membership condition always has two parts:

1. **Where to look**: the bigger set the candidates come from.
2. **The test**: a yes/no question asked of each candidate.

For example, "real numbers from −2 up to but not including 3" looks in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> and tests <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>2</mn><mo>≤</mo><mi>x</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&lt;</mo><mn>3</mn></mrow></math></span>. The next lesson writes this as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><mi mathvariant="normal">ℝ</mi><mo>:</mo><mo>−</mo><mn>2</mn><mo>≤</mo><mi>x</mi><mo>&lt;</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span>, read "all real <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> such that <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is at least <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>2</mn></mrow></math></span> and less than <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn></mrow></math></span>".

<div class="ex">
<div class="lab">Worked example</div>

**describing sets by a test**

"Even numbers from 0 to 9": look in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>9</mn><mo stretchy="false">}</mo></mrow></math></span>, test "divisible by 2". Result <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>6</mn><mo separator="true">,</mo><mn>8</mn><mo stretchy="false">}</mo></mrow></math></span>.

"Numbers from 1 to 20 divisible by 6": result <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>6</mn><mo separator="true">,</mo><mn>12</mn><mo separator="true">,</mo><mn>18</mn><mo stretchy="false">}</mo></mrow></math></span>.

"Scores above 0.5": look in the list of scores, test "greater than 0.5".

</div>

~~~dg-raw
%%viz f_line x=-4,5 iv=-2,3,co,the_set | The set of real x with −2 ≤ x < 3. A filled dot means the endpoint is in the set, a hollow dot means it is not.
~~~

## In code: a boolean mask

NumPy writes the test as an array of True/False values, one per element, called a **mask**. Indexing with the mask keeps the elements that pass.

```
scores = np.array([0.2, 0.9, 0.55, 0.1, 0.7])
mask = scores > 0.5
print(mask)  # [False  True  True False  True]
print(scores[mask])  # [0.9  0.55 0.7 ]
```

Often you want *which positions* pass instead of the values:

```
pred = np.array([1, 0, 2, 2, 1])
y = np.array([1, 2, 2, 0, 1])
print(np.flatnonzero(pred != y))  # [1 3]
```

That is the set of misclassified examples: positions 1 and 3.

## Combining tests

Join conditions with `&` (and), `|` (or) and `~` (not). Put brackets around each comparison.

```
ages = np.array([15, 22, 37, 64, 70])
print(ages[(ages >= 18) & (ages < 65)])  # [22 37 64]
print(ages[(ages < 18) | (ages >= 65)])  # [15 70]
```

The second line is exactly the elements the first one left out.

## Where this breaks in practice

- **Missing brackets.** `ages >= 18 & ages < 65` is parsed as `ages >= (18 & ages) < 65` and either errors or gives nonsense.
- **Python `and` on arrays.** `(a > 0) and (b > 0)` raises "truth value of an array is ambiguous". Use `&`.

<div class="take">
<div class="lab">Key takeaway</div>

a membership condition is "where to look" plus "a test". In NumPy the test is a mask, and the set is whatever the mask keeps.

</div>

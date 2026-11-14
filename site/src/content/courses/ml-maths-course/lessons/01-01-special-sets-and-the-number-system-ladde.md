---
id: "01.01"
course: "ml-maths-course"
module: "01"
slug: "special-sets-and-the-number-system-ladder"
title: "Special Sets and the Number-System Ladder"
keyIdeas: ["A set is a collection of distinct things with no order. ⟦0⟧ is read \"x is in A\".", "The number ladder ⟦1⟧: counting numbers, integers, fractions, reals.", "⟦2⟧ means lists of ⟦3⟧ real numbers. Intervals like ⟦4⟧ and ⟦5⟧ are pieces of the real line."]
question: "A spam filter has a label (spam or not), a word count, a predicted spam probability from a sigmoid, and 5,000 weights. Name the set each one lives in."
answer: "Step 1, the label has two possible values, so it lives in ⟦37⟧. Step 2, a word count is a whole number that cannot be negative, so it lives in ⟦11⟧. Step 3, a probability lies between 0 and 1. A sigmoid never outputs exactly 0 or 1 in exact maths, so the output lives in the open interval ⟦5⟧, which sits inside ⟦4⟧. Step 4, each weight is a real number and there are 5,000 of them, so the weight vector lives in ⟦38⟧."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["Python"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

Before a maths paper or an ML library does anything, it tells you **where each thing lives**. "The input is 784 pixel values." "The label is one of 10 digits." "The weights are real numbers." Sets are the language for saying this precisely, and the set decides what you can do. You can nudge a weight from 0.31 to 0.32, so you can take a gradient with respect to it. You cannot nudge a label from "cat" to "slightly more cat", so there is no gradient with respect to a label.

## What a set is

A **set** is a collection of distinct objects, called its **elements**. We write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><mi>A</mi></mrow></math></span>, read "x is in A", when <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is one of the elements of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo mathvariant="normal">∉</mo><mi>A</mi></mrow></math></span> when it is not.

## The number-system ladder

A few sets of numbers have standard names. Each one sits inside the next:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">ℕ</mi><mo>⊂</mo><mi mathvariant="normal">ℤ</mi><mo>⊂</mo><mi mathvariant="normal">ℚ</mi><mo>⊂</mo><mi mathvariant="normal">ℝ</mi></mrow></math></span></span>
The symbol <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>⊂</mo></mrow></math></span> is read "is contained in".

| Symbol | Name | What is in it | ML example |
| --- | --- | --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℕ</mi></mrow></math></span> | natural numbers | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span> (some books start at 1) | class labels, counts, indices |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℤ</mi></mrow></math></span> | integers | adds the negatives: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>…</mo><mo separator="true">,</mo><mo>−</mo><mn>2</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mo>…</mo></mrow></math></span> | offsets, differences of counts |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℚ</mi></mrow></math></span> | rationals | fractions <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mi mathvariant="normal">/</mi><mi>q</mi></mrow></math></span> like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn><mi mathvariant="normal">/</mi><mn>4</mn></mrow></math></span> or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>7</mn><mi mathvariant="normal">/</mi><mn>2</mn></mrow></math></span> | any finite decimal, like 0.125 |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> | reals | everything on the number line, including <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>2</mn></msqrt></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>π</mi></mrow></math></span> | weights, losses, probabilities |

The letters come from words: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℕ</mi></mrow></math></span> for natural, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℤ</mi></mrow></math></span> from German *Zahlen* (numbers), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℚ</mi></mrow></math></span> for quotient, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> for real.

<div class="ex">
<div class="lab">Worked example</div>

**climbing the ladder**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>3</mn></mrow></math></span> is in all four sets. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>3</mn></mrow></math></span> is in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℤ</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℚ</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> but not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℕ</mi></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.75</mn><mo>=</mo><mn>3</mn><mi mathvariant="normal">/</mi><mn>4</mn></mrow></math></span> is in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℚ</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msqrt><mn>2</mn></msqrt><mo>≈</mo><mn>1.41421</mn></mrow></math></span> is only in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span>: no fraction squares to exactly 2.

</div>

## Powers of a set

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>3</mn></msup></mrow></math></span> means "lists of 3 real numbers", like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0.5</mn><mo separator="true">,</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> means lists of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> real numbers. A colour pixel is a point in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>255</mn><msup><mo stretchy="false">}</mo><mn>3</mn></msup></mrow></math></span>: three whole numbers (red, green, blue) from 0 to 255.

<div class="ex">
<div class="lab">Worked example</div>

**reading a paper's first line**

"Let <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>∈</mo><msup><mi mathvariant="normal">ℝ</mi><mn>784</mn></msup></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi><mo>∈</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><mn>9</mn><mo stretchy="false">}</mo></mrow></math></span>." Read aloud: "x is a list of 784 real numbers, and y is one of the ten digits 0 to 9." That is MNIST: a 28 by 28 image flattened into 784 numbers, plus a digit label.

</div>

## Intervals

Intervals are pieces of the real line:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span> includes both ends: every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>≤</mo><mi>x</mi><mo>≤</mo><mn>1</mn></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> excludes both ends: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>&lt;</mo><mi>x</mi><mo>&lt;</mo><mn>1</mn></mrow></math></span>.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span> includes 0 but not 1.

Probabilities live in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></math></span>. A sigmoid output <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup><mo stretchy="false">)</mo></mrow></math></span> lives in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

## Floats are not real numbers

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> has infinitely many numbers between 0 and 1. A computer stores a number as a **float** with a fixed number of bits, so there are only finitely many floats, and results get rounded.

```
print(0.1 + 0.2 == 0.3)  # False
x = np.float32(30.0)
print(1 / (1 + np.exp(-x)) == 1.0)  # True
```

In exact maths <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo stretchy="false">(</mo><mn>30</mn><mo stretchy="false">)</mo></mrow></math></span> is slightly less than 1, but in float32 it rounds to exactly 1.0. Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>σ</mi><mo stretchy="false">(</mo><mn>30</mn><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>=</mo><mi>log</mi><mo>⁡</mo><mn>0</mn><mo>=</mo><mo>−</mo><mi mathvariant="normal">∞</mi></mrow></math></span>. This is why libraries offer a combined, safe `BCEWithLogitsLoss` instead of "sigmoid, then log".

<div class="take">
<div class="lab">Key takeaway</div>

the set an object lives in decides which operations make sense. Floats only approximate <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span>, and the gap shows up as rounding bugs.

</div>

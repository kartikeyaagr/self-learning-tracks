---
id: "02.01"
course: "ml-maths-course"
module: "02"
slug: "statements-and-predicates"
title: "Statements and Predicates"
keyIdeas: ["A statement is a sentence that is definitely true or definitely false, like \"7 is prime\".", "A predicate ⟦93⟧ is a statement with a blank, like \"x is prime\". It gets a truth value only when you fill in ⟦7⟧.", "In code a predicate is a function that returns a boolean, and a statement is one call of it."]
question: "Which of these are statements, which are predicates, and which are neither? (a) \"The validation loss at epoch 10 was 0.31.\" (b) \"The loss at epoch ⟦166⟧ is below 0.3.\" (c) \"Lower the learning rate.\" For each predicate, fill the blank once to make a true or false statement."
answer: "Step 1, (a) refers to one fixed epoch and one number, so it is either true or false: a statement. Step 2, (b) has a blank, the epoch ⟦166⟧. On its own it has no truth value, so it is a predicate ⟦445⟧. Filling it in, ⟦446⟧ is \"the loss at epoch 10 is below 0.3\", which is false if the loss was 0.31. Step 3, (c) is an instruction. It is neither true nor false, so it is neither."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

Maths papers, proofs and even code reviews are built from sentences that are either true or false. Before you can combine them, negate them or prove them, you need to know which sentences have a truth value at all. That is the whole job of this lesson.

## Statements

A **statement** (also called a proposition) is a sentence that is definitely true or definitely false.

- "7 is prime" is a true statement.
- "2 > 3" is a false statement.
- "Close the door" is not a statement. It is neither true nor false.
- "This sentence is short" depends on nothing outside itself, so it is a statement (a true one).

We usually name statements with small letters: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>r</mi></mrow></math></span>.

## Predicates

A **predicate** is a statement with a blank. Write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> for "x is prime". On its own it has no truth value, because we don't know what <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is. Fill the blank and you get a statement: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mn>7</mn><mo stretchy="false">)</mo></mrow></math></span> is true, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mn>8</mn><mo stretchy="false">)</mo></mrow></math></span> is false.

A predicate can have more than one blank. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>: "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&lt;</mo><mi>y</mi></mrow></math></span>". Then <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo></mrow></math></span> is true and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo stretchy="false">(</mo><mn>5</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></math></span> is false.

In code a predicate is a function that returns a boolean, and a statement is one call of it:

```
def is_prime(n):
    return n > 1 and all(n % k != 0 for k in range(2, int(n ** 0.5) + 1))

print(is_prime(7))  # True
print(is_prime(8))  # False
```

<div class="ex">
<div class="lab">Worked example</div>

**spotting predicates in ML**

"The loss is below 0.1" is a predicate in the variable "which training step". At step 500 it might be false, at step 5,000 true.

"Token <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> is padding" is a predicate in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span>. A padding mask is this predicate evaluated on every token at once.

"The model has 7 billion parameters" is a statement about one fixed model.

</div>

## Predicates evaluated on many inputs

A predicate applied to every element of a list gives a list of truth values. That list is exactly a boolean mask.

```
tokens = ["the", "cat", "<pad>", "<pad>"]
print([t == "<pad>" for t in tokens])  # [False, False, True, True]
```

## Where this breaks in practice

- **Vague words.** "The model is good" is not a statement until "good" is pinned down, for example "accuracy above 90% on the test set".
- **Hidden blanks.** "Loss is decreasing" hides the blank "over which steps?". Write the blank out before arguing about whether it is true.

<div class="take">
<div class="lab">Key takeaway</div>

a statement has a truth value. A predicate has a blank, and becomes a statement only when the blank is filled.

</div>

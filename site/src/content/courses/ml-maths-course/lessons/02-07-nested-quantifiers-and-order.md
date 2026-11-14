---
id: "02.07"
course: "ml-maths-course"
module: "02"
slug: "nested-quantifiers-and-order"
title: "Nested Quantifiers and Order"
keyIdeas: ["With two quantifiers, each later variable may depend on the earlier ones.", "⟦508⟧ lets ⟦509⟧ depend on ⟦7⟧. ⟦510⟧ needs one ⟦509⟧ that works for every ⟦7⟧, which is a much stronger claim.", "Quantifiers of the same kind can be swapped freely: ⟦511⟧."]
question: "For integers, decide: (a) ⟦506⟧; (b) ⟦507⟧."
answer: "Step 1, (a): given any integer ⟦7⟧, choose ⟦514⟧. Then ⟦515⟧. The choice of ⟦509⟧ depends on ⟦7⟧, which is allowed. True. Step 2, (b): we need one fixed integer ⟦509⟧ bigger than every integer. Take ⟦516⟧: then ⟦515⟧ fails. So no such ⟦509⟧ exists. False. Step 3, the only change was the order, and it changed the answer."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

"Every person owns a phone" and "there is a phone that every person owns" use the same words in a different order, and say very different things. Mathematical claims stack quantifiers all the time, so reading them in order is essential.

## Order is meaning

When there are two quantifiers, each later variable is allowed to depend on the earlier ones.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> person <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> phone: the person owns the phone. Each person can have a different phone.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> phone <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> person: the person owns the phone. One single phone is owned by everybody.

~~~dg-raw
%%viz f_map A=Ana,Ben,Cy B=phone_1,phone_2,phone_3 f=Ana:phone_1,Ben:phone_2,Cy:phone_3 an=person bn=phone | ∀ person ∃ phone: each person points to some phone, and the phones can differ.
~~~

~~~dg-raw
%%viz f_map A=Ana,Ben,Cy B=phone_1,phone_2,phone_3 f=Ana:phone_1,Ben:phone_1,Cy:phone_1 an=person bn=phone | ∃ phone ∀ person: one phone that every arrow reaches. A far stronger claim.
~~~

## A game reading

Think of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> as a move by an opponent and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> as your move. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>x</mi><mtext> </mtext><mi mathvariant="normal">∃</mi><mi>y</mi></mrow></math></span>, the opponent picks <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> first, then you pick <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> knowing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>. In <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi><mi>y</mi><mtext> </mtext><mi mathvariant="normal">∀</mi><mi>x</mi></mrow></math></span>, you must commit to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span> first, before seeing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>. Going second is always easier.

## In code

Nested `all` and `any` follow the same order as the symbols:

```
X = range(5)
Y = range(6)
print(all(any(y > x for y in Y) for x in X))  # True
print(any(all(y > x for x in X) for y in Y))  # True
```

Here both are true, because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> has an element (5) bigger than all of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span>. Shrink <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> to equal <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> and the second fails:

```
X = range(5)
print(any(all(y > x for x in X) for y in X))  # False
```

<div class="ex">
<div class="lab">Worked example</div>

**a model claim**

"For every input there is a model that classifies it correctly" is trivial: memorise each input.

"There is a model that classifies every input correctly" is the claim you actually want from training.

</div>

## Where this breaks in practice

- **Restating results loosely.** "For each dataset we tuned hyperparameters" (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> dataset <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> setting) is often reported as if one setting worked everywhere (<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> setting <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> dataset).

<div class="take">
<div class="lab">Key takeaway</div>

read quantifiers left to right. A later variable may depend on the earlier ones, so swapping <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi></mrow></math></span> changes the claim.

</div>

---
id: "02.14"
course: "ml-maths-course"
module: "02"
slug: "necessary-and-sufficient-conditions"
title: "Necessary and Sufficient Conditions"
keyIdeas: ["\"⟦102⟧ is sufficient for ⟦404⟧\" means ⟦554⟧: having ⟦102⟧ guarantees ⟦404⟧.", "\"⟦102⟧ is necessary for ⟦404⟧\" means ⟦563⟧: you cannot have ⟦404⟧ without ⟦102⟧.", "A condition can be sufficient without being necessary, and the other way round."]
question: "Is \"⟦569⟧ at a stationary point ⟦570⟧\" necessary, sufficient, both or neither for ⟦570⟧ to be a strict local minimum?"
answer: "Step 1, sufficient: if ⟦576⟧ and ⟦569⟧, the curve bends upward at ⟦570⟧, so it is a strict local minimum. Yes. Step 2, necessary: is every strict local minimum one with ⟦577⟧? No. ⟦578⟧ has a strict minimum at 0, but ⟦579⟧. Step 3, so it is sufficient but not necessary."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Conditional logic"]
tools: ["Python"]
topic: {"id": "02.02", "title": "Conditional logic"}
---

## The big picture

"Normalisation is necessary for stable training." "A large batch is sufficient for good gradients." These words are implications in disguise, and each points in a specific direction. Mixing them up reverses the claim.

## The definitions

- "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is **sufficient** for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>" means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>⇒</mo><mi>q</mi></mrow></math></span>. Having <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is enough to guarantee <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>.
- "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> is **necessary** for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span>" means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi><mo>⇒</mo><mi>p</mi></mrow></math></span>. Whenever <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> holds, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> holds too.

In set language, if <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> is the set where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> holds and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi></mrow></math></span> where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>q</mi></mrow></math></span> holds, "sufficient" means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo>⊆</mo><mi>Q</mi></mrow></math></span> and "necessary" means <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Q</mi><mo>⊆</mo><mi>P</mi></mrow></math></span>.

~~~dg-raw
%%viz f_line x=0,9 iv=5,inf,oo,x_>_5_(sufficient);3,inf,oo,x_>_3 | Every x above 5 is also above 3, so x > 5 is sufficient for x > 3. It is not necessary: x = 4 is above 3 without being above 5.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**necessary vs sufficient with numbers**

"<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>5</mn></mrow></math></span> is sufficient for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>3</mn></mrow></math></span>": true.

"<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>5</mn></mrow></math></span> is necessary for <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>&gt;</mo><mn>3</mn></mrow></math></span>": false, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>4</mn></mrow></math></span> breaks it.

"Divisible by 4 is sufficient for even": true. Necessary? No, 6 is even but not divisible by 4.

</div>

## Checking in code

Sufficiency is a "for all" over the set where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi></mrow></math></span> holds:

```
xs = range(-10, 11)
print(all(x > 3 for x in xs if x > 5))  # True
print(all(x > 5 for x in xs if x > 3))  # False
```

## Where this breaks in practice

- **"Necessary" claims in papers.** "Attention is necessary for this task" says nothing without attention can do it. Usually what was shown is that attention was sufficient in the authors' setup.

<div class="take">
<div class="lab">Key takeaway</div>

sufficient means "enough to guarantee", necessary means "can't do without". They point in opposite directions.

</div>

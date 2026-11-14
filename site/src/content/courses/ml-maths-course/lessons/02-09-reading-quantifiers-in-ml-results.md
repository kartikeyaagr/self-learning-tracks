---
id: "02.09"
course: "ml-maths-course"
module: "02"
slug: "reading-quantifiers-in-ml-results"
title: "Reading Quantifiers in ML Results"
keyIdeas: ["Before trusting a result, find each quantifier and what it ranges over.", "\"With probability at least ⟦546⟧, for all ⟦547⟧\" covers the model chosen after seeing the data. \"For all ⟦547⟧, with probability\" does not.", "Hidden quantifiers (\"robust\", \"always\", \"generalises\") need a stated set and size before they mean anything."]
question: "A paper says \"our method beats the baseline\". List the hidden quantifiers you would ask about."
answer: "Step 1, over which datasets: for all of the tested ones, or some? Step 2, over which seeds: is it the mean over seeds, or for every seed? Step 3, over which budgets and sizes: for every compute budget, or only the one reported? Step 4, over which metrics. Each answer turns the vague claim into a precise statement that could be checked or broken."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

The previous lessons gave you the grammar. This lesson applies it to the kind of sentence you will actually read: a theorem or claim in an ML paper. The skill is to find each quantifier, see what it ranges over, and notice its order.

## A generalisation bound

A typical learning-theory result reads: "With probability at least <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>−</mo><mi>δ</mi></mrow></math></span>, for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi><mo>∈</mo><mi mathvariant="normal">ℋ</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mo stretchy="false">(</mo><mi>h</mi><mo stretchy="false">)</mo><mo>≤</mo><mover accent="true"><mi>R</mi><mo>^</mo></mover><mo stretchy="false">(</mo><mi>h</mi><mo stretchy="false">)</mo><mo>+</mo><mi>ϵ</mi><mo stretchy="false">(</mo><mi>n</mi><mo separator="true">,</mo><mi>δ</mi><mo stretchy="false">)</mo></mrow></math></span>."

Translate the symbols:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span> is a model (a "hypothesis") and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℋ</mi></mrow></math></span> is the set of models you choose from.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>R</mi><mo stretchy="false">(</mo><mi>h</mi><mo stretchy="false">)</mo></mrow></math></span> is the true error on new data and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>R</mi><mo>^</mo></mover><mo stretchy="false">(</mo><mi>h</mi><mo stretchy="false">)</mo></mrow></math></span> is the error on your training set.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> is the number of training examples and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>δ</mi></mrow></math></span> is a small failure probability.

Now the order. "With probability" comes first, then "for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>". So one draw of the training data covers every model at once. That matters, because your training algorithm picks <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span> after seeing the data. If the bound were "for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>, with probability...", it would only cover a model fixed before you looked at the data.

<div class="ex">
<div class="lab">Worked example</div>

**why the order matters, with numbers**

Suppose each single model has a 1% chance of a misleadingly low training error.

With 1,000 models, a bound of the "for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>, with probability 99%" kind says nothing about the one you pick, because among 1,000 models about 10 are expected to look misleadingly good.

A "with probability 99%, for all <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>h</mi></mrow></math></span>" bound must pay for all 1,000 at once, which is why such bounds grow with the size of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℋ</mi></mrow></math></span>.

</div>

## Checking a claim on seeds

"Our method beats the baseline" might mean the mean over seeds, or every seed:

```
ours = [0.81, 0.79, 0.84]
base = [0.80, 0.80, 0.80]
print(sum(ours) / 3 > sum(base) / 3, all(o > b for o, b in zip(ours, base)))  # True False
```

The average is higher, but it is not true that every seed wins.

## Where this breaks in practice

- **Hidden quantifiers.** "The model is robust to noise" hides "for all noise of size at most what?". Ask for the set and the size.
- **Vacuous truth.** `all([])` is `True`. A claim about every example in an empty split is true and worthless.

<div class="take">
<div class="lab">Key takeaway</div>

find each quantifier, ask what it ranges over, and check the order. That tells you what a result really promises.

</div>

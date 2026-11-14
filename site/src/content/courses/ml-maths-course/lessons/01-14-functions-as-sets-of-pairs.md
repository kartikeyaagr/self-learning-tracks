---
id: "01.14"
course: "ml-maths-course"
module: "01"
slug: "functions-as-sets-of-pairs"
title: "Functions as Sets of Pairs"
keyIdeas: ["A function ⟦268⟧ gives exactly one output in ⟦41⟧ for each input in ⟦8⟧.", "Its graph ⟦269⟧ is a subset of ⟦220⟧ with each input appearing exactly once.", "An embedding table is a function on a finite set, stored as its graph."]
question: "Which of these sets of pairs are functions from ⟦68⟧ to ⟦19⟧: (a) ⟦265⟧, (b) ⟦266⟧, (c) ⟦267⟧? Explain each."
answer: "Step 1, (a) is a function: each of 0, 1, 2 appears exactly once. Two inputs sharing the output 5 is allowed. Step 2, (b) is not: input 0 appears twice with different outputs (5 and 7). Step 3, (c) is not a function on ⟦68⟧: input 1 has no output. It is a function on the smaller set ⟦277⟧."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Set operations"]
tools: ["Python", "NumPy"]
topic: {"id": "01.02", "title": "Set operations"}
---

## The big picture

A model is a function: input in, prediction out. Seeing a function as a set of (input, output) pairs makes two things precise: what counts as a function, and why a lookup table is one.

## The definition

A function <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo>:</mo><mi>A</mi><mo>→</mo><mi>B</mi></mrow></math></span> ("f from A to B") assigns **exactly one** output in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> to each input in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>.

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the **domain**: allowed inputs.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> is the **codomain**: where outputs live.

Its **graph** is the set of pairs
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>f</mi><mo stretchy="false">(</mo><mi>a</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo>:</mo><mi>a</mi><mo>∈</mo><mi>A</mi><mo stretchy="false">}</mo><mo>⊆</mo><mi>A</mi><mo>×</mo><mi>B</mi></mrow></math></span></span>
"Exactly one output" means each input appears in exactly one pair.

<div class="ex">
<div class="lab">Worked example</div>

**an embedding table is a function on a finite set**

Token ids <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span>, outputs in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>=</mo><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup></mrow></math></span>.

The table <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>0.1</mn><mo separator="true">,</mo><mn>0.3</mn><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mo stretchy="false">(</mo><mo>−</mo><mn>0.5</mn><mo separator="true">,</mo><mn>0.2</mn><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>0.9</mn><mo separator="true">,</mo><mn>0.0</mn><mo stretchy="false">)</mo><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span> is a function: each id appears once.

Stored as a matrix, row <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>i</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></math></span>. That is `nn.Embedding`.

</div>

```
E = np.array([[0.1, 0.3], [-0.5, 0.2], [0.9, 0.0]])
ids = np.array([2, 0, 2])
print(E[ids].tolist())  # [[0.9, 0.0], [0.1, 0.3], [0.9, 0.0]]
```

Looking up a batch of ids is evaluating the function at each one.

~~~dg-raw
%%viz f_map A=1,2,3 B=a,b f=1:a,2:a,3:b | A function: each input has exactly one arrow out. Two inputs may share an output (1 and 2 both go to a).
~~~

## Not every set of pairs is a function

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>5</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>7</mn><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span> gives input 0 two outputs, so it is not a function. A Python `dict` enforces the rule: assigning a second value to the same key replaces the first.

```
f = {0: 5}
f[0] = 7
print(f)  # {0: 7}
```

## Where this breaks in practice

- **Duplicate keys in data.** Joining tables on a key that is not unique turns a function into a relation with several outputs per input, and rows multiply.

<div class="take">
<div class="lab">Key takeaway</div>

a function is a set of pairs where each input appears exactly once. Tables, dictionaries and embedding layers are functions stored as their graphs.

</div>

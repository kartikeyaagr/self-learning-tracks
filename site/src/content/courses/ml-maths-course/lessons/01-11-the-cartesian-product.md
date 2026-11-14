---
id: "01.11"
course: "ml-maths-course"
module: "01"
slug: "the-cartesian-product"
title: "The Cartesian Product"
keyIdeas: ["⟦220⟧ is the set of all ordered pairs ⟦221⟧ with ⟦222⟧ and ⟦223⟧.", "Sizes multiply: ⟦224⟧. Grid search over several settings costs the product of their sizes.", "⟦225⟧, so a feature vector is one point of a product."]
question: "You grid-search learning rate in ⟦217⟧, batch size in ⟦218⟧ and weight decay in ⟦219⟧. Write the search space, its size, and say how random search over the same ranges differs."
answer: "Step 1, every combination of one value from each list is a Cartesian product: ⟦238⟧. Step 2, sizes multiply: ⟦239⟧ runs. Step 3, random search draws 24 random points from ranges (learning rate on a log scale in ⟦240⟧, and so on). It then tries 24 different learning rates instead of 3. Since the learning rate usually matters most, random search usually finds a better one for the same cost."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Set operations"]
tools: ["Python"]
topic: {"id": "01.02", "title": "Set operations"}
---

## The big picture

Many things in ML are "one choice from here and one choice from there": a hyperparameter setting, a (input, label) pair, a point with an x and a y coordinate. The set of all such combinations is the Cartesian product.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>×</mo><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mi>a</mi><mo separator="true">,</mo><mi>b</mi><mo stretchy="false">)</mo><mo>:</mo><mi>a</mi><mo>∈</mo><mi>A</mi><mo separator="true">,</mo><mtext> </mtext><mi>b</mi><mo>∈</mo><mi>B</mi><mo stretchy="false">}</mo></mrow></math></span></span>
Read "A cross B": every ordered pair with a first entry from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and a second from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>. Order matters inside a pair: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">)</mo><mo mathvariant="normal">≠</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>.

<div class="ex">
<div class="lab">Worked example</div>

**a small product**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mtext>small</mtext><mo separator="true">,</mo><mtext>large</mtext><mo stretchy="false">}</mo></mrow></math></span> (model size), <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi><mo>=</mo><mo stretchy="false">{</mo><mn>0.1</mn><mo separator="true">,</mo><mn>0.01</mn><mo separator="true">,</mo><mn>0.001</mn><mo stretchy="false">}</mo></mrow></math></span> (learning rate).

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>×</mo><mi>B</mi></mrow></math></span> has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>3</mn><mo>=</mo><mn>6</mn></mrow></math></span> pairs: (small, 0.1), (small, 0.01), (small, 0.001), (large, 0.1), (large, 0.01), (large, 0.001).

</div>

~~~dg-raw
%%viz f_prod A=a,b,c B=1,2 c=A×B | A × B for A = {a, b, c} and B = {1, 2}. Each cell is one ordered pair: 3 columns times 2 rows gives 6 pairs.
~~~

## Sizes multiply

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mi>A</mi><mo>×</mo><mi>B</mi><mi mathvariant="normal">∣</mi><mo>=</mo><mi mathvariant="normal">∣</mi><mi>A</mi><mi mathvariant="normal">∣</mi><mo>⋅</mo><mi mathvariant="normal">∣</mi><mi>B</mi><mi mathvariant="normal">∣</mi></mrow></math></span>, and with more sets the sizes keep multiplying.

```
from itertools import product
lrs = [1e-4, 3e-4, 1e-3]
batch = [32, 64, 128, 256]
wd = [0, 0.01]
grid = list(product(lrs, batch, wd))
print(len(grid))  # 24
print(grid[0], grid[-1])  # (0.0001, 32, 0) (0.001, 256, 0.01)
```

~~~dg-flow Grid search walks through a Cartesian product, so its cost is the product of the sizes.
{lr} x {batch} x {wd} -(list every combination)-> *24 settings -(train and evaluate each)-> best setting
~~~

## Spaces are products

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mn>2</mn></msup><mo>=</mo><mi mathvariant="normal">ℝ</mi><mo>×</mo><mi mathvariant="normal">ℝ</mi></mrow></math></span> is the plane, pairs <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></math></span>. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi mathvariant="normal">ℝ</mi><mi>d</mi></msup></mrow></math></span> is lists of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>d</mi></mrow></math></span> real numbers. A labelled example is a pair (input, label), a point of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒳</mi><mo>×</mo><mi mathvariant="normal">𝒴</mi></mrow></math></span>, and a dataset of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> examples is a point of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi mathvariant="normal">𝒳</mi><mo>×</mo><mi mathvariant="normal">𝒴</mi><msup><mo stretchy="false">)</mo><mi>n</mi></msup></mrow></math></span>.

## Where this breaks in practice

- **Grid search wastes runs.** Bergstra and Bengio (2012) showed random search finds good settings faster, because a grid tests very few distinct values of the one setting that matters.
- **Not every combination is valid.** A large batch with the large model might not fit in memory, so the real search space is a subset of the product.
- **<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>×</mo></mrow></math></span> is not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>∪</mo></mrow></math></span>.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo><mo>×</mo><mo stretchy="false">{</mo><mn>3</mn><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mo stretchy="false">(</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span> (pairs), while <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo stretchy="false">}</mo><mo>∪</mo><mo stretchy="false">{</mo><mn>3</mn><mo stretchy="false">}</mo><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo stretchy="false">}</mo></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

a Cartesian product is every combination, and its size is the product of the sizes.

</div>

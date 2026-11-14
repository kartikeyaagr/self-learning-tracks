---
id: "01.07"
course: "ml-maths-course"
module: "01"
slug: "indicator-functions"
title: "Indicator Functions"
keyIdeas: ["⟦143⟧ is 1 if ⟦0⟧ and 0 if not. A boolean array is already a list of indicators.", "Indicators turn logic into arithmetic: count with a sum, \"and\" with a product, \"not\" with ⟦144⟧.", "The average of indicators is the fraction that pass, such as accuracy."]
question: "Predictions ⟦141⟧ and labels ⟦142⟧. Write accuracy with indicators and compute it. Then write the number of class-2 examples predicted correctly as a sum of products of indicators."
answer: "Step 1, the indicators ⟦157⟧ are ⟦154⟧. Step 2, accuracy is their average: ⟦158⟧. Step 3, \"label is 2 and prediction is right\" is an \"and\" of two conditions, so multiply indicators: ⟦159⟧. Step 4, ⟦160⟧ is ⟦161⟧; the products are ⟦162⟧, so the answer is 1."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["NumPy"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

An indicator function answers "is it in the set?" with a number: 1 for yes, 0 for no. That one trick turns counting, accuracy and masking into ordinary arithmetic that a computer (and a formula) can handle.

## The definition

For a set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>,
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true">{</mo><mtable rowspacing="0.36em" columnalign="left left" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mtext>if </mtext><mi>x</mi><mo>∈</mo><mi>A</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mtext>if </mtext><mi>x</mi><mo mathvariant="normal">∉</mo><mi>A</mi></mrow></mstyle></mtd></mtr></mtable></mrow></mrow></math></span></span>
Read <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> as "the indicator of A at x". In NumPy a boolean array already behaves like indicators, because True counts as 1 and False as 0.

~~~dg-raw
%%viz f_ind a=0 b=2 e=co x=-1,4 lab=1_on_[0,2) | The indicator of [0, 2). It is 1 for x inside the set and 0 everywhere else. Read the dots: at x = 0 the value is 1, at x = 2 it is 0.
~~~

## Logic becomes arithmetic

- **Counting:** the number of elements in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∑</mo><mi>x</mi></msub><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>.
- **And:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mn mathvariant="normal">𝟏</mn><mrow><mi>A</mi><mo>∩</mo><mi>B</mi></mrow></msub><mo>=</mo><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><mo>⋅</mo><msub><mn mathvariant="normal">𝟏</mn><mi>B</mi></msub></mrow></math></span>, because only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>×</mo><mn>1</mn><mo>=</mo><mn>1</mn></mrow></math></span>.
- **Not:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mn mathvariant="normal">𝟏</mn><msup><mi>A</mi><mi>c</mi></msup></msub><mo>=</mo><mn>1</mn><mo>−</mo><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub></mrow></math></span>.
- **Or:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mn mathvariant="normal">𝟏</mn><mrow><mi>A</mi><mo>∪</mo><mi>B</mi></mrow></msub><mo>=</mo><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><mo>+</mo><msub><mn mathvariant="normal">𝟏</mn><mi>B</mi></msub><mo>−</mo><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><msub><mn mathvariant="normal">𝟏</mn><mi>B</mi></msub></mrow></math></span> (subtract the overlap so it is not counted twice).
- **Fraction:** the average of indicators is the fraction that pass.

<div class="ex">
<div class="lab">Worked example</div>

**checking the "or" rule**

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> is in both <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>+</mo><mn>1</mn><mo>−</mo><mn>1</mn><mo>⋅</mo><mn>1</mn><mo>=</mo><mn>1</mn></mrow></math></span>. In only <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mo>+</mo><mn>0</mn><mo>−</mo><mn>0</mn><mo>=</mo><mn>1</mn></mrow></math></span>. In neither: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>+</mo><mn>0</mn><mo>−</mo><mn>0</mn><mo>=</mo><mn>0</mn></mrow></math></span>. Correct in every case.

</div>

```
pred = np.array([1, 0, 2, 2, 1])
y = np.array([1, 2, 2, 0, 1])
correct = (pred == y)
print(correct.astype(int))  # [1 0 1 0 1]
print(correct.sum(), correct.mean())  # 3 0.6
```

<div class="ex">
<div class="lab">Worked example</div>

**accuracy by hand**

Indicators <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo separator="true">,</mo><mn>0</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></math></span>. Their sum is 3, so 3 of 5 are right. Accuracy <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mn>3</mn><mi mathvariant="normal">/</mi><mn>5</mn><mo>=</mo><mn>0.6</mn></mrow></math></span>.

Because accuracy is (number right)<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span>, on 5 examples it can only be 0, 0.2, 0.4, 0.6, 0.8 or 1.

</div>

Later (module 10) you will estimate probabilities exactly this way: the probability of an event is approximated by the average of its indicator over many samples.

<div class="take">
<div class="lab">Key takeaway</div>

an indicator is a 0/1 version of "is it in the set?". Sums count, products mean "and", and averages give fractions such as accuracy.

</div>

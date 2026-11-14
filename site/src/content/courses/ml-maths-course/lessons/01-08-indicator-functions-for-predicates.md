---
id: "01.08"
course: "ml-maths-course"
module: "01"
slug: "indicator-functions-for-predicates"
title: "Indicator Functions for Predicates"
keyIdeas: ["⟦163⟧ is 1 when the condition is true and 0 when false. It is the indicator of the set where the condition holds.", "Padding masks are indicators: average the loss over real tokens only by dividing by the sum of the mask.", "Indicators have zero slope almost everywhere, so you cannot train on them directly."]
question: "A batch has sentences of lengths 3 and 5, padded to length 5. Per-token losses are 1 everywhere. Write the mask, the correct mean loss, and the mean you would get by dividing by all 10 positions if padded positions had loss 0."
answer: "Step 1, the mask is ⟦175⟧: ⟦180⟧ and ⟦181⟧. Step 2, the number of real tokens is ⟦182⟧. Step 3, the correct mean is ⟦183⟧. Step 4, dividing by 10 with zeros at the padding gives ⟦184⟧: too low, because padding is being treated as perfectly predicted text."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Introduction to set theory"]
tools: ["NumPy"]
topic: {"id": "01.01", "title": "Introduction to set theory"}
---

## The big picture

A **predicate** is a statement with a blank in it, like "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> is before the end of the sentence" or "the prediction equals the label". Its indicator is 1 wherever the predicate is true. This is how masks, metrics and conditional counts are written in papers.

## The notation

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo><mo>=</mo><mrow><mo fence="true">{</mo><mtable rowspacing="0.36em" columnalign="left left" columnspacing="1em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>1</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mrow><mtext>if </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> is true</mtext></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="false"><mn>0</mn></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="false"><mtext>otherwise</mtext></mstyle></mtd></mtr></mtable></mrow></mrow></math></span></span>
It is the same as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mn mathvariant="normal">𝟏</mn><mi>A</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mi>x</mi><mo>:</mo><mi>P</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo stretchy="false">}</mo></mrow></math></span>. Some papers write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝕀</mi><mo stretchy="false">[</mo><mo>⋅</mo><mo stretchy="false">]</mo></mrow></math></span> or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">[</mo><mo>⋅</mo><mo stretchy="false">]</mo></mrow></math></span> (the Iverson bracket) for the same thing.

<div class="ex">
<div class="lab">Worked example</div>

**three predicates**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mi>x</mi><mo>&gt;</mo><mn>0</mn><mo stretchy="false">]</mo></mrow></math></span> at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow></math></span> is 1, at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi><mo>=</mo><mo>−</mo><mn>2</mn></mrow></math></span> is 0. This is the derivative of ReLU (away from 0).

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>y</mi><mo stretchy="false">]</mo></mrow></math></span> is "the prediction is right".

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mi>t</mi><mo>&lt;</mo><mtext>len</mtext><mo stretchy="false">]</mo></mrow></math></span> is "position <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>t</mi></mrow></math></span> is a real token, not padding".

</div>

## Padding masks

Sentences in a batch have different lengths, so short ones are padded with filler tokens to a common length. The mask <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>m</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mo>=</mo><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mi>t</mi><mo>&lt;</mo><msub><mtext>len</mtext><mi>i</mi></msub><mo stretchy="false">]</mo></mrow></math></span> marks the real tokens.

```
lengths = np.array([3, 5])
T = 5
mask = (np.arange(T)[None, :] < lengths[:, None]).astype(int)
print(mask.tolist())  # [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
print(mask.sum())  # 8
```

The mean loss over real tokens is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∑</mo><mrow><mi>i</mi><mo separator="true">,</mo><mi>t</mi></mrow></msub><msub><mi>m</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><msub><mi mathvariant="normal">ℓ</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub><mi mathvariant="normal">/</mi><msub><mo>∑</mo><mrow><mi>i</mi><mo separator="true">,</mo><mi>t</mi></mrow></msub><msub><mi>m</mi><mrow><mi>i</mi><mi>t</mi></mrow></msub></mrow></math></span>. Dividing by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>2</mn><mo>×</mo><mn>5</mn><mo>=</mo><mn>10</mn></mrow></math></span> instead quietly counts padding as zero-loss text.

## Conditional counts

"Accuracy on class 2" is a ratio of two indicator sums:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mfrac><mrow><munder><mo>∑</mo><mi>i</mi></munder><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mn>2</mn><mo stretchy="false">]</mo><mtext> </mtext><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><msub><mover accent="true"><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mn>2</mn><mo stretchy="false">]</mo></mrow><mrow><munder><mo>∑</mo><mi>i</mi></munder><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mn>2</mn><mo stretchy="false">]</mo></mrow></mfrac></mrow></math></span></span>

```
pred = np.array([1, 0, 2, 2, 1])
y = np.array([1, 2, 2, 0, 1])
m = (y == 2)
print(((pred == 2) & m).sum() / m.sum())  # 0.5
```

Two examples have label 2 and one of them is predicted 2, so the class accuracy is 0.5.

## Where this breaks in practice

- **No useful gradient.** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mi>x</mi><mo>&gt;</mo><mn>0</mn><mo stretchy="false">]</mo></mrow></math></span> is flat except for a jump at 0, so its slope is 0 almost everywhere. You cannot train by maximising accuracy directly; you train on a smooth stand-in such as cross-entropy.
- **Empty denominators.** If class 2 is missing from the batch, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mo>∑</mo><mi>i</mi></msub><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><msub><mi>y</mi><mi>i</mi></msub><mo>=</mo><mn>2</mn><mo stretchy="false">]</mo><mo>=</mo><mn>0</mn></mrow></math></span> and the class accuracy is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mi mathvariant="normal">/</mi><mn>0</mn></mrow></math></span>. Check for it.

<div class="take">
<div class="lab">Key takeaway</div>

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn mathvariant="normal">𝟏</mn><mo stretchy="false">[</mo><mtext>condition</mtext><mo stretchy="false">]</mo></mrow></math></span> turns any yes/no question into 0 or 1. Masks, per-class metrics and ReLU's slope are all written this way.

</div>

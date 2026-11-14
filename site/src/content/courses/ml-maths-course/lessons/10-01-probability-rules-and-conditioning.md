---
id: "10.01"
course: "ml-maths-course"
module: "10"
slug: "probability-rules-and-conditioning"
title: "Probability Rules and Conditioning"
keyIdeas: ["Axioms: ⟦3206⟧, ⟦3207⟧, and probabilities of events that cannot happen together add up.", "Conditional probability ⟦3208⟧: keep only the outcomes in ⟦41⟧ and rescale.", "Product and chain rule: ⟦3209⟧, and ⟦3210⟧.", "Total probability: ⟦3211⟧ when the ⟦3212⟧ split all possibilities into non-overlapping cases."]
question: "Roll one fair die. ⟦8⟧ is \"even\" and ⟦41⟧ is \"at least 4\". Find ⟦3204⟧ and ⟦3205⟧."
answer: "Step 1, ⟦3229⟧ and ⟦3230⟧, so ⟦3231⟧. Step 2, ⟦3232⟧. Step 3, ⟦3233⟧: among ⟦3234⟧, two are even."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Probability rules and Bayesian updating"]
tools: ["NumPy"]
topic: {"id": "10.01", "title": "Probability rules and Bayesian updating"}
---

## The big picture

Machine learning models don't output certainties; they output beliefs. "This email is spam with probability 0.93." Probability is the language for those beliefs. This lesson covers the few rules everything else is built from.

## Outcomes and events

- The **sample space** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Ω</mi></mrow></math></span> is the set of all possible outcomes. Roll a die: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Ω</mi><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo separator="true">,</mo><mn>2</mn><mo separator="true">,</mo><mn>3</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>5</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">}</mo></mrow></math></span>.
- An **event** is a subset of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Ω</mi></mrow></math></span> (module 01's sets). "Roll an even number" is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>=</mo><mo stretchy="false">{</mo><mn>2</mn><mo separator="true">,</mo><mn>4</mn><mo separator="true">,</mo><mn>6</mn><mo stretchy="false">}</mo></mrow></math></span>.
- A **probability** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi></mrow></math></span> gives each event a number between 0 and 1.

## The rules

Three axioms:

1. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>≥</mo><mn>0</mn></mrow></math></span> for every event.
2. <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi mathvariant="normal">Ω</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></mrow></math></span>: something happens.
3. If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> cannot both happen, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∪</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>+</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span>.

Consequences you use constantly:

- **Complement:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>not </mtext><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo>−</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo></mrow></math></span>.
- **Union:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∪</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>+</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo><mo>−</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span>. Subtract the overlap so it isn't counted twice.
- **Union bound:** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mn>1</mn></msub><mo>∪</mo><msub><mi>A</mi><mn>2</mn></msub><mo>∪</mo><mo>…</mo><mtext> </mtext><mo stretchy="false">)</mo><mo>≤</mo><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mo>+</mo><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mn>2</mn></msub><mo stretchy="false">)</mo><mo>+</mo><mo>…</mo></mrow></math></span>. Crude but always true.

~~~dg-raw
%%viz f_venn sh=i a=even b=at_least_4 n=1,2,1,2 | One die. "Even" is {2, 4, 6} and "at least 4" is {4, 5, 6}. The overlap {4, 6} holds 2 of the 6 outcomes.
~~~

## Conditioning

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∣</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo stretchy="false">)</mo></mrow><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span></span>
Throw away every outcome outside <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span>, then rescale so what's left adds up to 1. Rearranged, it gives the **product rule** <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∣</mo><mi>B</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span>, and applied again and again, the **chain rule**:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>A</mi><mi>n</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mn>2</mn></msub><mo>∣</mo><msub><mi>A</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mo>⋯</mo><mi>P</mi><mo stretchy="false">(</mo><msub><mi>A</mi><mi>n</mi></msub><mo>∣</mo><msub><mi>A</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>A</mi><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow></math></span></span>

<div class="ex">
<div class="lab">Worked example</div>

**the chain rule in a language model**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>&quot;the cat sat&quot;</mtext><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mtext>the</mtext><mo stretchy="false">)</mo><mo>⋅</mo><mi>P</mi><mo stretchy="false">(</mo><mtext>cat</mtext><mo>∣</mo><mtext>the</mtext><mo stretchy="false">)</mo><mo>⋅</mo><mi>P</mi><mo stretchy="false">(</mo><mtext>sat</mtext><mo>∣</mo><mtext>the cat</mtext><mo stretchy="false">)</mo></mrow></math></span>.

With <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.05</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.01</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.2</mn></mrow></math></span>: the sentence has probability <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.0001</mn></mrow></math></span>. In log space: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>log</mi><mo>⁡</mo><mn>0.05</mn><mo>+</mo><mi>log</mi><mo>⁡</mo><mn>0.01</mn><mo>+</mo><mi>log</mi><mo>⁡</mo><mn>0.2</mn><mo>≈</mo><mo>−</mo><mn>9.2</mn></mrow></math></span>.

</div>

## Total probability

If <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>B</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>B</mi><mi>k</mi></msub></mrow></math></span> split all possibilities into non-overlapping cases,
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mo>∑</mo><mi>i</mi></munder><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∣</mo><msub><mi>B</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><msub><mi>B</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></math></span></span>
The chance of <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span> is the average of its chance in each case, weighted by how likely each case is.

```
d = np.arange(1, 7)                    # the six equally likely faces
A, B = d % 2 == 0, d >= 4
print(round((A | B).mean(), 3), round(A[B].mean(), 3))  # 0.667 0.667
```

<div class="take">
<div class="lab">Key takeaway</div>

conditioning keeps the outcomes in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> and rescales; the product and chain rules multiply conditionals; total probability averages over cases.

</div>

## Where this breaks in practice

- Multiplying many small probabilities underflows to 0. Add log-probabilities instead.

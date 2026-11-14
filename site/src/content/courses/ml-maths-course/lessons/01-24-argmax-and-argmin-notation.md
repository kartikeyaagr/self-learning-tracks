---
id: "01.24"
course: "ml-maths-course"
module: "01"
slug: "argmax-and-argmin-notation"
title: "Argmax and Argmin Notation"
keyIdeas: ["⟦416⟧ is the best value, a number. ⟦417⟧ is the set of inputs that achieve it.", "Argmax is defined with set-builder notation, so it can have one element, several (ties), or none.", "\"Max\" answers \"how good?\"; \"argmax\" answers \"where?\"."]
question: "For each, give the max (or sup) and the argmax: (a) ⟦413⟧ on ⟦19⟧, (b) ⟦414⟧ on ⟦19⟧, (c) ⟦415⟧ on ⟦19⟧."
answer: "Step 1, (a) ⟦428⟧, with equality only at ⟦391⟧: max 0, argmax ⟦429⟧. Step 2, (b) ⟦430⟧, with equality when ⟦431⟧: max 0, argmax ⟦432⟧, a tie between two points. Step 3, (c) ⟦433⟧ is always negative and approaches 0 as ⟦7⟧ grows: sup 0, never reached, so the argmax is ⟦55⟧."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 1, 8"}, {"title": "Lebl: Basic Analysis I", "url": "https://www.jirka.org/ra/", "note": "Read: §1.2, §7.2"}]
tags: ["Properties of sets"]
tools: ["NumPy"]
topic: {"id": "01.03", "title": "Properties of sets"}
---

~~~dg-raw
%%viz f_bars v=0.1,0.7,0.2 l=cat,dog,fox hi=max yl=probability | argmax returns the label of the tallest bar (dog), not the height 0.7. max would return 0.7.
~~~

## The big picture

Training finds weights, not a loss value; classification picks a class, not a score. In both cases you want the *location* of the best value. That is the argmax (or argmin), and its notation appears in almost every ML paper.

## The definitions

For a function <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi></mrow></math></span> on a set <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span>:

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mrow><mi>max</mi><mo>⁡</mo></mrow><mrow><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></msub><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, the **best value**: a number.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>arg</mi><mo>⁡</mo><msub><mrow><mi>max</mi><mo>⁡</mo></mrow><mrow><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></msub><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, the **set of inputs** that give it:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>arg</mi><mo>⁡</mo><munder><mrow><mi>max</mi><mo>⁡</mo></mrow><mrow><mi>x</mi><mo>∈</mo><mi>S</mi></mrow></munder><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">{</mo><mi>x</mi><mo>∈</mo><mi>S</mi><mo>:</mo><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≥</mo><mi>f</mi><mo stretchy="false">(</mo><mi>z</mi><mo stretchy="false">)</mo><mtext> for every </mtext><mi>z</mi><mo>∈</mo><mi>S</mi><mo stretchy="false">}</mo></mrow></math></span></span>
Read it: "the <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>S</mi></mrow></math></span> whose value is at least as big as every other value". Argmin is the same with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>≤</mo></mrow></math></span>.

## One, many or none

<div class="ex">
<div class="lab">Worked example</div>

**three argmaxes**

One: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mo stretchy="false">(</mo><mi>x</mi><mo>−</mo><mn>2</mn><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span> peaks only at 2: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>arg</mi><mo>⁡</mo><mi>max</mi><mo>⁡</mo><mo>=</mo><mo stretchy="false">{</mo><mn>2</mn><mo stretchy="false">}</mo></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>max</mi><mo>⁡</mo><mo>=</mo><mn>0</mn></mrow></math></span>.

Many: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">(</mo><msup><mi>x</mi><mn>2</mn></msup><mo>−</mo><mn>1</mn><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span> has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>arg</mi><mo>⁡</mo><mi>min</mi><mo>⁡</mo><mo>=</mo><mo stretchy="false">{</mo><mo>−</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">}</mo></mrow></math></span>, two equally good minima. Neural networks have many such ties: swapping two hidden units gives the same loss.

None: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>f</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>x</mi></mrow></msup></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">ℝ</mi></mrow></math></span> approaches 0 but never reaches it: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>inf</mi><mo>⁡</mo><mi>f</mi><mo>=</mo><mn>0</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>arg</mi><mo>⁡</mo><mi>min</mi><mo>⁡</mo><mi>f</mi><mo>=</mo><mi mathvariant="normal">∅</mi></mrow></math></span>.

</div>

## Argmax in ML formulas

- Classification: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>arg</mi><mo>⁡</mo><msub><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>k</mi></msub><msub><mi>p</mi><mi>k</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span>, the class with the highest predicted probability.
- Training: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>w</mi><mo>^</mo></mover><mo>=</mo><mi>arg</mi><mo>⁡</mo><msub><mrow><mi>min</mi><mo>⁡</mo></mrow><mi>w</mi></msub><mi>L</mi><mo stretchy="false">(</mo><mi>w</mi><mo stretchy="false">)</mo></mrow></math></span>, the best weights.
- Maximum likelihood (module 12): <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>θ</mi><mo>^</mo></mover><mo>=</mo><mi>arg</mi><mo>⁡</mo><msub><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>θ</mi></msub><mi>p</mi><mo stretchy="false">(</mo><mtext>data</mtext><mo>∣</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></math></span>.

When the argmax has exactly one element, people write <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover accent="true"><mi>y</mi><mo>^</mo></mover><mo>=</mo><mi>arg</mi><mo>⁡</mo><mi>max</mi><mo>⁡</mo></mrow></math></span> as if it were a single value. That shortcut assumes no ties.

```
p = np.array([0.1, 0.6, 0.3])
print(p.max(), np.argmax(p))  # 0.6 1
```

<div class="take">
<div class="lab">Key takeaway</div>

max is a number, argmax is a set of locations. Always ask whether it is empty or has ties.

</div>

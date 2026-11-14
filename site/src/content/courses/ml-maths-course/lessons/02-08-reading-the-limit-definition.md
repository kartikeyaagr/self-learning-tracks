---
id: "02.08"
course: "ml-maths-course"
module: "02"
slug: "reading-the-limit-definition"
title: "Reading the Limit Definition"
keyIdeas: ["\"⟦522⟧\" means ⟦523⟧.", "You choose the tolerance ⟦305⟧, then a cut-off ⟦519⟧ must exist, and it may depend on ⟦305⟧.", "Negating it with the flipping rule gives the meaning of \"does not converge to ⟦491⟧\"."]
question: "For ⟦517⟧ and ⟦518⟧, find a cut-off ⟦519⟧ that works for ⟦520⟧. Then explain why swapping to ⟦521⟧ would fail."
answer: "Step 1, we need ⟦541⟧ for all ⟦538⟧, which means ⟦542⟧. So ⟦543⟧ works. Step 2, smaller ⟦305⟧ needs larger ⟦519⟧: ⟦532⟧ needs ⟦533⟧. Step 3, ⟦521⟧ would require one fixed ⟦519⟧ that works for every tolerance, including ⟦544⟧. But ⟦545⟧, so that ⟦519⟧ fails. No fixed ⟦519⟧ works."
sources: [{"title": "Hammack: Book of Proof", "url": "https://richardhammack.github.io/BookOfProof/", "note": "Read: ch. 2, 4-6"}, {"title": "Levin: Discrete Mathematics, an Open Introduction", "url": "https://discrete.openmathbooks.org/", "note": "Read: §0.2, 3.1"}]
tags: ["Statements, predicates and truth"]
tools: ["Python"]
topic: {"id": "02.01", "title": "Statements, predicates and truth"}
---

## The big picture

Convergence of training, of estimators and of series is defined with three quantifiers in a row. Once you can read this one definition, most "as <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>→</mo><mi mathvariant="normal">∞</mi></mrow></math></span>" statements in ML papers become readable.

## The definition

"The sequence <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>n</mi></msub></mrow></math></span> goes to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi></mrow></math></span>" is written
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∀</mi><mi>ε</mi><mo>&gt;</mo><mn>0</mn><mtext> </mtext><mi mathvariant="normal">∃</mi><mi>N</mi><mtext> </mtext><mi mathvariant="normal">∀</mi><mi>n</mi><mo>≥</mo><mi>N</mi><mo>:</mo><mtext> </mtext><mi mathvariant="normal">∣</mi><msub><mi>a</mi><mi>n</mi></msub><mo>−</mo><mi>L</mi><mi mathvariant="normal">∣</mi><mo>&lt;</mo><mi>ε</mi></mrow></math></span></span>
Read it aloud: "for every tolerance epsilon, there is a cut-off N, such that every term after N is within epsilon of L".

You pick how close you want to be. Then I am allowed to pick <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi></mrow></math></span>, and my <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi></mrow></math></span> may depend on your <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=1/x x=0,25 y=0,0.6 seq=2 hb=0,0.1 hbl=within_ε_=_0.1_of_0 vl=11,N_=_11 tk=5 | The terms 1/n from n = 2. For ε = 0.1 the band is shaded. From n = 11 on, every term stays inside the band.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the limit definition with numbers**

Take <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>n</mi></msub><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>n</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi><mo>=</mo><mn>0</mn></mrow></math></span>.

If you choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi><mo>=</mo><mn>0.1</mn></mrow></math></span>, I choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo>=</mo><mn>11</mn></mrow></math></span>: every <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>≥</mo><mn>11</mn></mrow></math></span> has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><mi>n</mi><mo>≤</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>11</mn><mo>&lt;</mo><mn>0.1</mn></mrow></math></span>.

If you choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi><mo>=</mo><mn>0.001</mn></mrow></math></span>, I choose <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi><mo>=</mo><mn>1001</mn></mrow></math></span>. My <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi></mrow></math></span> grew as your <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi></mrow></math></span> shrank. That is allowed because <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∃</mi><mi>N</mi></mrow></math></span> comes after <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∀</mi><mi>ε</mi></mrow></math></span>.

</div>

## Finding the cut-off in code

For each tolerance, search for the smallest cut-off that works:

```
def cutoff(eps):
    N = 1
    while 1 / N >= eps:
        N += 1
    return N

print([cutoff(e) for e in (0.1, 0.01, 0.001)])  # [11, 101, 1001]
```

## Not converging

Negate with the flipping rule, three times. "<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>n</mi></msub></mrow></math></span> does **not** go to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>L</mi></mrow></math></span>" is
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">∃</mi><mi>ε</mi><mo>&gt;</mo><mn>0</mn><mtext> </mtext><mi mathvariant="normal">∀</mi><mi>N</mi><mtext> </mtext><mi mathvariant="normal">∃</mi><mi>n</mi><mo>≥</mo><mi>N</mi><mo>:</mo><mtext> </mtext><mi mathvariant="normal">∣</mi><msub><mi>a</mi><mi>n</mi></msub><mo>−</mo><mi>L</mi><mi mathvariant="normal">∣</mi><mo>≥</mo><mi>ε</mi></mrow></math></span></span>
In words: "there is some tolerance that the sequence keeps breaking, however far out you go".

<div class="ex">
<div class="lab">Worked example</div>

**a sequence that does not converge**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>n</mi></msub><mo>=</mo><mo stretchy="false">(</mo><mo>−</mo><mn>1</mn><msup><mo stretchy="false">)</mo><mi>n</mi></msup></mrow></math></span> jumps between <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>−</mo><mn>1</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn></mrow></math></span>. Does it go to 1?

Pick <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>ε</mi><mo>=</mo><mn>1</mn></mrow></math></span>. For any <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>N</mi></mrow></math></span>, some odd <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>≥</mo><mi>N</mi></mrow></math></span> has <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>a</mi><mi>n</mi></msub><mo>=</mo><mo>−</mo><mn>1</mn></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">∣</mi><mrow><mo>−</mo><mn>1</mn></mrow><mo>−</mo><mn>1</mn><mi mathvariant="normal">∣</mi><mo>=</mo><mn>2</mn><mo>≥</mo><mn>1</mn></mrow></math></span>. So the sequence does not converge to 1.

</div>

## Where this breaks in practice

- **Plots are not proofs.** A loss curve that flattens out after 10,000 steps shows small changes in that window. It does not show convergence, which is a claim about all later steps.

<div class="take">
<div class="lab">Key takeaway</div>

for every tolerance there must be a cut-off. The cut-off may depend on the tolerance, never the other way round.

</div>

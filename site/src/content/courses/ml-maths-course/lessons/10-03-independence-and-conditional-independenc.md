---
id: "10.03"
course: "ml-maths-course"
module: "10"
slug: "independence-and-conditional-independence"
title: "Independence and Conditional Independence"
keyIdeas: ["⟦8⟧ and ⟦41⟧ are independent when ⟦3255⟧, or equivalently ⟦3256⟧.", "Conditional independence ⟦3257⟧: independent once ⟦1203⟧ is known. Naive Bayes and graphical models rely on it.", "Independence and conditional independence do not imply each other."]
question: "One die. Are \"even\" and \"at least 4\" independent? Are two separate fair coin flips?"
answer: "Step 1, ⟦3261⟧ but ⟦3262⟧. Not independent. Step 2, ⟦3263⟧. Independent."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Probability rules and Bayesian updating"]
tools: ["NumPy"]
topic: {"id": "10.01", "title": "Probability rules and Bayesian updating"}
---

## The big picture

Independence lets you multiply probabilities, which is what makes most models computable. It is a modelling assumption, not a default, and getting it wrong makes a model overconfident.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>A</mi><mo>⊥</mo><mi>B</mi><mtext>  </mtext><mo>⟺</mo><mtext>  </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span></span>
Knowing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>B</mi></mrow></math></span> tells you nothing about <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi></mrow></math></span>.

| Statement | Meaning |
| --- | --- |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>⊥</mo><mi>B</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo stretchy="false">)</mo></mrow></math></span> |
| <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>A</mi><mo>⊥</mo><mi>B</mi><mo>∣</mo><mi>C</mi></mrow></math></span> | <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∩</mo><mi>B</mi><mo>∣</mo><mi>C</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>A</mi><mo>∣</mo><mi>C</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>B</mi><mo>∣</mo><mi>C</mi><mo stretchy="false">)</mo></mrow></math></span> |
| pairwise independent | does not imply that three or more events are jointly independent |

## Conditional independence

Two symptoms of the same disease are dependent: one makes the other more likely. Once you know whether the patient has the disease, they may be independent. Naive Bayes assumes exactly this: words are independent **given the class**.

```
rng = np.random.default_rng(0)
ill = rng.random(1_000_000) < 0.3
s1 = rng.random(ill.size) < np.where(ill, 0.8, 0.1)
s2 = rng.random(ill.size) < np.where(ill, 0.8, 0.1)
print(round(s1.mean() * s2.mean(), 3), round((s1 & s2).mean(), 3))  # 0.096 0.199
print(round(s1[ill].mean() * s2[ill].mean(), 2), round((s1 & s2)[ill].mean(), 2))  # 0.64 0.64
```

Overall, both symptoms together are twice as common as independence would predict. Among the ill, the product rule holds.

~~~dg-raw
%%viz f_bars v=9.6,19.9 l=if_independent,actual hi=none yl=P(both_symptoms)_in_% | Two symptoms of one illness. Ignoring the shared cause, they look dependent: both together happen about twice as often as the product of their rates.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

independence means probabilities multiply; conditional independence means they multiply once a shared cause is known.

</div>

## Where this breaks in practice

- Naive Bayes counts correlated words as separate evidence, so its probabilities are overconfident even when its rankings are good.

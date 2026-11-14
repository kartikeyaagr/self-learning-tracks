---
id: "10.33"
course: "ml-maths-course"
module: "10"
slug: "the-multinomial-distribution"
title: "The Multinomial Distribution"
keyIdeas: ["Multinomial(⟦3536⟧) counts how many of ⟦233⟧ categorical draws land in each class.", "⟦3537⟧; each count has mean ⟦3538⟧ and variance ⟦3539⟧.", "Bag-of-words document models are multinomials over the vocabulary."]
question: "Roll a fair die 6 times. What is the chance each face appears exactly once?"
answer: "Step 1, ⟦3544⟧. Step 2, that is about 0.0154."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
---

## The big picture

The binomial with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>K</mi></mrow></math></span> classes: class counts in a batch, word counts in a document.

## The PMF

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><msub><mi>k</mi><mn>1</mn></msub><mo separator="true">,</mo><mo>…</mo><mo separator="true">,</mo><msub><mi>k</mi><mi>K</mi></msub><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>n</mi><mo stretchy="false">!</mo></mrow><mrow><msub><mi>k</mi><mn>1</mn></msub><mo stretchy="false">!</mo><mo>⋯</mo><msub><mi>k</mi><mi>K</mi></msub><mo stretchy="false">!</mo></mrow></mfrac><msubsup><mi>π</mi><mn>1</mn><msub><mi>k</mi><mn>1</mn></msub></msubsup><mo>⋯</mo><msubsup><mi>π</mi><mi>K</mi><msub><mi>k</mi><mi>K</mi></msub></msubsup></mrow></math></span></span>
The fraction counts the orderings that give the same counts.

```
from scipy import stats
print(round(stats.multinomial(6, [1 / 6] * 6).pmf([1] * 6), 4))  # 0.0154
rng = np.random.default_rng(0)
c = rng.multinomial(100, [0.7, 0.2, 0.1], size=100000)
print(np.round(c.mean(0), 1), np.round(c.var(0)))  # [70. 20. 10.] [21. 16.  9.]
```

Each count's variance matches <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><msub><mi>π</mi><mi>i</mi></msub><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msub><mi>π</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></math></span>: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>21</mn></mrow></math></span>, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>16</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>9</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

the multinomial counts class frequencies in <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> independent categorical draws.

</div>

## Where this breaks in practice

- Counts are negatively correlated: more of one class means fewer of the others, since they add up to <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span>.

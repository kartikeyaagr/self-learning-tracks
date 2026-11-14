---
id: "11.01"
course: "ml-maths-course"
module: "11"
slug: "pairs-of-random-variables"
title: "Pairs of Random Variables"
keyIdeas: ["Many questions involve two random quantities at once: true and predicted label, a word and the next word.", "Studying each variable alone loses how they move together.", "The joint distribution describes the pair; everything else is derived from it."]
question: "A classifier's accuracy is 90% and the positive rate in the data is 10%. Can you work out its precision from these two numbers alone?"
answer: "Step 1, precision depends on how predictions and truth combine, cell by cell. Step 2, the two numbers are summaries of each variable separately, so they don't fix the joint table. Precision cannot be determined."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two discrete variables"]
tools: ["NumPy"]
topic: {"id": "11.01", "title": "Two discrete variables"}
---

## The big picture

Module 10 dealt with one random variable at a time. Real problems involve several, and the interesting part is usually how they relate: does a symptom predict a disease, does a feature predict a label?

## Two variables, one experiment

A pair <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo stretchy="false">(</mo><mi>X</mi><mo separator="true">,</mo><mi>Y</mi><mo stretchy="false">)</mo></mrow></math></span> is two numbers produced by the same random experiment. Pick a random email: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> is "contains the word free", <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> is "is spam". The pair has four possible outcomes.

<div class="ex">
<div class="lab">Worked example</div>

**same marginals, different relationships**

Two tables both have <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>Y</mi><mo>=</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span>.

In one, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> always agree: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mn>1</mn><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span>. In the other they are unrelated: each of the four cells is <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.25</mn></mrow></math></span>.

Looking at <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Y</mi></mrow></math></span> separately, you cannot tell these apart.

</div>

```
agree = np.array([[0.5, 0.0], [0.0, 0.5]])
unrelated = np.full((2, 2), 0.25)
for P in (agree, unrelated):
    print(P.sum(1), P.sum(0), P[0, 0] + P[1, 1])
# [0.5 0.5] [0.5 0.5] 1.0
# [0.5 0.5] [0.5 0.5] 0.5
```

~~~dg-raw
%%viz f_bars v=50,0,0,50,25,25,25,25 l=00,01,10,11,00,01,10,11 hi=none yl=%_of_emails_in_each_(X~_Y)_cell | Left four bars: X and Y always agree. Right four: they are unrelated. Both have the same separate rates, 50% each.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

the behaviour of a pair is not determined by each variable alone; you need their joint distribution.

</div>

## Where this breaks in practice

- Reporting only per-feature statistics hides interactions. Two features can be useless alone and decisive together.

---
id: "12.20"
course: "ml-maths-course"
module: "12"
slug: "choosing-a-test-set-size"
title: "Choosing a Test-Set Size"
keyIdeas: ["The margin of error for a proportion is about ⟦3944⟧.", "To get margin ⟦958⟧: ⟦3945⟧.", "Halving the margin needs four times the data."]
question: "How many examples measure a 90% accuracy to within ±1 point?"
answer: "Step 1, ⟦3947⟧. Step 2, round up to 3,458."
sources: [{"title": "MIT 18.05 Introduction to Probability and Statistics", "url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/", "note": "Read: readings 10-24"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §8.3"}, {"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 10"}]
tags: ["Interval estimation"]
tools: ["NumPy"]
topic: {"id": "12.03", "title": "Interval estimation"}
---

## The big picture

Before collecting or labelling a test set, decide how precise the answer needs to be. The formula tells you the cost.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>n</mi><mo>≈</mo><mfrac><mrow><mn>1.9</mn><msup><mn>6</mn><mn>2</mn></msup><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>p</mi><mo stretchy="false">)</mo></mrow><msup><mi>m</mi><mn>2</mn></msup></mfrac></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=3.8416*0.09/(x*x/10000)/1000 x=0.5,3 y=0,14 tk=0.5 ytk=2 pts=1,3~458;2,865 | Test-set size, in thousands of examples, needed to measure a 90% accuracy to within ±m percentage points. Halving the margin quadruples the size.
~~~

```
for m in (0.02, 0.01, 0.005):
    print(m, int(np.ceil(1.96 ** 2 * 0.09 / m ** 2)))
# 0.02 865
# 0.01 3458
# 0.005 13830
```

<div class="take">
<div class="lab">Key takeaway</div>

test-set size grows with the inverse square of the margin you want.

</div>

## Where this breaks in practice

- For comparing two models, a paired analysis (lesson 12.22) needs far fewer examples than two separate margins suggest.

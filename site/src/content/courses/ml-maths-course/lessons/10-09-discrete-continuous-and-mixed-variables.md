---
id: "10.09"
course: "ml-maths-course"
module: "10"
slug: "discrete-continuous-and-mixed-variables"
title: "Discrete, Continuous and Mixed Variables"
keyIdeas: ["Discrete variables have a PMF ⟦3318⟧; continuous ones have a density; mixed ones have a CDF with jumps plus a smooth part.", "A ReLU of a Gaussian is mixed: a lump of 0.5 at zero plus a density on ⟦279⟧.", "The CDF is the one description that works for all three."]
question: "⟦3314⟧ and ⟦3315⟧. What are ⟦3316⟧ and ⟦3317⟧?"
answer: "Step 1, ⟦3320⟧. Step 2, ⟦3321⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous random variables"]
tools: ["NumPy"]
topic: {"id": "10.02", "title": "Continuous random variables"}
---

## The big picture

Real quantities in ML are not always purely one type. Activations after a ReLU, rainfall, and clipped losses all have an exact value that occurs with positive probability, plus a continuous spread.

## Three types

| Type | Described by | Example |
| --- | --- | --- |
| Discrete | PMF <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo></mrow></math></span> | class label, token id |
| Continuous | density <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> | pixel intensities, a latent vector |
| Mixed | CDF with jumps | ReLU of a Gaussian |

~~~dg-raw
%%viz f_curve f=x<0?0:1/(1+Math.exp(-1.702*x)) x=-2,3 y=0,1 tk=1 ytk=0.25 pts=0,jump_of_0.5 | The CDF of a ReLU of a standard normal. It is 0 for negative values, jumps by 0.5 at zero, then follows the normal CDF up to 1.
~~~

```
rng = np.random.default_rng(0)
y = np.maximum(0, rng.normal(size=1_000_000))
print(round((y == 0).mean(), 3), round((y > 1).mean(), 3))  # 0.5 0.159
```

<div class="take">
<div class="lab">Key takeaway</div>

discrete variables have point masses, continuous ones have densities, and mixed ones have both; the CDF handles all of them.

</div>

## Where this breaks in practice

- A mixed variable has no density at its jump. A Gaussian likelihood fitted to ReLU outputs assigns zero probability to the most common value.

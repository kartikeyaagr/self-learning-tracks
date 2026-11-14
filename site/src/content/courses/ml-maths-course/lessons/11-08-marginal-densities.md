---
id: "11.08"
course: "ml-maths-course"
module: "11"
slug: "marginal-densities"
title: "Marginal Densities"
keyIdeas: ["The marginal density integrates out the other variable: ⟦3114⟧.", "Read the limits of the inner integral from a sketch of the support.", "Marginals of a valid joint density are valid densities."]
question: "For the uniform density 2 on ⟦3075⟧, find ⟦3624⟧ and ⟦3325⟧."
answer: "Step 1, at fixed ⟦7⟧, ⟦509⟧ runs from 0 to ⟦7⟧: ⟦3659⟧. Step 2, at fixed ⟦509⟧, ⟦7⟧ runs from ⟦509⟧ to 1: ⟦3660⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two continuous variables"]
tools: ["NumPy"]
topic: {"id": "11.02", "title": "Two continuous variables"}
---

## The big picture

Integrating out a variable is how you get the distribution of the variable you care about. It is the same as summing a table's rows, done with an integral.

## Integrating out

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo>∫</mo><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi></mrow></math></span></span>

~~~dg-raw
%%viz f_curve f=2*x g=2*(1-x) x=0,1 y=0,2.2 tk=0.25 fl=p_X(x)_=_2x gl=p_Y(y)_=_2(1_−_y) | The two marginals of the uniform triangle. X tends to be large and Y small, because the triangle sits below the diagonal.
~~~

```
rng = np.random.default_rng(0)
u = rng.random((400000, 2))
x, y = u.max(1), u.min(1)
print(round(x.mean(), 2), round(y.mean(), 2))  # 0.67 0.33
```

The means match <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mi>x</mi><mo>⋅</mo><mn>2</mn><mi>x</mi><mtext> </mtext><mi>d</mi><mi>x</mi><mo>=</mo><mn>2</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mo>∫</mo><mn>0</mn><mn>1</mn></msubsup><mi>y</mi><mo>⋅</mo><mn>2</mn><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>y</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>y</mi><mo>=</mo><mn>1</mn><mi mathvariant="normal">/</mi><mn>3</mn></mrow></math></span>.

<div class="take">
<div class="lab">Key takeaway</div>

integrate the joint density over the other variable to get a marginal density.

</div>

## Where this breaks in practice

- Wrong limits are the usual mistake; check that the marginal integrates to 1.

---
id: "11.10"
course: "ml-maths-course"
module: "11"
slug: "independence-for-continuous-variables"
title: "Independence for Continuous Variables"
keyIdeas: ["⟦513⟧ and ⟦512⟧ are independent when the density factorises: ⟦3642⟧ everywhere.", "A support that is not a rectangle rules out independence at once.", "A formula that looks like a product, such as ⟦3673⟧ on a triangle, can still be dependent."]
question: "⟦3671⟧ for ⟦3672⟧. Are ⟦513⟧ and ⟦512⟧ independent? Find ⟦3122⟧."
answer: "Step 1, it splits as ⟦3678⟧ on a quadrant, so yes: two independent Exponential(1) variables. Step 2, ⟦3679⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two continuous variables"]
tools: ["NumPy"]
topic: {"id": "11.02", "title": "Two continuous variables"}
---

## The big picture

Independence makes joint densities easy: two one-dimensional densities multiplied together. The trap is the support.

## Factorising

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mspace width="1em"/><mtext>for all </mtext><mi>x</mi><mo separator="true">,</mo><mi>y</mi></mrow></math></span></span>
This must hold where the density is zero too. If the region where <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo>&gt;</mo><mn>0</mn></mrow></math></span> is not a product of an <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>-range and a <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>-range, knowing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span> restricts <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>y</mi></mrow></math></span>.

```
rng = np.random.default_rng(1)
e = rng.exponential(size=(1_000_000, 2))
print(round((e.sum(1) <= 1).mean(), 3), round(1 - 2 / np.e, 3))  # 0.264 0.264
```

<div class="ex">
<div class="lab">Worked example</div>

**looks like a product, isn't**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo separator="true">,</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mn>8</mn><mi>x</mi><mi>y</mi></mrow></math></span> on <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0</mn><mo>≤</mo><mi>y</mi><mo>≤</mo><mi>x</mi><mo>≤</mo><mn>1</mn></mrow></math></span>. Marginals <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>4</mn><msup><mi>x</mi><mn>3</mn></msup></mrow></math></span> and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>p</mi><mi>Y</mi></msub><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mn>4</mn><mi>y</mi><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><msup><mi>y</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></math></span>. Their product is not <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>8</mn><mi>x</mi><mi>y</mi></mrow></math></span>.

</div>

~~~dg-raw
%%viz f_plane x=-0.2,1.3 y=-0.2,1.3 tk=1 pg=0:0_1:0_1:1,a sg=0.3,0,0.3,0.3,f;0.8,0,0.8,0.8,f | On a triangle, the range of y depends on x (orange slices), so X and Y cannot be independent, whatever the formula.
~~~

<div class="take">
<div class="lab">Key takeaway</div>

independence needs the density to factorise everywhere, including on a rectangular support.

</div>

## Where this breaks in practice

- A scatter plot can look patternless while a nonlinear dependence hides in it. Plot binned conditionals as well.

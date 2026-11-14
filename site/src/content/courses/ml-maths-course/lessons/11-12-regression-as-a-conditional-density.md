---
id: "11.12"
course: "ml-maths-course"
module: "11"
slug: "regression-as-a-conditional-density"
title: "Regression as a Conditional Density"
keyIdeas: ["A supervised model describes ⟦3633⟧. Linear regression with Gaussian noise says ⟦3685⟧.", "Discriminative models skip ⟦3319⟧; generative models describe ⟦3319⟧ or ⟦3112⟧.", "A classifier cannot tell you whether an input is unusual, because it never modelled ⟦3319⟧."]
question: "A regression model predicts mean 3 and ⟦3684⟧ at some ⟦7⟧. What range covers about 95% of ⟦509⟧ there?"
answer: "Step 1, ⟦3687⟧. Step 2, from 2 to 4."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 7, 9"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: §6.4-6.6"}, {"title": "Bishop: Pattern Recognition and Machine Learning", "url": "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/", "note": "Read: §2.3"}]
tags: ["Two continuous variables"]
tools: ["NumPy"]
topic: {"id": "11.02", "title": "Two continuous variables"}
---

## The big picture

A regression line is not just a line: it is a family of distributions, one for each input.

## A model is a conditional density

~~~dg-flow A regression model is a conditional density.
Input x -(model)-> mean f(x) and noise scale σ -(Gaussian)-> *p(y given x) = N(f(x), σ²)
~~~

~~~dg-raw
%%viz f_plane x=-0.5,4.5 y=-0.5,5 tk=1 ln=0.8,-1,-0.5,mean_f(x)_=_0.8x_+_0.5,a,0.9 sg=1,0.3,1,2.3,f;3,1.9,3,3.9,f | At each x the model predicts a whole distribution for y. The orange bars show the ±2σ range (σ = 0.5) around the mean line.
~~~

```
rng = np.random.default_rng(0)
x = rng.uniform(0, 4, 100000)
y = 0.8 * x + 0.5 + rng.normal(0, 0.5, x.size)
near = np.abs(x - 3) < 0.02
print(round(y[near].mean(), 1), round(y[near].std(), 1))  # 2.9 0.5
```

<div class="take">
<div class="lab">Key takeaway</div>

supervised learning fits a conditional distribution of the output given the input.

</div>

## Where this breaks in practice

- A constant <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi></mrow></math></span> is an assumption. When noise grows with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>x</mi></mrow></math></span>, predict <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></math></span> as well.

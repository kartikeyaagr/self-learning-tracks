---
id: "09.45"
course: "ml-maths-course"
module: "09"
slug: "monte-carlo-integration"
title: "Monte Carlo Integration"
keyIdeas: ["Monte Carlo estimates ⟦3030⟧ with samples ⟦3031⟧.", "Its standard error is ⟦3032⟧ in any dimension, where ⟦1239⟧ is the standard deviation of ⟦3014⟧.", "Halving the error costs 4 times the samples; variance-reduction tricks attack ⟦1239⟧ instead."]
question: "With ⟦3028⟧ samples and ⟦3029⟧, what is the standard error? How many samples for a tenth of it?"
answer: "Step 1, ⟦3038⟧. Step 2, a tenth needs ⟦3039⟧ ten times larger, so ⟦3040⟧."
sources: [{"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 5, 7"}, {"title": "MIT 18.02SC Multivariable Calculus", "url": "https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/", "note": "Read: units 2-4"}, {"title": "Khan Academy: Multivariable calculus", "url": "https://www.khanacademy.org/math/multivariable-calculus", "note": "Read: all units"}, {"title": "Parr, Howard: The Matrix Calculus You Need for Deep Learning", "url": "https://explained.ai/matrix-calculus/", "note": "Read: whole article"}, {"title": "Boyd, Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Read: ch. 5"}]
tags: ["From partitions to volume"]
tools: ["NumPy"]
topic: {"id": "09.06", "title": "From partitions to volume"}
---

## The big picture

Grids fail in high dimensions. Random sampling does not: its accuracy depends on the number of samples, not on the dimension. Every training loss is a Monte Carlo estimate of the true expected loss.

## The estimator

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><msub><mover accent="true"><mi>μ</mi><mo>^</mo></mover><mi>N</mi></msub><mo>=</mo><mfrac><mn>1</mn><mi>N</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>g</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo stretchy="false">)</mo><mo separator="true">,</mo><mspace width="2em"/><mtext>standard error</mtext><mo>=</mo><mfrac><mi>σ</mi><msqrt><mi>N</mi></msqrt></mfrac></mrow></math></span></span>

```
rng = np.random.default_rng(0)
est = [(rng.normal(size=10000) ** 2).mean() for _ in range(200)]
print(round(np.std(est), 3))  # 0.014
```

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mo>=</mo><mn>1</mn></mrow></math></span> for a standard normal, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo stretchy="false">(</mo><msup><mi>X</mi><mn>2</mn></msup><mo stretchy="false">)</mo><mo>=</mo><msqrt><mn>2</mn></msqrt></mrow></math></span>. Across 200 repeats the estimates scatter with standard deviation <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.014</mn><mo>=</mo><msqrt><mn>2</mn></msqrt><mi mathvariant="normal">/</mi><mn>100</mn></mrow></math></span>.

~~~dg-raw
%%viz f_curve f=2/Math.sqrt(x) x=100,10000 y=0,0.2 tk=2000 pts=10000,N_=_10⁴:_0.02;2500,N_=_2500:_0.04 | Standard error σ/√N with σ = 2. Quadrupling the samples from 2,500 to 10,000 only halves the error.
~~~

~~~dg-flow Picking a method by dimension.
1 to 3 dimensions -(smooth integrand)-> *quadrature: scipy.integrate.quad, Simpson
Many dimensions -(can sample from p)-> *Monte Carlo: average g(x_i), error σ / √(N)
Many dimensions -(cannot sample p directly)-> importance sampling or MCMC
~~~

<div class="take">
<div class="lab">Key takeaway</div>

Monte Carlo averages samples; its error falls like <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>1</mn><mi mathvariant="normal">/</mi><msqrt><mi>N</mi></msqrt></mrow></math></span> whatever the dimension.

</div>

## Where this breaks in practice

- A heavy-tailed integrand can have infinite variance: the estimate looks stable for a long time, then jumps.

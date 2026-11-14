---
id: "10.15"
course: "ml-maths-course"
module: "10"
slug: "expected-value"
title: "Expected Value"
keyIdeas: ["⟦3396⟧ for discrete ⟦513⟧ and ⟦3397⟧ for continuous ⟦513⟧.", "The mean is the balance point of the distribution; it need not be a possible value.", "The sample average of many independent draws approaches ⟦3012⟧."]
question: "⟦3393⟧ with probability 0.1 and ⟦3394⟧ otherwise. Find ⟦3395⟧."
answer: "Step 1, ⟦3400⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
steps: [{"kind": "explain", "title": "A weighted average", "body": "A ticket pays 10 with probability 0.1 and 0 otherwise. What does it pay on average? Weight each value by its probability and add:\n\n$$\\mathbb{E}[X] = \\sum_x x\\,P(X = x) = 10 \\times 0.1 + 0 \\times 0.9 = 1$$\n\nThis is the **expected value**, or mean, of $X$. Training loss and accuracy are both means of this kind.", "question": {"q": "A bet wins 5 with probability 0.2 and loses 1 otherwise. What is its expected value?", "options": ["0.2", "2", "1.8"], "answer": 0, "why": "$5 \\times 0.2 + (-1) \\times 0.8 = 1 - 0.8 = 0.2$. Averaging 5 and $-1$ without weights would give 2.", "hint": "Weight each payoff by its probability, and keep the loss negative."}}, {"kind": "explain", "title": "The balance point", "body": "Picture the distribution as weights placed along a ruler: the mean is where the ruler balances. It doesn't have to be a value $X$ can take.\n\nA fair die puts weight $1/6$ on each of 1 to 6, so\n\n$$\\mathbb{E}[X] = \\frac{1 + 2 + 3 + 4 + 5 + 6}{6} = 3.5$$\n\nNo roll ever shows 3.5.", "question": {"q": "What is the expected value of a fair four-sided die showing 1 to 4?", "options": ["2", "2.5", "3"], "answer": 1, "why": "$(1 + 2 + 3 + 4)/4 = 2.5$, halfway between the faces, and again not a value you can roll.", "hint": "Each face has weight 1/4. Add up value times weight."}}, {"kind": "explain", "title": "Pulled by big values", "body": "In the ticket example, nine times out of ten you get 0, yet the mean is 1. A single large value, even a rare one, can pull the balance point a long way from where most of the outcomes sit.\n\nSo \"the mean\" and \"a typical outcome\" are not the same thing. For skewed distributions they can be far apart.", "question": {"q": "$X = 100$ with probability 0.01, else 0. Which is true?", "options": ["𝔼X = 50", "𝔼X = 0, usually X = 0", "𝔼X = 1, usually X = 0"], "answer": 2, "why": "$100 \\times 0.01 = 1$, while 99% of draws are 0. 50 would be the midpoint of the two values, ignoring the probabilities.", "hint": "Weight 100 by its probability 0.01."}}, {"kind": "explain", "title": "Continuous variables", "body": "For a continuous variable with density $p(x)$, the sum becomes an integral:\n\n$$\\mathbb{E}[X] = \\int x\\,p(x)\\,dx$$\n\nFor $X$ uniform on $[0, 2]$, $p(x) = 1/2$, so $\\mathbb{E}[X] = \\int_0^2 \\frac{x}{2}\\,dx = 1$, the middle of the interval. With an uneven density, the mean shifts towards where the density is high.", "question": {"q": "$X$ has density $p(x) = 2x$ on $[0, 1]$. What is $\\mathbb{E}[X]$?", "options": ["2/3", "1/2", "1"], "answer": 0, "why": "$\\int_0^1 x \\cdot 2x\\,dx = \\int_0^1 2x^2\\,dx = 2/3$. The density grows with $x$, so the mean sits right of the midpoint 1/2.", "hint": "Multiply $x$ by the density and integrate from 0 to 1."}}, {"kind": "explain", "title": "Averages settle", "body": "Draw many independent samples and their average approaches $\\mathbb{E}[X]$. This is the **law of large numbers**. A million simulated die rolls average 3.50 to two decimal places.\n\nThis is why a mean over a large batch, or a large test set, estimates the true expected loss or accuracy. A small sample can still land well away from the mean.", "question": {"q": "Ten die rolls average 4.1. What happens as you average a million rolls?", "options": ["The average stays near 4.1", "The average settles near 3.5", "The average keeps growing"], "answer": 1, "why": "With more draws the average converges to $\\mathbb{E}[X] = 3.5$. The 4.1 was sampling noise from a small sample, not a property of the die.", "hint": "What value does the law of large numbers say the average approaches?"}}, {"kind": "explain", "title": "When there is no mean", "body": "The law of large numbers needs the mean to exist. For some heavy-tailed distributions it doesn't: the integral never settles to a finite number.\n\nThe **Cauchy distribution** is the standard example. Its curve looks like a bell centred on 0, but its tails are so heavy that the running average of draws keeps getting knocked off course by huge values and never converges.", "question": {"q": "You average more and more Cauchy draws. What does the running average do?", "options": ["Keeps jumping, never settles", "Settles at 0", "Settles at 1"], "answer": 0, "why": "With no mean, there is nothing to converge to: a single huge draw can shift the average at any time. The curve's centre is 0, but the average doesn't settle there.", "hint": "The law of large numbers only applies when $\\mathbb{E}[X]$ exists."}}, {"kind": "question", "title": "Recap and a check", "body": "The expected value weights each value by its probability: a sum for discrete variables, an integral for continuous ones. It is the balance point, need not be a possible value, and sample averages approach it.", "question": {"q": "$X$ is 3 with probability 0.5, 1 with probability 0.3 and $-5$ with probability 0.2. What is $\\mathbb{E}[X]$?", "options": ["−1/3", "1.8", "0.8"], "answer": 2, "why": "$1.5 + 0.3 - 1.0 = 0.8$. The unweighted average of the three values would be $-1/3$.", "hint": "Weight each value by its probability, keeping the sign of $-5$."}}]
---

## The big picture

A distribution is a whole curve, but you usually want a few numbers that summarise it. The first is where it is centred. Training loss is a mean; so is accuracy.

## A weighted average

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mi>X</mi><mo stretchy="false">]</mo><mo>=</mo><munder><mo>∑</mo><mi>x</mi></munder><mi>x</mi><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo stretchy="false">)</mo><mspace width="1em"/><mtext>or</mtext><mspace width="1em"/><mo>∫</mo><mi>x</mi><mtext> </mtext><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>x</mi></mrow></math></span></span>
Each value is weighted by how likely it is. Picture the distribution as weights on a ruler: the mean is where the ruler balances.

~~~dg-raw
%%viz f_bars v=1,1,1,1,1,1 l=1,2,3,4,5,6 hi=none yl=chance_in_sixths | A fair die puts equal weight on 1 to 6. It balances at 3.5, a value you can never roll.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a skewed distribution**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>X</mi><mo>=</mo><mn>10</mn></mrow></math></span> with probability 0.1, else 0. Mean <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mo>=</mo><mn>1</mn></mrow></math></span>. Nine times out of ten you get 0, yet the mean is 1: one large value pulls it.

</div>

```
rng = np.random.default_rng(0)
x = rng.integers(1, 7, 1_000_000)
print(round(x.mean(), 2), sum(range(1, 7)) / 6)  # 3.5 3.5
```

<div class="take">
<div class="lab">Key takeaway</div>

the mean is the probability-weighted average, the balance point of the distribution.

</div>

## Where this breaks in practice

- Some distributions have no mean. The Cauchy distribution's sample averages never settle down.

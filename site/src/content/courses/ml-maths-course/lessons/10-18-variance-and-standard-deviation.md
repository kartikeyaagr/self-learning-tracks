---
id: "10.18"
course: "ml-maths-course"
module: "10"
slug: "variance-and-standard-deviation"
title: "Variance and Standard Deviation"
keyIdeas: ["⟦3420⟧.", "The standard deviation ⟦3421⟧ is in the same units as ⟦513⟧.", "Two distributions with the same mean can have very different spreads."]
question: "⟦3393⟧ with probability 0.1, else 0. Find the variance and standard deviation."
answer: "Step 1, ⟦3425⟧ and ⟦3426⟧. Step 2, ⟦3427⟧, so ⟦3428⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Moments, means and variability"]
tools: ["NumPy"]
topic: {"id": "10.04", "title": "Moments, means and variability"}
steps: [{"kind": "explain", "title": "Same mean, different spread", "body": "$Y$ is always 1. $X$ is 0 or 2, each with probability 0.5. Both have mean 1, but $Y$ never moves and $X$ is always 1 away from its mean.\n\nThe mean tells you where a distribution is centred, not how spread out it is. The noise in a minibatch gradient and the uncertainty in an estimate are both questions of spread.", "question": {"q": "Which is true of $X$ and $Y$?", "options": ["Same mean, X more spread", "Different means", "Same mean and same spread"], "answer": 0, "why": "$\\mathbb{E}[X] = 0 \\times 0.5 + 2 \\times 0.5 = 1 = \\mathbb{E}[Y]$, but $X$ always lands 1 away from 1 while $Y$ never moves.", "hint": "Compute both means first, then compare how far each lands from its mean."}}, {"kind": "explain", "title": "Variance", "body": "**Variance** is the average squared distance from the mean $\\mu$:\n\n$$\\text{Var}\\,X = \\mathbb{E}[(X - \\mu)^2]$$\n\nFor $X = 0$ or $2$ with equal chance, $\\mu = 1$, and both values are 1 away: $\\text{Var}\\,X = \\frac{1^2 + 1^2}{2} = 1$. For $Y$, always 1, the variance is 0. Squaring keeps every distance positive and weights big misses more.", "question": {"q": "$X$ is 0 or 4 with equal chance. What is $\\text{Var}\\,X$?", "options": ["2", "4", "8"], "answer": 1, "why": "$\\mu = 2$, and both values are 2 away, so the average squared distance is $2^2 = 4$. 2 is the average distance without squaring, and 8 is $\\mathbb{E}[X^2]$.", "hint": "Find the mean, then square each value's distance from it before averaging."}}, {"kind": "explain", "title": "A shortcut", "body": "Expanding the square gives an easier formula:\n\n$$\\text{Var}\\,X = \\mathbb{E}[X^2] - \\mu^2$$\n\nFor the ticket that pays 10 with probability 0.1: $\\mathbb{E}[X^2] = 100 \\times 0.1 = 10$ and $\\mu = 1$, so $\\text{Var}\\,X = 10 - 1 = 9$.", "question": {"q": "For a fair die, $\\mathbb{E}[X^2] = 91/6 \\approx 15.17$ and $\\mu = 3.5$. What is the variance?", "options": ["11.67", "15.17", "2.92"], "answer": 2, "why": "$15.17 - 3.5^2 = 15.17 - 12.25 \\approx 2.92$. Subtracting $\\mu$ instead of $\\mu^2$ gives 11.67.", "hint": "Subtract the square of the mean, not the mean itself."}}, {"kind": "explain", "title": "Standard deviation", "body": "Variance is in squared units: metres squared, or pounds squared. The **standard deviation** $\\sigma = \\sqrt{\\text{Var}\\,X}$ brings it back to the units of $X$.\n\nThe ticket has $\\sigma = \\sqrt{9} = 3$ and the die has $\\sigma = \\sqrt{2.92} \\approx 1.71$. A bell curve with $\\sigma = 2$ is twice as wide as one with $\\sigma = 1$, and half as tall.", "question": {"q": "Heights have variance 49 cm². What is their standard deviation?", "options": ["7 cm", "49 cm", "24.5 cm"], "answer": 0, "why": "$\\sigma = \\sqrt{49} = 7$ cm, in the same units as the heights themselves.", "hint": "Undo the squaring."}}, {"kind": "explain", "title": "n or n − 1", "body": "For data, the variance divides the sum of squared deviations by a count, and libraries disagree about which count. `np.var` divides by $n$. pandas' `.var()` divides by $n - 1$, the **sample variance**, which corrects for measuring deviations from the sample's own mean.\n\nFor $[1, 3]$, the mean is 2 and the squared deviations sum to 2. For small samples, the difference matters.", "question": {"q": "For the data $[1, 3]$, `np.var` gives 1. What does pandas' `.var()` give?", "options": ["1", "2", "0.5"], "answer": 1, "why": "$2 / (2 - 1) = 2$, against $2 / 2 = 1$ for NumPy. With large $n$ the two converge.", "hint": "Divide the same sum of squared deviations by $n - 1$."}}, {"kind": "question", "title": "Recap and a check", "body": "Variance is the mean squared distance from the mean, $\\mathbb{E}[X^2] - \\mu^2$. Its square root, the standard deviation, is in the units of $X$. Equal means can hide very different spreads.", "question": {"q": "$X$ is $-1$ or $1$ with equal chance; $Y$ is $-3$ or $3$ with equal chance. Which is true?", "options": ["Same mean; σ is 1 and 9", "Different means", "Same mean; σ is 1 and 3"], "answer": 2, "why": "Both means are 0. $\\text{Var}\\,X = 1$ and $\\text{Var}\\,Y = 9$, so the standard deviations are 1 and 3. 9 is $Y$'s variance, not its standard deviation.", "hint": "Find each variance, then take square roots."}}]
---

## The big picture

The second summary number is spread. The noise in a minibatch gradient, the uncertainty in an estimate and the risk of a strategy are all variances.

## The definition

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><mo stretchy="false">(</mo><mi>X</mi><mo>−</mo><mi>μ</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup><mo stretchy="false">]</mo><mo>=</mo><mi mathvariant="normal">𝔼</mi><mo stretchy="false">[</mo><msup><mi>X</mi><mn>2</mn></msup><mo stretchy="false">]</mo><mo>−</mo><mo stretchy="false">(</mo><mi mathvariant="normal">𝔼</mi><mi>X</mi><msup><mo stretchy="false">)</mo><mn>2</mn></msup></mrow></math></span></span>
It is the average squared distance from the mean.

~~~dg-raw
%%viz f_curve f=Math.exp(-x*x/2)/Math.sqrt(2*Math.PI) g=Math.exp(-x*x/8)/(2*Math.sqrt(2*Math.PI)) x=-6,6 y=0,0.45 tk=2 fl=σ_=_1 gl=σ_=_2 | Two distributions with mean 0. Doubling σ makes the curve twice as wide and half as tall.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the die's spread**

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">Var</mi><mo>⁡</mo><mi>X</mi><mo>=</mo><mn>91</mn><mi mathvariant="normal">/</mi><mn>6</mn><mo>−</mo><mn>3.</mn><msup><mn>5</mn><mn>2</mn></msup><mo>≈</mo><mn>2.92</mn></mrow></math></span>, so <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi><mo>≈</mo><mn>1.71</mn></mrow></math></span>.

</div>

```
x = np.arange(1, 7)
print(round(x.var(), 2), round(x.std(), 2))  # 2.92 1.71
```

<div class="take">
<div class="lab">Key takeaway</div>

variance is the mean squared distance from the mean; its square root is in the original units.

</div>

## Where this breaks in practice

- `np.var` divides by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi></mrow></math></span> by default; `pandas` divides by <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow></math></span>. For small samples the difference matters (12.04).

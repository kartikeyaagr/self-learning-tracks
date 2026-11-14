---
id: "10.37"
course: "ml-maths-course"
module: "10"
slug: "the-normal-distribution"
title: "The Normal Distribution"
keyIdeas: ["⟦3565⟧ has density ⟦3566⟧.", "About 68, 95 and 99.7% of the probability lies within 1, 2 and 3 standard deviations.", "Sums of many small independent effects are approximately normal (the central limit theorem, lesson 12.06)."]
question: "Test scores are ⟦3564⟧. What fraction lies between 50 and 90?"
answer: "Step 1, 50 and 90 are 2 standard deviations from the mean. Step 2, about 95%."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Continuous probability distributions"]
tools: ["SciPy"]
topic: {"id": "10.06", "title": "Continuous probability distributions"}
steps: [{"kind": "explain", "title": "The bell curve", "body": "Test scores, measurement noise and initial weights are often modelled by the **normal** (Gaussian) distribution $\\mathcal{N}(\\mu, \\sigma^2)$. It has density\n\n$$p(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)$$\n\n$\\mu$ is the centre and $\\sigma$ the standard deviation. Note the convention: the second number in $\\mathcal{N}(\\mu, \\sigma^2)$ is the variance.", "question": {"q": "Scores follow $\\mathcal{N}(70, 10^2)$. What is their standard deviation?", "options": ["10", "100", "70"], "answer": 0, "why": "The second argument is $\\sigma^2 = 10^2 = 100$, so $\\sigma = 10$. 70 is the mean.", "hint": "$\\mathcal{N}(\\mu, \\sigma^2)$ is written with the variance."}}, {"kind": "explain", "title": "Reading the formula", "body": "The exponent $-(x - \\mu)^2 / 2\\sigma^2$ makes the density fall with the **square** of the distance from $\\mu$, measured in units of $\\sigma$. At one $\\sigma$ from the centre the exponent is $-1/2$; at $k$ standard deviations it is $-k^2/2$.\n\nThe factor $\\frac{1}{\\sigma\\sqrt{2\\pi}}$ in front is what makes the total area exactly 1. A wider curve, with a bigger $\\sigma$, has to be lower.", "question": {"q": "How much lower is the density at $\\mu + 2\\sigma$ than at $\\mu$?", "options": ["Half as high", "e⁻¹ ≈ 0.37 times", "e⁻² ≈ 0.14 times"], "answer": 2, "why": "At $k = 2$ the exponent is $-2^2/2 = -2$, so the density is $e^{-2} \\approx 0.14$ times its peak. The square makes it fall faster than linearly.", "hint": "Put $x - \\mu = 2\\sigma$ into the exponent."}}, {"kind": "explain", "title": "68, 95, 99.7", "body": "Three numbers cover most quick calculations. For any normal distribution, about\n\n- 68% of the probability lies within 1 standard deviation of $\\mu$,\n- 95% within 2,\n- 99.7% within 3.\n\nThey hold whatever $\\mu$ and $\\sigma$ are, because only the distance in units of $\\sigma$ matters.", "question": {"q": "Scores follow $\\mathcal{N}(70, 10^2)$. What fraction lies between 50 and 90?", "options": ["68%", "95%", "99.7%"], "answer": 1, "why": "50 and 90 are each 20 points, or 2 standard deviations, from 70, so about 95% lies between them.", "hint": "Convert 50 and 90 into standard deviations from the mean."}}, {"kind": "explain", "title": "The tails", "body": "The curve is symmetric, so whatever lies outside a band splits equally between the two tails. Outside $\\pm 1\\sigma$ is 32%, so 16% is above $\\mu + \\sigma$. Outside $\\pm 2\\sigma$ is 5%, so 2.5% is above $\\mu + 2\\sigma$.\n\nTail fractions are what you usually want in practice: how often noise exceeds a threshold, or how rare an observation is.", "question": {"q": "With scores $\\mathcal{N}(70, 10^2)$, what fraction scores above 90?", "options": ["2.5%", "5%", "16%"], "answer": 0, "why": "90 is $2\\sigma$ above the mean. 5% lies outside $\\pm 2\\sigma$, split equally between the two tails, so 2.5% is above 90.", "hint": "Only one of the two tails counts here."}}, {"kind": "explain", "title": "Why it is everywhere", "body": "Sums of many small, independent effects come out approximately normal, whatever the distribution of each effect. This is the **central limit theorem**. A height or a measurement error is the sum of many small influences, so it looks Gaussian.\n\nTwo more reasons: among all distributions with a given mean and variance, the normal assumes the least (maximum entropy), and linear maps of Gaussians stay Gaussian.", "question": {"q": "Which is closest to a normal distribution?", "options": ["One die roll", "Heads in 2 coin flips", "The total of 100 dice"], "answer": 2, "why": "A total of 100 independent dice is a sum of many small effects, so the central limit theorem applies. One die is flat, and 2 flips give only 3 values.", "hint": "The central limit theorem needs many independent pieces added together."}}, {"kind": "explain", "title": "Variance or σ in code", "body": "Maths writes $\\mathcal{N}(\\mu, \\sigma^2)$ with the variance. Code usually takes the standard deviation: `np.random.normal(loc, scale)` and `torch.normal(mean, std)` both expect $\\sigma$.\n\nPass the variance where $\\sigma$ belongs and your noise ends up with the wrong spread. For $\\mathcal{N}(0, 0.01)$, the standard deviation is $\\sqrt{0.01} = 0.1$, so the call is `np.random.normal(0, 0.1)`.", "question": {"q": "You want noise from $\\mathcal{N}(0, 0.04)$. Which call is right?", "options": ["np.random.normal(0, 0.04)", "np.random.normal(0, 0.2)", "np.random.normal(0, 0.0016)"], "answer": 1, "why": "$\\sigma = \\sqrt{0.04} = 0.2$. Passing 0.04 as the scale would give a variance of 0.0016, 25 times too small.", "hint": "The `scale` argument is the standard deviation. Take the square root of the variance."}}, {"kind": "question", "title": "Recap and a check", "body": "$\\mathcal{N}(\\mu, \\sigma^2)$ is set by its mean and variance; about 68, 95 and 99.7% lies within 1, 2 and 3 standard deviations. Sums of many small effects are close to normal.", "question": {"q": "Weights are drawn from $\\mathcal{N}(0, 0.01)$. Roughly what fraction lie outside $(-0.3, 0.3)$?", "options": ["5%", "≈ 0%", "0.3%"], "answer": 2, "why": "$\\sigma = \\sqrt{0.01} = 0.1$, so $\\pm 0.3$ is $\\pm 3\\sigma$, which holds 99.7%. That leaves 0.3% outside.", "hint": "Find $\\sigma$ from the variance first, then express 0.3 in standard deviations."}}]
---

## The big picture

The Gaussian is everywhere: measurement noise, weight initialisation, diffusion models, the latent space of a VAE. It is the one distribution to know by heart.

## Reading the formula

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span> is the centre, <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>σ</mi></mrow></math></span> the spread. The exponent makes the density fall off with the square of the distance from <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>μ</mi></mrow></math></span>, and <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mfrac><mn>1</mn><mrow><mi>σ</mi><msqrt><mrow><mn>2</mn><mi>π</mi></mrow></msqrt></mrow></mfrac></mrow></math></span> makes the area 1 (lesson 09.51).

~~~dg-raw
%%viz f_curve f=Math.exp(-(x-70)*(x-70)/200)/(10*Math.sqrt(2*Math.PI)) x=30,110 y=0,0.045 tk=10 dom=50,90,cc | N(70, 10²). The shaded band from 50 to 90, two standard deviations each side, holds about 95% of the probability.
~~~

```
from scipy import stats
print([float(round(stats.norm.cdf(k) - stats.norm.cdf(-k), 4)) for k in (1, 2, 3)])  # [0.6827, 0.9545, 0.9973]
```

## Why it is everywhere

- Sums of many small independent effects are approximately Gaussian.
- Among distributions with a given mean and variance, it assumes the least (maximum entropy).
- Linear maps and conditionals of Gaussians are Gaussian (module 11).

<div class="take">
<div class="lab">Key takeaway</div>

the normal is set by its mean and standard deviation; 68-95-99.7 gives quick probabilities.

</div>

## Where this breaks in practice

- `np.random.normal(loc, scale)` and `torch.normal(mean, std)` take the standard deviation, while <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">𝒩</mi><mo stretchy="false">(</mo><mi>μ</mi><mo separator="true">,</mo><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></math></span> is written with the variance. Mixing them up squares your noise level.

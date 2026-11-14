---
id: "10.02"
course: "ml-maths-course"
module: "10"
slug: "bayes-theorem"
title: "Bayes' Theorem"
keyIdeas: ["Bayes: ⟦3235⟧. The posterior is likelihood times prior, rescaled to add up to 1.", "The denominator comes from total probability: ⟦3236⟧.", "When the event is rare, the base rate dominates: a very accurate test can still be wrong most of the time."]
question: "A screening test has sensitivity 99% and specificity 95%. The condition affects 1% of the people tested. A person tests positive: what is the probability they have the condition?"
answer: "Step 1, ⟦3250⟧, ⟦3251⟧, ⟦3252⟧. Step 2, ⟦3253⟧. Step 3, ⟦3254⟧. Only about one positive in six is real."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Probability rules and Bayesian updating"]
tools: ["NumPy"]
topic: {"id": "10.01", "title": "Probability rules and Bayesian updating"}
steps: [{"kind": "explain", "title": "The wrong way round", "body": "A screening test catches 99% of people who have a condition $D$. That is $P(+ \\mid D) = 0.99$, called its **sensitivity**.\n\nBut a patient who tests positive wants the reverse: $P(D \\mid +)$, the chance they have the condition given the positive test. These are different numbers, and they can be very far apart. Bayes' theorem turns one into the other.", "question": {"q": "Which quantity does the 99% sensitivity describe?", "options": ["P(+ | D)", "P(D | +)", "P(D)"], "answer": 0, "why": "Sensitivity is the chance of a positive test among people who have the condition. The patient's question, $P(D \\mid +)$, needs more information.", "hint": "Sensitivity starts from people who have $D$ and asks how many of them test positive."}}, {"kind": "explain", "title": "Count the people", "body": "Add two facts: 1% of the people tested have the condition, and the test wrongly flags 5% of healthy people. Picture 10,000 people:\n\n- 100 are sick, and 99 of them test positive;\n- 9,900 are healthy, and 5% of them, 495, test positive anyway.\n\nSo there are $99 + 495 = 594$ positive tests in total.", "question": {"q": "What fraction of the positive tests are real?", "options": ["99%", "≈ 17%", "1%"], "answer": 1, "why": "$99 / 594 \\approx 0.17$. Healthy people outnumber sick ones 99 to 1, so their 5% false-positive rate produces most of the positives.", "hint": "Divide the sick positives by all the positives."}}, {"kind": "explain", "title": "The formula", "body": "Bayes' theorem does that counting with probabilities:\n\n$$P(H \\mid D) = \\frac{P(D \\mid H)\\,P(H)}{P(D)}$$\n\n- $P(H)$ is the **prior**: how plausible the hypothesis was before the data.\n- $P(D \\mid H)$ is the **likelihood**: how well the hypothesis predicts the data.\n- $P(H \\mid D)$ is the **posterior**: how plausible it is after the data.\n\nIn short, the posterior is likelihood times prior, rescaled so it adds up to 1.", "question": {"q": "In the screening example, what is the prior $P(D)$?", "options": ["0.01", "0.99", "0.167"], "answer": 0, "why": "The prior is the base rate before any test result: 1% of people have the condition. 0.99 is the likelihood and 0.167 the posterior.", "hint": "The prior describes your belief before you see the test result."}}, {"kind": "explain", "title": "The denominator", "body": "$P(D)$ in the formula is the overall chance of the data, built with total probability over every hypothesis:\n\n$$P(+) = P(+ \\mid D)P(D) + P(+ \\mid \\neg D)P(\\neg D)$$\n\nFor the test: $0.99 \\times 0.01 + 0.05 \\times 0.99 = 0.0594$. Then $P(D \\mid +) = 0.0099 / 0.0594 \\approx 0.167$, matching the count of people.", "question": {"q": "Sensitivity 90%, false-positive rate 10%, prevalence 10%. What is $P(+)$?", "options": ["0.09", "0.10", "0.18"], "answer": 2, "why": "$0.9 \\times 0.1 + 0.1 \\times 0.9 = 0.18$. Half of those positives are real, so $P(D \\mid +) = 0.09 / 0.18 = 0.5$.", "hint": "Add the positives from sick people and the positives from healthy people."}}, {"kind": "explain", "title": "Base rates dominate", "body": "When the condition is rare, the prior dominates: even a very accurate test is wrong about most positives. Mistaking $P(D \\mid H)$ for $P(H \\mid D)$ is the **base-rate fallacy**.\n\nThe same arithmetic hits classifiers. With 1% positives, 99% recall and a 5% false-positive rate, only about 17% of flagged examples are true positives. On imbalanced data, report precision and recall, not just accuracy.", "question": {"q": "Same test (99% sensitivity, 5% false positives), but half the people tested have the condition. What is $P(D \\mid +)$?", "options": ["≈ 95%", "≈ 17%", "99%"], "answer": 0, "why": "$\\frac{0.99 \\times 0.5}{0.99 \\times 0.5 + 0.05 \\times 0.5} \\approx 0.95$. The test didn't change, but with a high prior the positives are mostly real.", "hint": "Redo the formula with $P(D) = 0.5$. Only the prior changed."}}, {"kind": "explain", "title": "Updating again", "body": "Today's posterior is tomorrow's prior. Suppose a second test, independent of the first given the true state, also comes back positive. Start from $P(D) = 0.167$:\n\n$$P(D \\mid ++) = \\frac{0.99 \\times 0.167}{0.99 \\times 0.167 + 0.05 \\times 0.833} \\approx 0.80$$\n\nOne positive gives 17%; two give 80%.", "question": {"q": "A third independent test is also positive. Roughly what is the posterior now?", "options": ["≈ 80%", "≈ 99%", "100%"], "answer": 1, "why": "Starting from 0.80: $\\frac{0.99 \\times 0.80}{0.99 \\times 0.80 + 0.05 \\times 0.20} \\approx 0.99$. It gets close to certain but never reaches it.", "hint": "Use 0.80 as the prior and apply the same update once more."}}, {"kind": "explain", "title": "Naive Bayes", "body": "A spam filter applies Bayes with several clues. **Naive Bayes** assumes the words are independent given the class, so their likelihoods multiply.\n\nPrior $P(\\text{spam}) = 0.4$. \"free\" appears in 30% of spam and 2% of ham; \"meeting\" in 1% of spam and 10% of ham. For an email with both words, the spam score is $0.4 \\times 0.3 \\times 0.01 = 0.0012$ and the ham score is $0.6 \\times 0.02 \\times 0.1 = 0.0012$. Rescaled, $P(\\text{spam}) = 0.5$.", "question": {"q": "An email contains only “free”. What is $P(\\text{spam})$?", "options": ["30%", "≈ 94%", "≈ 91%"], "answer": 2, "why": "Spam score $0.4 \\times 0.3 = 0.12$; ham score $0.6 \\times 0.02 = 0.012$; so $0.12 / 0.132 \\approx 0.91$. Leaving out the prior gives 94%.", "hint": "Multiply each class's prior by its likelihood for “free”, then rescale the two scores to add up to 1."}}, {"kind": "question", "title": "Recap and a check", "body": "The posterior is likelihood times prior, divided by $P(D)$ from total probability. When the event is rare, the base rate dominates.", "question": {"q": "A fraud model flags 95% of fraud and 2% of honest transactions. Fraud is 0.1% of transactions. Roughly what share of flagged transactions are fraud?", "options": ["≈ 95%", "≈ 5%", "≈ 50%"], "answer": 1, "why": "$\\frac{0.95 \\times 0.001}{0.95 \\times 0.001 + 0.02 \\times 0.999} \\approx 0.045$. Honest transactions are so common that their 2% false flags swamp the real fraud.", "hint": "Compare the number of flags from fraud with the number from honest transactions, out of, say, a million."}}]
---

## The big picture

Bayes' theorem flips a conditional probability around. You know how likely the evidence is under each explanation; you want how likely each explanation is given the evidence. It explains why a "99% accurate" test can mislead, and it is how a spam filter or a Bayesian model updates.

## The formula

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span></span>

- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow></math></span>, the **prior**: how plausible the hypothesis was before the data.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo></mrow></math></span>, the **likelihood**: how well the hypothesis predicts the data.
- <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo></mrow></math></span>, the **posterior**: how plausible it is after the data.

The denominator is built by total probability over all hypotheses:
<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo></mrow><mrow><munder><mo>∑</mo><msup><mi>H</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup></munder><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><msup><mi>H</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">)</mo><mi>P</mi><mo stretchy="false">(</mo><msup><mi>H</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo stretchy="false">)</mo></mrow></mfrac></mrow></math></span></span>

## Counting people

Out of 10,000 people, 100 are sick and 99 of them test positive. 9,900 are healthy and 495 of them test positive anyway. So <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>99</mn><mi mathvariant="normal">/</mi><mo stretchy="false">(</mo><mn>99</mn><mo>+</mo><mn>495</mn><mo stretchy="false">)</mo><mo>≈</mo><mn>17</mn><mi mathvariant="normal">%</mi></mrow></math></span> of positives are real.

~~~dg-raw
%%viz f_bars v=99,495 l=sick_and_positive,healthy_and_positive hi=none yl=people_out_of_10~000 | Positive tests among 10,000 people. Healthy people far outnumber sick ones, so most positives come from the 5% false-positive rate.
~~~

```
sens, spec, prev = 0.99, 0.95, 0.01
p_pos = sens * prev + (1 - spec) * (1 - prev)
print(round(p_pos, 4), round(sens * prev / p_pos, 3))  # 0.0594 0.167
rng = np.random.default_rng(0)
sick = rng.random(1_000_000) < prev
pos = np.where(sick, rng.random(sick.size) < sens, rng.random(sick.size) < 1 - spec)
print(round(sick[pos].mean(), 2))  # 0.17
```

## Updating twice

Today's posterior is tomorrow's prior.

~~~dg-flow Bayesian updating.
Prior P(H) -(observe D_1, multiply by likelihood)-> posterior -(observe D_2)-> *updated posterior
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two positive tests**

Assume the two tests are independent given the true state. New prior: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.167</mn></mrow></math></span>.

<span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mo>+</mo><mo stretchy="false">)</mo><mo>=</mo><mn>0.99</mn><mo>×</mo><mn>0.167</mn><mo>+</mo><mn>0.05</mn><mo>×</mo><mn>0.833</mn><mo>≈</mo><mn>0.207</mn></mrow></math></span>. Posterior: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.99</mn><mo>×</mo><mn>0.167</mn><mi mathvariant="normal">/</mi><mn>0.207</mn><mo>≈</mo><mn>0.80</mn></mrow></math></span>.

One positive: 17%. Two positives: 80%.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**naive Bayes for spam**

Prior <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>spam</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.4</mn></mrow></math></span>. "free" appears in 30% of spam and 2% of ham; "meeting" in 1% of spam and 10% of ham.

An email contains both words. Spam score: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.4</mn><mo>×</mo><mn>0.3</mn><mo>×</mo><mn>0.01</mn><mo>=</mo><mn>0.0012</mn></mrow></math></span>. Ham score: <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mn>0.6</mn><mo>×</mo><mn>0.02</mn><mo>×</mo><mn>0.1</mn><mo>=</mo><mn>0.0012</mn></mrow></math></span>.

Posterior <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mtext>spam</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></mrow></math></span>. The two words cancel out.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

posterior ∝ likelihood × prior; base rates matter most when the event is rare.

</div>

## Where this breaks in practice

- Confusing <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>D</mi><mo>∣</mo><mi>H</mi><mo stretchy="false">)</mo></mrow></math></span> with <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>∣</mo><mi>D</mi><mo stretchy="false">)</mo></mrow></math></span> is the **base-rate fallacy**.
- A classifier with high recall and a 5% false-positive rate has precision near 17% when positives are 1% of the data. On imbalanced problems report precision and recall, not accuracy.

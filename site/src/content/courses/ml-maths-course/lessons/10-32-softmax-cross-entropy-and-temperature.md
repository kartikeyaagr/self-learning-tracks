---
id: "10.32"
course: "ml-maths-course"
module: "10"
slug: "softmax-cross-entropy-and-temperature"
title: "Softmax, Cross-Entropy and Temperature"
keyIdeas: ["Cross-entropy ⟦3527⟧ is the negative log-likelihood of a categorical model.", "Training a classifier with cross-entropy is maximum likelihood (lesson 12.11).", "Temperature: ⟦3528⟧ sharpens for ⟦3529⟧ and flattens for ⟦3530⟧."]
question: "Logits ⟦3526⟧, true class 0. Compute the cross-entropy."
answer: "Step 1, ⟦3533⟧. Step 2, ⟦3534⟧. Step 3, loss ⟦3535⟧."
sources: [{"title": "Blitzstein, Hwang: Introduction to Probability (free PDF)", "url": "https://stat110.hsites.harvard.edu/", "note": "Read: ch. 1-6, 8"}, {"title": "Stat 110 lectures (Harvard, YouTube)", "url": "https://stat110.hsites.harvard.edu/youtube", "note": "Read: lectures 1-22"}, {"title": "Deisenroth, Faisal, Ong: Mathematics for Machine Learning", "url": "https://mml-book.github.io/", "note": "Read: ch. 6"}]
tags: ["Discrete probability distributions"]
tools: ["NumPy"]
topic: {"id": "10.05", "title": "Discrete probability distributions"}
steps: [{"kind": "explain", "title": "Scores to probabilities", "body": "A classifier outputs one **logit** per class, a score that can be any number. Softmax turns them into probabilities:\n\n$$\\pi_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$$\n\nFor $z = (2, 1, 0)$: $e^2 \\approx 7.39$, $e^1 \\approx 2.72$ and $e^0 = 1$, which sum to 11.11. So $\\pi \\approx (0.665, 0.245, 0.090)$: every entry positive, summing to 1, largest where the logit is largest.", "question": {"q": "What does softmax give for logits $(0, 0, 0)$?", "options": ["(1/3, 1/3, 1/3)", "(0, 0, 0)", "(1, 1, 1)"], "answer": 0, "why": "Each $e^0 = 1$, so every class gets $1/3$. Equal logits give equal probabilities, whatever their value.", "hint": "Compute $e^0$ for each class, then divide by the total."}}, {"kind": "explain", "title": "Cross-entropy", "body": "To train, compare $\\pi$ with the true class $y$. The **cross-entropy** loss is\n\n$$\\ell = -\\log \\pi_y$$\n\nIt is near 0 when the model puts almost all its probability on the true class, and grows without limit as $\\pi_y$ falls towards 0. For $z = (2, 1, 0)$ with true class 0, $\\ell = -\\log 0.665 \\approx 0.41$.", "question": {"q": "Same logits, but the true class is 2. What is the loss?", "options": ["0.09", "2.41", "0.41"], "answer": 1, "why": "$\\pi_2 \\approx 0.090$, and $-\\log 0.090 \\approx 2.41$. The model gave little probability to the right answer, so the loss is large.", "hint": "Take minus the log of the probability on class 2, not the probability itself."}}, {"kind": "explain", "title": "Maximum likelihood", "body": "Why minus the log? Treat the model as a categorical distribution over classes. Then $\\pi_y$ is the likelihood of the observed label, and $-\\log \\pi_y$ is its **negative log-likelihood**.\n\nOver a dataset, the losses add, and the log turns a product of likelihoods into a sum. So minimising total cross-entropy is the same as maximising the probability the model gives to all the true labels together: **maximum likelihood**.", "question": {"q": "A model gives the true labels of 3 examples probabilities 0.5, 0.25 and 0.8. What is the total cross-entropy?", "options": ["≈ 2.30", "≈ 0.10", "≈ 1.55"], "answer": 0, "why": "$0.69 + 1.39 + 0.22 \\approx 2.30$, which is $-\\log(0.5 \\times 0.25 \\times 0.8) = -\\log 0.1$. 0.10 is the likelihood itself, before the log.", "hint": "Add $-\\log$ of each probability, or take $-\\log$ of their product."}}, {"kind": "explain", "title": "The logit form", "body": "Substituting softmax into $-\\log \\pi_y$ gives a form that works on the logits directly:\n\n$$\\ell = -z_y + \\log \\sum_k e^{z_k}$$\n\nFor $z = (2, 1, 0)$ and $y = 0$: $-2 + \\log 11.11 \\approx -2 + 2.41 = 0.41$. Libraries compute the log-sum-exp after subtracting the largest logit, which changes nothing in the result but keeps $e^{z}$ from overflowing.", "question": {"q": "With a stable implementation, what is the loss for logits $(1000, 999, 998)$ and true class 0?", "options": ["≈ 1000", "≈ 2.41", "≈ 0.41"], "answer": 2, "why": "Only the gaps between logits matter, and they are the same as for $(2, 1, 0)$, so the loss is again about 0.41. A naive $e^{1000}$ would overflow.", "hint": "Subtract the largest logit from every logit first. What logits are left?"}}, {"kind": "explain", "title": "Temperature", "body": "When sampling from a model, you often divide the logits by a **temperature** $T$ first: $\\text{softmax}(z / T)$.\n\nFor $z = (2, 1, 0)$:\n\n| $T$ | probabilities |\n| --- | --- |\n| 0.5 | 0.87, 0.12, 0.02 |\n| 1 | 0.67, 0.24, 0.09 |\n| 2 | ? |\n\n$T < 1$ widens the gaps between logits and sharpens the distribution. $T > 1$ narrows them and flattens it.", "question": {"q": "At $T = 2$, roughly what probability does the top class get?", "options": ["0.87", "0.51", "0.67"], "answer": 1, "why": "$\\text{softmax}(1, 0.5, 0) \\approx (0.51, 0.31, 0.19)$. The order of the classes stays the same; only the gaps shrink.", "hint": "Dividing by 2 halves every gap between the logits. Does that sharpen or flatten?"}}, {"kind": "interact", "title": "Temperature in action", "body": "This widget runs the same softmax on real scores: how strongly the word **it** matches each word of a sentence. Each bar is a probability from $\\text{softmax}(s / T)$. Drag the temperature slider from 0.25 up to 3 and watch how the bars change.", "widget": "softmax-temperature", "question": {"q": "As you push $T$ towards its largest value, what do the bars approach?", "options": ["One bar taking everything", "Bars proportional to the scores", "All bars equal"], "answer": 2, "why": "A large $T$ shrinks every gap between scores towards 0, so every probability drifts towards $1/n$. At small $T$ the top word takes nearly everything.", "hint": "Dividing by a large $T$ makes all the scores close to each other. What does softmax do with nearly equal inputs?"}}, {"kind": "question", "title": "Recap and a check", "body": "Softmax turns logits into probabilities. Cross-entropy, $-\\log \\pi_y$, is the negative log-likelihood, so training with it is maximum likelihood. Temperature sharpens ($T < 1$) or flattens ($T > 1$) the distribution.", "question": {"q": "Logits $(2, 1, 0)$, true class 0. You compute the loss at $T = 2$ instead of $T = 1$. What happens?", "options": ["It falls to about 0.14", "It rises to about 0.68", "It stays at 0.41"], "answer": 1, "why": "At $T = 2$, $\\pi_0$ drops from 0.665 to about 0.51, and $-\\log 0.51 \\approx 0.68$. Flattening moves probability away from the top class, which here is the true one.", "hint": "Find $\\pi_0$ at $T = 2$ from the earlier step, then take minus its log."}}]
---

## The big picture

This is the loss used to train nearly every classifier and language model. Its probability meaning explains its behaviour.

## Cross-entropy is a log-likelihood

<span class="katex-display"><span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mo>−</mo><mi>log</mi><mo>⁡</mo><mi>P</mi><mo stretchy="false">(</mo><mi>Y</mi><mo>=</mo><mi>y</mi><mo stretchy="false">)</mo><mo>=</mo><mo>−</mo><mi>log</mi><mo>⁡</mo><msub><mi>π</mi><mi>y</mi></msub><mo>=</mo><mo>−</mo><msub><mi>z</mi><mi>y</mi></msub><mo>+</mo><mi>log</mi><mo>⁡</mo><munder><mo>∑</mo><mi>k</mi></munder><msup><mi>e</mi><msub><mi>z</mi><mi>k</mi></msub></msup></mrow></math></span></span>

## Temperature

Sampling uses <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi mathvariant="normal">softmax</mi><mo>⁡</mo><mo stretchy="false">(</mo><mi>z</mi><mi mathvariant="normal">/</mi><mi>T</mi><mo stretchy="false">)</mo></mrow></math></span>.

~~~dg-raw
%%viz f_bars v=0.87,0.12,0.02,0.67,0.24,0.09,0.51,0.31,0.19 l=T=0.5,·,·,T=1,·,·,T=2,·,· hi=none yl=probability_of_each_token | Softmax of logits (2, 1, 0) at three temperatures. Low T piles probability on the top token; high T spreads it out.
~~~

```
z = np.array([2.0, 1, 0])
sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
print(round(-np.log(sm(z)[0]), 3))  # 0.408
print(np.round(sm(z / 0.5), 2), np.round(sm(z / 2), 2))  # [0.87 0.12 0.02] [0.51 0.31 0.19]
```

<div class="take">
<div class="lab">Key takeaway</div>

softmax plus cross-entropy is maximum likelihood for a categorical model; temperature reshapes it.

</div>

## Where this breaks in practice

- Compute softmax after subtracting the largest logit, or <span class="katex"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>e</mi><mi>z</mi></msup></mrow></math></span> overflows.

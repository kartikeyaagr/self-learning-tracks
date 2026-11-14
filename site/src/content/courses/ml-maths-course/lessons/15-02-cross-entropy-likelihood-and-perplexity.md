---
id: "15.02"
course: "ml-maths-course"
module: "15"
slug: "cross-entropy-likelihood-and-perplexity"
title: "Cross-entropy, likelihood and perplexity"
keyIdeas: ["Cross-entropy $H(p, q) = -\\sum_x p(x)\\log q(x)$ is the average code length when data from $p$ is coded with a code built for $q$; it equals $H(p) + D_{\\mathrm{KL}}(p\\,\\|\\,q)$.", "With one-hot targets, cross-entropy reduces to $-\\log q(y)$, so the average classification loss is the negative log-likelihood and minimising it is maximum likelihood.", "Perplexity is $\\exp$ of the cross-entropy in nats: the effective number of equally likely choices the model is unsure between.", "Bits per byte divides out the tokeniser, so it compares language models with different vocabularies fairly."]
question: "A language model reaches a validation loss of 2.0 nats per token. Its tokeniser averages 4.0 bytes of text per token. (a) What is the perplexity? (b) What is the loss in bits per token and in bits per byte? (c) A second model has loss 2.2 nats per token but its tokeniser averages 4.6 bytes per token. Which model compresses text better?"
answer: "(a) Perplexity $= e^{2.0} \\approx 7.39$: on average the model is as uncertain as a uniform choice among about 7.4 tokens. (b) $2.0/\\ln 2 \\approx 2.885$ bits per token; divided by 4.0 bytes per token gives about 0.72 bits per byte. (c) $2.2/\\ln 2 \\approx 3.174$ bits per token, divided by 4.6 gives about 0.69 bits per byte. The second model has higher loss per token but lower bits per byte, so it compresses text better; per-token loss is not comparable across tokenisers."
sources: [{"title": "Wikipedia, Cross-entropy", "url": "https://en.wikipedia.org/wiki/Cross-entropy", "note": "Definitions and the link to likelihood."}, {"title": "Wikipedia, Perplexity", "url": "https://en.wikipedia.org/wiki/Perplexity", "note": "Definition and interpretation for language models."}, {"title": "Deletang et al., Language Modeling Is Compression (2023)", "url": "https://arxiv.org/abs/2309.10668", "note": "Language models used as compressors via arithmetic coding."}, {"title": "MacKay, Information Theory, Inference, and Learning Algorithms (free book)", "url": "https://www.inference.org.uk/itprnn/book.pdf", "note": "Chapter 6 on arithmetic coding with a probabilistic model."}]
practice: ["cross-entropy-loss", "log-loss-binary-cross-entropy", "perplexity-computation", "softmax-cross-entropy-gradient"]
tags: ["cross-entropy", "negative log-likelihood", "perplexity", "bits per byte"]
tools: ["NumPy", "PyTorch"]
---

## From entropy to cross-entropy

Lesson 15.01 showed that the ideal code for a distribution $p$ gives symbol $x$ a codeword of $-\log p(x)$ bits. Now suppose you do not know $p$ and build the code from a model $q$ instead. Each symbol costs $-\log q(x)$ bits, and since symbols really come from $p$, the average cost is
$H(p, q) = -\sum_x p(x) \log q(x)$.
This is the cross-entropy. Splitting it,
$H(p, q) = H(p) + D_{\mathrm{KL}}(p\,\|\,q)$.
The first term is the unavoidable cost; the second is the extra cost of using the wrong model. Since $H(p)$ does not depend on $q$, minimising cross-entropy over $q$ is the same as minimising the KL divergence from $p$ to $q$.

## Cross-entropy as a classification loss

In classification, the target distribution for one example is one-hot on the true class $y$, so the sum keeps only one term:
$H(\text{one-hot}, q) = -\log q(y)$.
Averaged over a dataset, this is the negative log-likelihood of the labels under the model. Minimising it is maximum likelihood estimation (ML Maths 12.11). With logits $z$ and softmax $q = \mathrm{softmax}(z)$, the gradient with respect to the logits is simply $q - \text{one-hot}(y)$, which is why the softmax and cross-entropy are fused in libraries (ML Maths 10.32).

For binary labels with predicted probability $\hat{p}$, the binary cross-entropy is $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$, the same idea with two classes.

## What the number means

A loss of 0.69 nats on a balanced binary task means the model is no better than a coin ($\ln 2 \approx 0.693$). For $K$ classes, the loss of a uniform guess is $\ln K$. Comparing a model's loss with $\ln K$ and with the loss of predicting class frequencies (the label entropy) tells you how much it has learned.

Cross-entropy punishes confident mistakes heavily: predicting the true class with probability 0.01 costs $\ln 100 \approx 4.6$ nats, while 0.5 costs 0.69. A few confident errors can dominate the average, which is often what you want for calibration but also why noisy labels hurt.

## Perplexity

For language models the loss is the average cross-entropy per token. Perplexity is
$\mathrm{PPL} = \exp\!\Big(-\tfrac{1}{N}\sum_{t=1}^{N} \ln q(x_t \mid x_{<t})\Big)$.
It reads as an effective branching factor: perplexity 20 means the model is, on average, as uncertain as if it were choosing uniformly among 20 tokens. ML Maths 12.16 shows how held-out likelihood and perplexity are computed; here the point is the information meaning.

~~~dg-flow Converting a language model loss into comparable units.
Loss in nats per token -(exp)-> Perplexity
Loss in nats per token -(divide by ln 2)-> Bits per token -(divide by bytes per token)-> *Bits per byte
~~~

## Bits per byte and compression

Per-token losses depend on the tokeniser: a tokeniser with longer tokens has fewer, harder predictions per text. Dividing bits per token by the average number of bytes per token gives bits per byte, which compares models with different tokenisers. Arithmetic coding with a model $q$ compresses text to almost exactly $\sum_t -\log_2 q(x_t \mid x_{<t})$ bits, so a language model's loss is literally its compression rate. Better prediction and better compression are the same goal.

<div class="ex">
<div class="lab">Worked example</div>

**comparing two tokenisers**

A team switched to a tokeniser with a larger vocabulary and saw validation loss rise from 2.0 to 2.2 nats per token, and nearly reverted the change. Converting to bits per byte showed the opposite: the old tokeniser averaged 4.0 bytes per token (0.72 bits per byte) and the new one 4.6 bytes per token (0.69 bits per byte). The new model predicted text better; it simply made fewer, harder predictions.

</div>

## Soft targets

When targets are themselves distributions (label smoothing, knowledge distillation, a teacher model's probabilities), the full sum matters and the minimum is no longer zero: it is $H(p)$ of the target. A distillation loss that stops falling may be close to the teacher's own entropy rather than stuck. Lesson 15.05 develops these cases.

## Common mistakes

- Comparing per-token losses or perplexities across tokenisers.
- Computing perplexity as the average of per-sequence perplexities; it is the exponential of the average loss over all tokens.
- Taking $\log$ of a softmax computed in low precision; use the fused log-softmax for numerical stability.
- Reading a loss in nats as bits, or the reverse.

<div class="take">
<div class="lab">Key takeaway</div>

cross-entropy is the code length under a model, equal to entropy plus KL; as a loss it is negative log-likelihood, its exponential is perplexity, and bits per byte is the fair way to compare language models.

</div>

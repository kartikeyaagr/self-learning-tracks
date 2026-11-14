---
id: "15.01"
course: "ml-maths-course"
module: "15"
slug: "entropy-as-average-surprise-and-code-length"
title: "Entropy as average surprise and code length"
keyIdeas: ["The surprise of an outcome with probability $p$ is $-\\log p$; entropy $H(X) = -\\sum_x p(x)\\log p(x)$ is the expected surprise.", "Entropy is the shortest possible average code length: no lossless code can beat $H(X)$ bits per symbol, and good codes come within a fraction of a bit.", "Entropy is largest ($\\log K$) for a uniform distribution over $K$ outcomes and zero for a certain outcome.", "The chain rule $H(X,Y) = H(X) + H(Y\\mid X)$ splits uncertainty about a pair into steps, and conditioning never increases entropy on average."]
question: "A weather station reports one of four states each hour: sunny with probability 1/2, cloudy 1/4, rain 1/8, snow 1/8. (a) Compute the entropy in bits. (b) Give a prefix code that achieves it. (c) What happens to the average code length if you use a fixed 2-bit code instead, and how does that relate to entropy?"
answer: "(a) $H = \\tfrac12\\cdot1 + \\tfrac14\\cdot2 + \\tfrac18\\cdot3 + \\tfrac18\\cdot3 = 1.75$ bits. (b) Sunny 0, cloudy 10, rain 110, snow 111. No codeword is a prefix of another, and each length equals $-\\log_2 p$, so the average length is exactly 1.75 bits. (c) A fixed 2-bit code averages 2 bits, 0.25 bits more than the entropy. That excess is the cost of coding as if the four states were equally likely: it equals the KL divergence from the true distribution to the uniform one, $2 - 1.75 = 0.25$ bits. Entropy is the floor for any lossless code; the fixed code wastes bits on frequent symbols."
sources: [{"title": "Shannon, A Mathematical Theory of Communication (1948)", "url": "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf", "note": "The original definition of entropy and the source coding theorem."}, {"title": "MacKay, Information Theory, Inference, and Learning Algorithms (free book)", "url": "https://www.inference.org.uk/itprnn/book.pdf", "note": "Chapters 2, 4 and 5 cover entropy and coding with many exercises."}, {"title": "Olah, Visual Information Theory", "url": "https://colah.github.io/posts/2015-09-Visual-Information/", "note": "Pictures of code lengths and entropy."}, {"title": "Wikipedia, Huffman coding", "url": "https://en.wikipedia.org/wiki/Huffman_coding", "note": "The optimal prefix code construction."}]
practice: ["perplexity-computation", "basic-probability-rules", "decision-tree-classifier-cart"]
tags: ["entropy", "source coding", "bits and nats"]
tools: ["NumPy"]
---

## Where this module sits

Probability (modules 10 to 12) gave you distributions and likelihoods. Information theory adds a way to measure uncertainty in bits, and it turns out to be the language of most ML losses: cross-entropy, KL divergence, perplexity and contrastive losses are all information quantities. The AI Research course (lesson 01.11) builds the intuition for entropy through surprise; this module adds the coding view, the identities you need for derivations, and the links to losses.

## Surprise and entropy

An outcome with probability $p$ carries surprise $-\log p$. A certain event ($p = 1$) carries none; a one-in-a-million event carries about 20 bits. Using base 2 gives bits; the natural log gives nats ($1$ nat $= 1/\ln 2 \approx 1.443$ bits). Deep learning libraries use nats.

Entropy is the expected surprise:
$H(X) = -\sum_x p(x) \log p(x)$.
Three facts are worth memorising:

- $H(X) \ge 0$, with equality only when one outcome is certain.
- For $K$ outcomes, $H(X) \le \log K$, with equality for the uniform distribution.
- $H$ is concave in $p$: mixing distributions increases entropy.

For a coin with probability $p$ of heads, $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$. It is 1 bit at $p = 0.5$, about 0.47 bits at $p = 0.9$, and 0 at $p = 1$.

## The coding view

Suppose you must send a long sequence of symbols in binary, and each symbol has a codeword. If frequent symbols get short codewords, the average length falls. Shannon's source coding theorem says the average length of any lossless code is at least $H(X)$ bits per symbol, and that codes exist with average length below $H(X) + 1$. Coding blocks of symbols together, or using arithmetic coding, gets as close to $H(X)$ as you like.

The ideal codeword length for a symbol is $-\log_2 p(x)$ bits. That is why surprise and code length are the same thing: an unlikely symbol needs a long description.

~~~dg-flow From a distribution to a code length.
Distribution p(x) -(surprise -log p)-> Ideal code length per symbol -(average over p)-> *Entropy H(X) -(Huffman, arithmetic coding)-> Achievable bits per symbol
~~~

<div class="ex">
<div class="lab">Worked example</div>

**Huffman coding a skewed alphabet**

Four symbols have probabilities 0.4, 0.3, 0.2 and 0.1. Entropy is about 1.85 bits. Huffman's algorithm repeatedly merges the two least likely symbols: 0.1 and 0.2 make 0.3; that and 0.3 make 0.6; that and 0.4 make 1. The resulting lengths are 1, 2, 3 and 3 bits, for an average of $0.4 + 0.6 + 0.6 + 0.3 = 1.9$ bits, within 0.05 bits of the entropy. A fixed-length code would need 2 bits.

</div>

## Joint and conditional entropy

For two variables, the joint entropy $H(X,Y)$ is the entropy of the pair. The conditional entropy $H(Y\mid X) = \sum_x p(x) H(Y \mid X = x)$ is the uncertainty left in $Y$ after seeing $X$, averaged over $X$. The chain rule connects them:
$H(X, Y) = H(X) + H(Y \mid X)$.
It extends to sequences: $H(X_1, \dots, X_n) = \sum_i H(X_i \mid X_1, \dots, X_{i-1})$. An autoregressive language model is exactly this factorisation, predicting each token given the previous ones.

On average, conditioning cannot increase entropy: $H(Y \mid X) \le H(Y)$, with equality when $X$ and $Y$ are independent. A particular observation can increase uncertainty (a surprising clue can make you less sure), but the average over observations cannot.

## Continuous variables

For a density $f$, differential entropy is $h(X) = -\int f(x)\log f(x)\,dx$. It behaves differently: it can be negative, and it changes when you rescale $X$. For a Gaussian with variance $\sigma^2$, $h = \tfrac12 \log(2\pi e \sigma^2)$, which is negative when $\sigma$ is small. Among all densities with a given variance, the Gaussian has the largest differential entropy, which is one reason Gaussian noise is a common default assumption. Differences of differential entropies (such as mutual information, lesson 15.04) do not depend on scale and are well behaved.

## Why entropy matters in ML

- Decision trees choose splits that reduce label entropy most (lesson 15.04).
- The entropy of a model's predictive distribution measures its uncertainty, used in active learning and for flagging inputs to review.
- An entropy bonus in reinforcement learning keeps a policy from collapsing too early onto one action.
- Tokenisers and compression connect directly to entropy: a better model of text gives shorter codes (lesson 15.02).

## Common mistakes

- Mixing bases: reporting a loss in nats as if it were bits. Convert by dividing by $\ln 2$.
- Treating differential entropy as a count of bits; it is only meaningful up to the scale of the variable.
- Estimating entropy from a small sample by plugging in frequencies; this underestimates it (lesson 15.06).

<div class="take">
<div class="lab">Key takeaway</div>

entropy is expected surprise and also the minimum average code length; it is largest for uniform distributions, splits by the chain rule, and appears wherever ML measures uncertainty.

</div>

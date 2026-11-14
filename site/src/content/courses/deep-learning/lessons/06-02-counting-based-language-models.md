---
id: "06.02"
course: "deep-learning"
module: "06"
slug: "counting-based-language-models"
title: "Counting-based language models"
keyIdeas: ["A language model assigns probabilities to sequences. The chain rule factors $P(w_1..w_N)$ into next-word probabilities; an n-gram model assumes each word depends only on the previous $n-1$.", "Maximum-likelihood n-gram estimates are count ratios, and any unseen n-gram gets probability zero, which makes a test sentence containing it infinitely surprising.", "Smoothing moves probability mass to unseen events: add-k (simple, steals too much), interpolation and backoff (mix orders), and Kneser-Ney (uses how many contexts a word appears in, the best classical method).", "Perplexity is the exponential of average negative log-likelihood per token: the effective number of equally likely choices. A uniform model over $V$ words has perplexity $V$.", "Perplexities are comparable only with the same tokenisation and test text. Convert to per-word or bits-per-byte figures to compare across tokenisers."]
question: "Model A uses a 50k word-level vocabulary and gets perplexity 120 on a test set. Model B uses an 8k BPE vocabulary and gets perplexity 40 on the same text, where B's tokeniser produces on average 1.4 tokens per word. Which model is better, and how do you know?"
answer: "The raw numbers are not comparable, because perplexity is per token and B has 1.4 tokens per word. Compare total log-likelihood of the same text instead. B's average negative log-likelihood is ln 40 ≈ 3.69 nats per token, so per word it is 1.4 × 3.69 ≈ 5.16 nats, a per-word perplexity of e^5.16 ≈ 175. A's per-word perplexity is 120 (ln 120 ≈ 4.79 nats per word). So A assigns higher probability to the test text and is the better model, despite B's lower number. Also check that A does not gain unfairly from mapping rare words to a cheap <unk>; bits per byte on raw text avoids both issues."
sources: [{"title": "Jurafsky and Martin, Speech and Language Processing (3rd ed. draft)", "url": "https://web.stanford.edu/~jurafsky/slp3/", "note": "The n-gram language model chapter: smoothing, Kneser-Ney and perplexity with worked examples."}, {"title": "An Empirical Study of Smoothing Techniques for Language Modeling (Chen and Goodman, 1996)", "url": "https://aclanthology.org/P96-1041/", "note": "The classic comparison establishing interpolated Kneser-Ney as the strongest n-gram method."}]
practice: ["n-gram-language-model", "perplexity-computation", "conditional-perplexity"]
tags: ["n-gram", "smoothing", "perplexity"]
tools: ["Python"]
---

## Language models before neural networks

A language model assigns a probability to any sequence of tokens. The chain rule makes this exact:

$$P(w_1, \dots, w_N) = \prod_{i=1}^{N} P(w_i \mid w_1, \dots, w_{i-1})$$

Every neural language model, including the RNN of 05.01 and every LLM, estimates these conditional probabilities. The n-gram model is the counting version, and it is still worth knowing: it is a fast baseline, a tool for data filtering and spelling correction, and it is where perplexity, smoothing and the sparsity problem come from.

## The Markov assumption and counting

An n-gram model assumes the next word depends only on the previous $n - 1$ words. For a trigram model, $P(w_i \mid w_{i-2}, w_{i-1})$, estimated by maximum likelihood as

$$P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$$

Pad each sentence with start symbols `<s>` and an end symbol `</s>` so the first words have a history and the model learns where sentences end. Work in log space: multiplying hundreds of probabilities below 0.1 underflows a float.

The fatal problem is sparsity. With a 50,000-word vocabulary there are $1.25 \times 10^{14}$ possible trigrams; even a billion-word corpus sees a tiny fraction. Any unseen trigram has MLE probability zero, so any test sentence containing one gets probability zero, and its perplexity is infinite.

## Smoothing

| Method | Idea | Weakness |
| --- | --- | --- |
| Add-k (Laplace when k = 1) | add $k$ to every count: $(C + k)/(C_{\text{hist}} + kV)$ | with large $V$ it gives far too much mass to unseen events |
| Interpolation | $\lambda_3 P_3 + \lambda_2 P_2 + \lambda_1 P_1$, weights tuned on held-out data | lower orders still use raw frequency |
| Backoff (Katz; "stupid backoff" at web scale) | use the trigram if seen, else a discounted bigram, else unigram | needs discounting to stay normalised |
| Kneser-Ney | absolute discounting plus a continuation probability for the lower order | more bookkeeping; the standard choice |

Kneser-Ney's key idea concerns what the lower-order model should measure. "Francisco" may be frequent, but it appears almost only after "San". When the model backs off to unigrams because the bigram "the Francisco" is unseen, raw frequency says "Francisco" is likely; it should not be, since it virtually never starts a new context. KN replaces the unigram probability with a continuation probability proportional to the number of distinct words that precede $w$. Modified KN with interpolation remained the best n-gram method in Chen and Goodman's extensive comparison.

```python
from collections import Counter
def bigram_lm(sentences, k=1.0):
    uni, bi = Counter(), Counter()
    for s in sentences:
        toks = ["<s>"] + s.split() + ["</s>"]
        uni.update(toks[:-1]); bi.update(zip(toks, toks[1:]))
    V = len(set(uni) | {"</s>"})
    return lambda prev, w: (bi[prev, w] + k) / (uni[prev] + k * V)   # add-k
```

~~~dg-flow Building and scoring an n-gram model. Counts come from training text only; perplexity is measured on held-out text.
Training text -(pad with <s>, </s>)-> n-gram counts -> *Smoothed P(w given history) -(held-out text)-> Perplexity
~~~

## Perplexity

Perplexity on a held-out text of $N$ tokens is

$$\text{PPL} = \exp\Big(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid \text{history}_i)\Big)$$

It is the inverse of the geometric-mean probability per token, and reads as the effective branching factor: perplexity 100 means the model is, on average, as uncertain as if choosing uniformly among 100 options. A uniform model over $V$ tokens has perplexity exactly $V$; lower is better. Cross-entropy in bits is $\log_2 \text{PPL}$.

<div class="ex">
<div class="lab">Worked example</div>

**smoothing and perplexity by hand**

Vocabulary $V = 10$, $C(\text{the}) = 10$, $C(\text{the cat}) = 3$, $C(\text{the dog}) = 0$.

MLE: $P(\text{cat} \mid \text{the}) = 0.3$, $P(\text{dog} \mid \text{the}) = 0$. Add-1: $(3+1)/(10+10) = 0.2$ and $(0+1)/20 = 0.05$. Note how much the seen event lost.

A 3-token test text with probabilities 0.2, 0.1, 0.05: mean log-probability $(-1.609 - 2.303 - 2.996)/3 = -2.303$, so PPL $= e^{2.303} = 10$.

</div>

## Comparing perplexities honestly

Perplexity is per token, so it depends on what a token is. A model with a smaller subword vocabulary predicts more, easier tokens per word, and its per-token perplexity looks lower even if it is a worse model of the text. To compare across tokenisers, compare total log-likelihood of the same text: convert to per-word perplexity, or to bits per byte (total negative log-likelihood in bits divided by the UTF-8 byte count). The unknown-word token is another trap: mapping many rare words to one cheap `<unk>` lowers perplexity artificially. Fix the vocabulary and test text before comparing.

## Where n-grams still matter

- Fast baselines and features (lesson 06.03 uses n-gram counts as features).
- Data pipelines: KenLM-style n-gram models score web text for quality filtering in LLM pretraining.
- Candidate generation and rescoring in speech recognition and spelling correction.
- Teaching: the n-gram's failure is exactly what neural LMs fix, by sharing statistics between similar contexts through embeddings rather than treating every history as a separate count.

<div class="take">
<div class="lab">Key takeaway</div>

an n-gram model is the chain rule plus a short memory, estimated by counting. Smoothing decides how much probability unseen events get, and perplexity measures average surprise per token, comparable only when the tokenisation and test text match.

</div>

---
id: "06.02"
course: "deep-learning"
module: "06"
slug: "counting-based-language-models"
title: "Counting-based language models"
keyIdeas: ["A **language model** gives a probability to every token sequence. The chain rule turns this into a product of next-token probabilities, and an **n-gram model** assumes each token depends only on the previous $n-1$.", "Maximum-likelihood n-gram probabilities are count ratios, $C(h, w)/C(h)$. Any n-gram unseen in training gets probability 0, and on real text most longer n-grams are unseen (71% of test trigrams in this lesson's corpus).", "**Smoothing** moves probability to unseen events: add-k is simple but over-feeds them; interpolation and backoff mix in lower orders; **Kneser-Ney** discounts every count by a fixed amount and backs off to a continuation probability (how many distinct words come before $w$), and is the best classical method.", "**Perplexity** is the exponential of the average negative log-likelihood per held-out token: the effective number of equally likely choices. A uniform model over $V$ words scores exactly $V$; lower is better, and training-set perplexity proves nothing.", "Perplexities are comparable only on the same text with the same tokenisation and `<unk>` convention. Convert to per-word perplexity or bits per byte to compare across tokenisers."]
question: "Model A uses a 50k word-level vocabulary and gets perplexity 120 on a test set. Model B uses an 8k BPE vocabulary and gets perplexity 40 on the same text, where B's tokeniser produces on average 1.4 tokens per word. Your team lead wants to ship B because 'its perplexity is three times lower'. Which model is better, how do you know, and what else would you check before deciding?"
answer: "The raw numbers are not comparable, because perplexity is an average per token and the two models use different tokens: B makes 1.4 predictions for every word that A predicts once. The fair comparison is the total log-likelihood each model gives the same text, expressed per word. B's average negative log-likelihood is ln 40 ≈ 3.69 nats per token, so per word it is 1.4 × 3.69 ≈ 5.16 nats, a per-word perplexity of e^5.16 ≈ 175. A's per-word perplexity is its per-token figure, 120 (ln 120 ≈ 4.79 nats per word). So A assigns higher probability to the test text (4.79 < 5.16 nats per word) and is the better language model, despite B's lower headline number. Before deciding, check A's out-of-vocabulary handling: a 50k word vocabulary maps every unseen word to <unk>, and if <unk> is frequent and cheap to predict, A's 120 is flattered, whereas B's subwords can spell any word. The cleanest comparison is bits per byte on the raw test text (total negative log-likelihood in bits divided by the UTF-8 byte count), which neither tokenisation nor <unk> can distort. Also confirm the test text is identical, was not seen in training (deduplicate), and that any smoothing or tuning used a separate validation set."
sources: [{"title": "Jurafsky and Martin, Speech and Language Processing (3rd ed. draft)", "url": "https://web.stanford.edu/~jurafsky/slp3/", "note": "The n-gram language model chapter: smoothing, Kneser-Ney and perplexity with worked examples."}, {"title": "An Empirical Study of Smoothing Techniques for Language Modeling (Chen and Goodman, 1996)", "url": "https://aclanthology.org/P96-1041/", "note": "The classic comparison establishing interpolated Kneser-Ney as the strongest n-gram method."}]
practice: ["n-gram-language-model", "perplexity-computation", "conditional-perplexity"]
tags: ["n-gram", "smoothing", "perplexity"]
tools: ["Python"]
---

## What you will learn

- What a language model is, and why the chain rule turns it into "predict the next word".
- How an n-gram model estimates next-word probabilities by counting, by hand on a six-sentence corpus.
- Why counting fails on new text (zero probabilities), and how much real text is unseen.
- How smoothing fixes it: add-k, interpolation, backoff and Kneser-Ney, measured on real English.
- How to compute and read perplexity, and compare it honestly across tokenisers.

*Before you start:* conditional probability, logarithms, and tokens from lesson 06.01. The RNN language model of 05.01 estimates the same quantities with a network; this is the counting version.

Lesson 06.01 turned raw text into tokens. The next question is how likely a given sequence is. A spelling corrector must rank "they're going home" above "their going home"; a speech recogniser must prefer "recognise speech" to "wreck a nice beach". Both need a number for "how plausible is this text", and the oldest way to get it is to count.

## The idea in one picture

Here is a complete training corpus of six sentences:

| # | Sentence
| --- | --- |
| 1 | the cat sat on the mat
| 2 | the dog sat on the rug
| 3 | the cat ate the fish
| 4 | a dog chased the cat
| 5 | the cat sat on the rug
| 6 | a cat ate a fish

Ask: after the word "the", what comes next? The word "the" appears 9 times. It is followed by "cat" 4 times, "rug" twice, and "mat", "dog" and "fish" once each. Dividing by 9 gives a probability for each possible next word.

~~~dg-fig 06.02-next-word.svg
Next-word probabilities after "the", read off the counts. The bars sum to 1; every other word gets 0.
~~~

That is the whole idea. A **language model** assigns a probability to any sequence of tokens; an **n-gram model** does it by counting how often short runs of words (**n-grams**: a bigram is 2 words, a trigram 3) occur in training text. The words the model conditions on are the **history** or context.

The rest of the lesson asks: why are next-word probabilities enough to score a sentence, what do we do about the zeros, and how do we measure the model?

## Step 1: the chain rule scores a whole sentence

Any joint probability can be split into a product of conditional ones, with no approximation at all:

$$P(w_1, \dots, w_N) = \prod_{i=1}^{N} P(w_i \mid w_1, \dots, w_{i-1})$$

Here $w_i$ is the $i$-th token and $N$ the sentence length: the first word's probability, times the second's given the first, and so on. So "score a sentence" and "predict the next word, repeatedly" are the same job. All language models, LLMs included, estimate these; they differ in how they handle the history.

## Step 2: the Markov assumption shortens the history

The exact history grows with every word, and most long histories never occur twice, so you cannot count them. An n-gram model makes the **Markov assumption**: the next word depends only on the previous $n - 1$ words.

$$P(w_i \mid w_1, \dots, w_{i-1}) \approx P(w_i \mid w_{i-n+1}, \dots, w_{i-1})$$

For a bigram model ($n = 2$) the history is one word; for a trigram model ($n = 3$) it is two. This is wrong for real language (choosing "are" in "the keys to the cabinet are" needs "keys", four words back), but it makes counting possible.

## Step 3: estimate by counting

The **maximum-likelihood estimate** (MLE) is the count of the n-gram divided by the count of its history. For a trigram:

$$P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$$

$C(\cdot)$ is the number of times that sequence appears in the training text. For a bigram it is $C(w_{i-1}, w_i)/C(w_{i-1})$, exactly what the bar chart did: $P(\text{cat} \mid \text{the}) = 4/9 = 0.444$.

Two practical details:

- **Padding.** Put $n-1$ start symbols `<s>` before each sentence and an end symbol `</s>` after it, so the first word has a history and the model learns where sentences end.
- **Log space.** Multiplying hundreds of probabilities below 0.1 underflows a float to exactly 0. Add log-probabilities instead.

<div class="ex">
<div class="lab">Worked example</div>

**score a sentence with a bigram model, by hand**

Test sentence "the dog sat on the mat", padded: `<s> the dog sat on the mat </s>`, 7 predictions. Counts from the six sentences: $C(\text{<s>}) = 6$, $C(\text{the}) = 9$, $C(\text{dog}) = 2$, $C(\text{sat}) = C(\text{on}) = 3$, $C(\text{mat}) = 1$.

Probabilities: $P(\text{the} \mid \text{<s>}) = 4/6$, $P(\text{dog} \mid \text{the}) = 1/9$, $P(\text{sat} \mid \text{dog}) = 1/2$, $P(\text{on} \mid \text{sat}) = 3/3$, $P(\text{the} \mid \text{on}) = 3/3$, $P(\text{mat} \mid \text{the}) = 1/9$, $P(\text{</s>} \mid \text{mat}) = 1/1$.

Product: $\frac{4}{6} \times \frac{1}{9} \times \frac{1}{2} \times 1 \times 1 \times \frac{1}{9} \times 1 = \frac{4}{972} = 0.00412$.

Log form: $\ln 0.00412 = -5.493$, or $-5.493 / 7 = -0.785$ per prediction. The sentence was never in training, but every bigram in it was, so it gets a sensible score.

Now try "the cat chased a dog". The bigram "cat chased" never occurred, so $P(\text{chased} \mid \text{cat}) = 0/5 = 0$ and the whole product is 0, although the sentence is perfectly reasonable.

</div>

## Step 4: measure the model with perplexity

To compare models, score held-out text with **perplexity**: the exponential of the average negative log-probability per token,

$$\text{PPL} = \exp\Big(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid \text{history}_i)\Big),$$

where $N$ is the number of predicted tokens (including each `</s>`) and the log is natural. It equals one over the geometric-mean probability per token. Read it as an **effective branching factor**: perplexity 100 means the model is, on average, as unsure as if it were picking uniformly among 100 equally likely words. Lower is better. A model that spreads probability uniformly over $V$ words has perplexity exactly $V$, which is a useful sanity bound.

Number checks:

- The hand example: average log-probability $-0.785$, so $\text{PPL} = e^{0.785} = 2.19$, about as uncertain as choosing between two words.
- Three tokens with probabilities 0.2, 0.1 and 0.05: mean log-probability $(-1.609 - 2.303 - 2.996)/3 = -2.303$, so $\text{PPL} = e^{2.303} = 10$.
- The exponent itself, $-\frac{1}{N}\sum \log P$, is the **cross-entropy** per token. With base-2 logs it is in bits, and bits $= \log_2 \text{PPL}$.

A single zero probability makes perplexity infinite, so an unsmoothed n-gram model on new text is unusable.

## Step 5: how big the zero problem is

Zeros are not rare. A 50,000-word vocabulary has $50{,}000^3 = 1.25 \times 10^{14}$ possible trigrams; a billion-word corpus contains at most a billion.

To see it on real text, I used the English prose of the Python documentation (the `pydoc_data` topics shipped with Python): 45k training and 13k test words from different pages, lowercased and deduplicated, with words seen once in training mapped to `<unk>` (2,016 vocabulary entries; 8.9% of test tokens are `<unk>`).

~~~dg-fig 06.02-sparsity.svg
Left: share of test n-grams never seen in training: 32% of bigrams and 71% of trigrams are new. Right: more training text helps only slowly.
~~~

Language keeps producing new combinations (Zipf's long tail, 06.01), so zeros are the normal case and a usable model must give probability to things it never counted.

## Step 6: smoothing moves probability to unseen events

**Smoothing** takes a little probability from seen n-grams and gives it to unseen ones, so nothing is exactly zero. The total for each history must stay 1, so mass for unseen words comes out of seen ones.

**Add-k smoothing.** Pretend every possible n-gram was seen $k$ extra times. For a bigram,

$$P_{\text{add-}k}(w \mid h) = \frac{C(h, w) + k}{C(h) + kV},$$

where $V$ is the number of possible next tokens. The $kV$ in the denominator keeps the total at 1. With $k = 1$ this is **Laplace smoothing**.

<div class="ex">
<div class="lab">Worked example</div>

**add-1 on the toy corpus**

The toy vocabulary has $V = 12$ possible next tokens (11 words plus `</s>`).

$P(\text{cat} \mid \text{the}) = (4 + 1)/(9 + 12) = 5/21 = 0.238$, down from 0.444.

$P(\text{sat} \mid \text{the}) = (0 + 1)/21 = 0.048$, up from 0. Seven words never followed "the", so together they now get $7/21 = 0.333$ of the probability.

The sentence "the cat chased a dog" now scores: $P(\text{chased} \mid \text{cat}) = (0+1)/(5+12) = 0.059$, and the sentence perplexity is 8.40 instead of infinity.

</div>

~~~dg-fig 06.02-smoothing.svg
Add-1 gives a third of the probability after "the" to words never seen there and cuts "cat" from 0.44 to 0.24. Add-0.1 is gentler: unseen words share only 7% of the mass.
~~~

With a large $V$ add-1 is far too generous: for a history seen 10 times among 2,016 possible next words, about 99% of the mass goes to words never seen after it. Smaller $k$ helps, but even the best $k$ is poor.

~~~dg-fig 06.02-addk.svg
Add-k test perplexity against k on the docs corpus. Too small a k leaves near-zeros, too large flattens everything; even the best k is far behind Kneser-Ney (dotted).
~~~

**Interpolation.** Instead of inventing counts, mix models of different orders, so a trigram estimate is always backed by a bigram and a unigram estimate:

$$P_{\text{interp}}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_3 P_3(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P_2(w_i \mid w_{i-1}) + \lambda_1 P_1(w_i),$$

with $\lambda_1 + \lambda_2 + \lambda_3 = 1$, tuned on held-out data (never on the test set). On the docs corpus, weights tuned on a held-out slice of training text came out at $(\lambda_1, \lambda_2, \lambda_3) = (0.4, 0.4, 0.2)$ and gave test perplexity 146, against 685 for the best add-k trigram.

**Backoff.** Use the trigram if it was seen, otherwise fall back to the bigram, then the unigram. Katz backoff discounts seen counts so the result stays a proper distribution. Google's web-scale **stupid backoff** skips normalisation and multiplies the lower-order score by 0.4; the scores no longer sum to 1, but with trillions of words it worked well and was cheap.

**Kneser-Ney (KN)** is the best classical method and has two ingredients:

- **Absolute discounting**: subtract a fixed $d$ (typically 0.75) from every seen count. A count of 50 barely notices; a count of 1 loses most of its weight, which matches reality, since n-grams seen once are less reliable than their count suggests.
- **Continuation probability** for the lower order. "Francisco" is frequent, but almost only after "San". If "the Francisco" was never seen, the fallback should not rate "Francisco" highly, because it rarely starts a new context. KN replaces the unigram probability with one proportional to the number of *distinct* words that precede $w$:

$$P_{\text{cont}}(w) = \frac{\left|\{v : C(v, w) > 0\}\right|}{\left|\{(v, u) : C(v, u) > 0\}\right|}$$

Numerator: distinct left neighbours of $w$. Denominator: distinct bigram types, so the values sum to 1.

~~~dg-fig 06.02-continuation.svg
Raw frequency against continuation probability. "same" occurs 127 times but after only 4 different words, so KN gives it a tiny fallback probability; "type" follows 99 different words and keeps most of its weight.
~~~

| Method | Weakness | Trigram test PPL (docs corpus)
| --- | --- | --- |
| Add-1 (Laplace) | far too much mass to unseen events when $V$ is large | 1,291
| Add-k, best k | one $k$ for every history | 685
| Interpolation | lower orders still use raw frequency | 146
| Backoff (Katz; stupid backoff) | needs discounting to stay normalised | not measured
| Kneser-Ney | more bookkeeping; the standard choice | 132

Interpolated modified Kneser-Ney came out best in Chen and Goodman's large comparison of smoothing methods, and KenLM builds it by default.

~~~dg-flow Building and scoring an n-gram model. Counts come from training text only; perplexity is measured on held-out text.
Training text -(pad with <s>, </s>)-> n-gram counts -(discount, interpolate)-> *Smoothed P(w given history) -(held-out text)-> Perplexity
~~~

## An n-gram model in code

A complete bigram model with add-k smoothing, perplexity and sampling fits on one screen of plain Python:

```python
import math, random
from collections import Counter

train = ["the cat sat on the mat", "the dog sat on the rug", "the cat ate the fish",
         "a dog chased the cat", "the cat sat on the rug", "a cat ate a fish"]

def pad(sentence):                       # bigram model: one <s> of history is enough
    return ["<s>"] + sentence.split() + ["</s>"]

hist, pair = Counter(), Counter()
for s in train:
    t = pad(s)
    hist.update(t[:-1])                  # every token that acts as a history
    pair.update(zip(t, t[1:]))           # every (previous, next) pair
V = len({w for s in train for w in pad(s)} - {"<s>"})   # 12 possible next tokens

def prob(prev, w, k=0.0):                # add-k estimate; k=0 is plain counting (MLE)
    return (pair[prev, w] + k) / (hist[prev] + k * V)

def perplexity(sentences, k=0.0):
    logp, n = 0.0, 0
    for s in sentences:
        t = pad(s)
        for prev, w in zip(t, t[1:]):
            p = prob(prev, w, k)
            if p == 0: return float("inf")       # one unseen pair ruins everything
            logp += math.log(p); n += 1
    return math.exp(-logp / n)

print(prob("the", "cat"), prob("the", "cat", k=1))         # 0.444  0.238
print(perplexity(["the dog sat on the mat"]))               # 2.19
print(perplexity(["the cat chased a dog"]))                 # inf
print(perplexity(["the cat chased a dog"], k=1))            # 8.40

def generate(rng):                       # sample one word at a time from P(w | prev)
    words, prev = [], "<s>"
    while prev != "</s>":
        nxt = [w for w in pair if w[0] == prev]
        prev = rng.choices([b for _, b in nxt], [prob(*x) for x in nxt])[0]
        words.append(prev)
    return " ".join(words[:-1])
rng = random.Random(0)
print([generate(rng) for _ in range(3)])   # ['a fish', 'the mat', 'a dog sat on the mat']
```

"a dog sat on the mat" was never in training: it is stitched from bigrams that were. For real work use a library with interpolated Kneser-Ney and compact storage, such as KenLM or `nltk.lm`.

## Reading the results

~~~dg-fig 06.02-order.svg
Perplexity against model order. Raw counts reach 1.7 on training text (memorisation) but are infinite on test text; add-1 worsens as n grows; Kneser-Ney improves from 294 (unigram) to 132 (trigram), then levels off.
~~~

- **Training perplexity is not evidence.** The unsmoothed 5-gram scores 1.7 on its training text and infinity on new text.
- **The best order depends on data size.** With 45k words, KN gains nothing beyond trigrams (132, 134, 135 for n = 3, 4, 5). With billions of words 5-grams pay off.
- **Check that probabilities sum to 1** for a few seen and unseen histories, over the vocabulary plus `</s>`. A wrong denominator silently breaks every perplexity.
- **Know your `<unk>` rate** (8.9% here): cheap `<unk>` predictions flatter a model.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| Test perplexity is infinite | some test n-gram has probability 0 | smoothing; map unknown words to `<unk>`
| Probabilities underflow to 0.0 for long texts | multiplying many small numbers | sum log-probabilities
| Perplexity gets worse as n grows | add-k over-feeding unseen events, or too little data | Kneser-Ney or interpolation; lower n
| Great perplexity that does not hold up | tuned $k$ or $\lambda$ on the test set, or test text duplicated in training | tune on a separate held-out set; deduplicate before splitting
| Two models' perplexities disagree with human judgement | different tokenisers or `<unk>` conventions | compare per word or in bits per byte on the same text

The duplicate row is real: a first sentence-level split of the docs corpus put 44% of test sentences in training too.

## Comparing perplexities honestly

Perplexity is per token, so it depends on what a token is. A small subword vocabulary makes more, easier predictions per word, so its per-token perplexity looks lower even for a worse model. Compare the total log-likelihood of the same text instead: per-word perplexity, or **bits per byte** (total negative log-likelihood in bits divided by the UTF-8 byte count).

The other trap is `<unk>`: mapping many rare words to one cheap token lowers perplexity artificially. Fix the vocabulary and test text before comparing, or use bits per byte on raw text.

## Where n-grams still matter

- Fast baselines and features (06.03 uses n-gram counts as classifier features).
- Data filtering: KenLM models trained on clean text score web text for LLM pretraining; very surprising text is often junk.
- Fast rescoring in speech recognition and spelling correction.

## Going deeper

**The full interpolated Kneser-Ney recursion.** For the highest order,

$$P_{\text{KN}}(w \mid h) = \frac{\max(C(h, w) - d,\ 0)}{C(h)} + \frac{d \cdot N_{1+}(h\,\bullet)}{C(h)}\, P_{\text{KN}}(w \mid h'),$$

where $h'$ is the history with its oldest word dropped and $N_{1+}(h\,\bullet)$ is the number of distinct words seen after $h$. The second weight is exactly the mass removed by discounting, so the distribution still sums to 1. At every lower order the raw count $C$ is replaced by the continuation count $N_{1+}(\bullet\, h\, w)$, the number of distinct words seen to the left; the recursion ends in $P_{\text{cont}}$, optionally mixed with a uniform distribution so that even a never-seen word keeps a little probability.

**Choosing the discount.** A standard estimate (Ney and colleagues) is $d = n_1/(n_1 + 2 n_2)$, with $n_1$, $n_2$ the numbers of n-gram types seen exactly once and twice: 0.68 for bigrams and 0.81 for trigrams on the docs corpus. For the trigram model, $d = 0.9$ gave test perplexity 128 and $d = 0.5$ gave 160 (vs 132 at 0.75). **Modified** KN uses separate discounts for counts of 1, 2 and 3+.

**Why counting hits a wall.** Each history is its own row of counts, so "the cat sat" tells the model nothing about "the kitten sat". Parameters grow like $V^n$ while data per parameter shrinks. Neural language models (Bengio and colleagues, 2003; the RNNs of 05.01; Transformers) replace the table with a function of word embeddings, so similar histories share evidence and long contexts become usable.

**Perplexity and compression.** Cross-entropy in bits is the code length per token of an ideal compressor built on the model, so a better language model is a better compressor.

## Key terms

| Term | Meaning
| --- | --- |
| Language model | A model that gives a probability to every sequence of tokens, usually via next-token probabilities
| n-gram | A run of $n$ consecutive tokens; an n-gram model conditions on the previous $n-1$ tokens
| Markov assumption | Only the last $n-1$ tokens matter for predicting the next one
| Smoothing | Moving probability from seen to unseen n-grams so nothing has probability 0
| Kneser-Ney | Absolute discounting plus continuation probabilities; the strongest classical smoothing
| Continuation probability | How many distinct words precede $w$, normalised; KN's lower-order model
| Perplexity | $\exp$ of the average negative log-probability per token; the effective number of choices
| Bits per byte | Total negative log-likelihood in bits divided by the text's byte count; tokeniser-independent

<div class="take">
<div class="lab">Key takeaway</div>

an n-gram model is the chain rule plus a short memory, estimated by counting. Almost every long n-gram in new text is unseen, so smoothing decides the model's quality: add-k is a weak fix, interpolated Kneser-Ney is the strong one. Judge models by held-out perplexity, and compare perplexities only on the same text with the same tokenisation (or in bits per byte).

</div>

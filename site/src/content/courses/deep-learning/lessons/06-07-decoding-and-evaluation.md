---
id: "06.07"
course: "deep-learning"
module: "06"
slug: "decoding-and-evaluation"
title: "Decoding and evaluation"
keyIdeas: ["Greedy decoding takes the most likely token at each step; **beam search** keeps the $B$ best partial sequences by total log-probability and so finds more probable outputs than greedy.", "Summed log-probabilities favour short outputs, so beam search needs **length normalisation** (divide by $|y|^{\\alpha}$). Very wide beams can still lower quality, even down to empty outputs: the most probable sequence is not the best one.", "For open-ended generation, likelihood-maximising search produces dull, repetitive text; sampling with **temperature**, **top-k** or **top-p** (nucleus) is the norm, and top-p adapts to how confident the model is.", "**BLEU** is clipped n-gram precision ($n$ = 1 to 4, geometric mean) times a brevity penalty, computed at corpus level; **ROUGE** is recall-oriented overlap, with ROUGE-L based on the longest common subsequence. Both measure surface overlap, not meaning.", "Deduplicate training data, exactly by hashing normalised text and approximately with **MinHash** and LSH. Duplicates waste compute, increase memorisation, and leak test examples into training so scores are inflated."]
question: "You raise the beam width of a translation model from 5 to 50. Model log-probability of the outputs goes up, but BLEU goes down and outputs get noticeably shorter; a few are empty. Explain why, and what you would change."
answer: "Beam search maximises the total log-probability, a sum of negative terms, so every extra token lowers the score and hypotheses that end early have an advantage. A narrow beam rarely contains those short, high-probability hypotheses, because the end token is not among its top choices at early steps; a 50-wide beam explores enough to find them, including in some cases ending immediately (the empty output). So the search is *succeeding*: it finds outputs the model rates more probable. What it exposes is a mismatch between the model's probability and translation quality, since the model's mode is not the best translation.\n\nBLEU drops for two reasons. The shorter outputs omit content, so fewer reference n-grams are matched, and once the total candidate length $c$ falls below the reference length $r$ the brevity penalty $e^{1 - r/c}$ multiplies the score down (a candidate 10% short pays about an 11% penalty on top).\n\nFixes: score finished hypotheses with length normalisation, $\\log P(y \\mid x) / |y|^{\\alpha}$ with $\\alpha$ around 0.6 to 1.0 (or GNMT's $((5 + |y|)/6)^{\\alpha}$), possibly with a minimum length or coverage penalty; then tune $B$ and $\\alpha$ jointly on a dev set while watching the length ratio $c/r$ that sacreBLEU reports. Usually a beam of about 4 to 8 with normalisation is best, and going wider gives no gain for extra cost. Report corpus-level BLEU with a fixed sacreBLEU signature so the comparison between settings is fair."
sources: [{"title": "The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)", "url": "https://arxiv.org/abs/1904.09751", "note": "Why likelihood-maximising decoding degenerates on open-ended text, and nucleus sampling."}, {"title": "BLEU: a Method for Automatic Evaluation of Machine Translation (Papineni et al., 2002)", "url": "https://aclanthology.org/P02-1040/", "note": "Clipped n-gram precision and the brevity penalty, defined in sections 2.1 to 2.3."}, {"title": "ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004)", "url": "https://aclanthology.org/W04-1013/", "note": "Definitions of ROUGE-N, ROUGE-L and their variants."}, {"title": "Deduplicating Training Data Makes Language Models Better (Lee et al., 2021)", "url": "https://arxiv.org/abs/2107.06499", "note": "Exact and MinHash near-duplicate removal at scale, and its effect on memorisation and train-test overlap."}]
practice: ["beam-search-decoding", "evaluation-metrics-bleu-rouge", "text-generation-sampling", "text-deduplication-minhash", "perform-stable-exact-deduplication"]
tags: ["beam search", "BLEU", "deduplication"]
tools: ["Python", "NumPy"]
---

## What you will learn

- How next-token probabilities become text, and why the **decoding** rule matters as much as the model.
- Greedy decoding and beam search, by hand on a tiny tree, and why beam search needs length normalisation.
- Why the most probable output is often not the best, and when to sample instead (temperature, top-k, top-p).
- How BLEU and ROUGE score outputs against references, computed by hand on one sentence pair.
- Why scores are only honest on deduplicated data, and how hashing and MinHash find duplicates.

*Before you start:* you need sequence models that output $P(y_t \mid y_{<t}, x)$ (lessons 05.03 and 05.04), log-probabilities and n-gram counts (06.02), and tokenisation (06.01). Lesson 06.06 built the models; this lesson uses their probabilities.

## From probabilities to text

A sequence model never writes a sentence. At each step it gives a probability for every vocabulary token, given the input $x$ and the tokens chosen so far. Producing text means picking a token, appending it, asking again, and stopping at the end token. The rule that picks is the decoding rule; the same weights with different rules give very different outputs. Scoring outputs then means comparing them with human-written **references**, which is only honest if test examples were not in the training data.

~~~dg-flow Every generated output passes through a decoding rule, and its score depends on references and on the test set being clean.
Model logits -> *Decoding rule (greedy, beam, sampling) -> Output text -(compare with references)-> BLEU / ROUGE
~~~

## The idea in one picture

A model is completing a sentence that starts with "the". It offers: "old" with probability 0.5, "cat" with 0.4 and "car" with 0.1. After "old" the options are fairly even (man 0.3, cat 0.3, house 0.4). After "cat" the model is sure: "sat" with 0.8.

~~~dg-fig 06.07-tree.svg
Greedy decoding grabs the best-looking first word (old, 0.5) and ends with a total probability of 0.20. Beam search keeps a second option alive and finds "the cat sat" at 0.32.
~~~

**Greedy decoding** takes the most likely token at each step: "old" (0.5), then "house" (0.4). The probability of the whole phrase is the product, $0.5 \times 0.4 = 0.20$.

But "the cat sat" has probability $0.4 \times 0.8 = 0.32$. Greedy missed it because "cat" looked slightly worse at step one. **Beam search** fixes much of this by keeping several partial sentences (**hypotheses**) alive at once. With a beam of width 2 it keeps both "old" and "cat" after step one, expands both, and finds the 0.32 path.

The catch: beam search is a better *search* for the most probable sequence, but whether that sequence is the *best* text is a separate question.

## Step 1: score whole sequences with log-probabilities

The probability of a sequence is the product of its step probabilities. Products of many small numbers underflow, so we add logs:

$$\text{score}(y) = \log P(y \mid x) = \sum_{t=1}^{|y|} \log P(y_t \mid y_{<t}, x).$$

Here $y$ is the output, $|y|$ its length in tokens, $y_t$ the token at position $t$, $y_{<t}$ the tokens before it, and $x$ the input. Every term is negative.

Number check on the tree: greedy's path scores $\log 0.5 + \log 0.4 = -0.693 - 0.916 = -1.609$, The better path scores $\log 0.4 + \log 0.8 = -0.916 - 0.223 = -1.139$. Higher (less negative) is better.

## Step 2: greedy and beam search

Greedy picks $\arg\max$ at every step: fast and deterministic, but myopic. Beam search keeps the $B$ highest-scoring partial hypotheses, where $B$ is the **beam width**. At each step:

1. Extend every live hypothesis by every vocabulary token, giving $B \times V$ candidates ($V$ is the vocabulary size).
2. Score each candidate with its running sum of log-probabilities.
3. Keep the top $B$. Any candidate that just produced the end token moves to a finished list instead.
4. Stop when no live hypothesis can beat the best finished one, or at a length limit.

With $B = 1$ it is exactly greedy. Typical widths are 4 to 5 for translation and summarisation; cost grows roughly linearly with $B$.

## Step 3: fix the length bias

Every extra token adds a negative number to the score, so hypotheses that end early pay fewer penalties. This is **length bias**.

The standard fix divides by a power of the length:

$$\text{score}_{\text{norm}}(y) = \frac{\log P(y \mid x)}{|y|^{\alpha}},$$

where $\alpha$ controls how strongly length is forgiven: $\alpha = 0$ is the raw score, $\alpha = 1$ is the average log-probability per token, and values between about 0.6 and 1.0 are common. Google's GNMT system used a smoothed version, dividing by $((5 + |y|)/6)^{\alpha}$.

<div class="ex">
<div class="lab">Worked example</div>

**length normalisation changes the winner**

A has 3 tokens and total log-probability $-2.4$; B has 6 tokens and $-3.6$.

Raw score: A wins, since $-2.4 > -3.6$.

With $\alpha = 1$: A scores $-2.4 / 3 = -0.80$ per token and B scores $-3.6 / 6 = -0.60$. B now wins: each of its tokens is more likely on average.

With $\alpha = 0.6$: $3^{0.6} = 1.933$, so A scores $-2.4 / 1.933 = -1.242$; $6^{0.6} = 2.930$, so B scores $-3.6 / 2.930 = -1.229$. B wins, but only just.

</div>

## Step 4: wider is not better

A wider beam does find higher-probability outputs. But for translation those are often shorter and worse, and at extreme widths the empty output can win outright. This is sometimes called the **beam search curse**.

To see it, take a toy model: random next-token probabilities over 40 words, and an end-token probability of 0.03 for the first 8 steps, 0.2 for steps 8 to 11, and 0.6 after. Natural outputs end after 8 to 12 tokens, but ending at once costs only $\log 0.03 = -3.5$, far less than any 10-word sentence. Beam search ran on 90 prompts at widths 1 to 50.

~~~dg-fig 06.07-length.svg
Toy experiment, 90 prompts. With the raw score (orange), outputs average about 8 tokens up to B = 5, almost half are empty at B = 10, and all are empty at B = 20. With alpha = 1 the length stays near 11 and nothing is empty.
~~~

Small beams never found the empty output because the end token was not among their top choices at step one; wide beams did, and the raw score rewarded it. Here $\alpha = 0.6$ barely helped and $\alpha = 1$ fixed it; on real models tune $\alpha$ and $B$ together on a dev set rather than turning $B$ up.

## Step 5: sample when variety is the point

For stories, dialogue and chat, greedy or beam decoding gives bland text that often loops ("I don't know. I don't know."). Holtzman et al. (2019) showed that human text is *not* the most probable text under a model: people often pick somewhat unlikely words. Sampling the next token by its probability restores that variety; three controls keep it sensible:

- **Temperature** $\tau$: sample from $\text{softmax}(z / \tau)$, where $z$ is the vector of raw scores (**logits**). $\tau < 1$ sharpens the distribution, $\tau > 1$ flattens it, and $\tau \to 0$ becomes greedy.
- **Top-k**: keep only the $k$ most likely tokens (say 40), renormalise, sample.
- **Top-p (nucleus)**: keep the smallest set of most-likely tokens whose probabilities sum to at least $p$ (say 0.9), renormalise, sample.

Number check with logits $(3.0, 2.2, 2.0, 1.5, 1.0, 0.6, -0.5, -2.0)$ for "sunny, cold, warm, nice, rainy, grey, odd, purple": "sunny" gets probability 0.707 at $\tau = 0.5$, 0.434 at $\tau = 1$ and 0.326 at $\tau = 1.5$. At $\tau = 1$ the cumulative sums are 0.434, 0.629, 0.789, 0.886, 0.945, so top-p 0.9 keeps five words and drops "grey", "odd" and "purple".

~~~dg-fig 06.07-sampling.svg
Left: temperature moves probability mass between head and tail. Middle and right: top-p adapts to the model's confidence (2 tokens kept when it is sure, 7 when it is not), while top-k keeps exactly 3 in both cases.
~~~

Top-p is usually preferred: when the model is confident, top-k = 3 still lets in a third token (probability 0.027) that top-p drops; when the model is uncertain, top-k = 3 throws away four options that each have more than 10% probability.

Rule of thumb: beam search (normalised) where there is roughly one right answer, such as translation or extraction; sampling where variety is the point.

## Step 6: score outputs with BLEU

**BLEU** (Papineni et al., 2002) compares a candidate translation with references by counting shared n-grams (runs of $n$ tokens). For $n$ = 1 to 4 it computes a **modified precision** $p_n$: candidate n-grams found in a reference divided by all candidate n-grams, with **clipping**: each n-gram is credited at most as often as it appears in a single reference. Without clipping, "the the the the the the the" against "the cat is on the mat" gets unigram precision 7/7; with clipping, 2/7.

Precision alone rewards short, safe outputs, so BLEU multiplies by a **brevity penalty**:

$$\text{BLEU} = \text{BP} \cdot \exp\Big(\sum_{n=1}^{4} \tfrac{1}{4} \log p_n\Big), \qquad \text{BP} = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}$$

with $c$ the candidate length and $r$ the reference length. The exponential of averaged logs is the geometric mean of the precisions.

<div class="ex">
<div class="lab">Worked example</div>

**BLEU by hand**

Reference: "the small cat is sitting on the mat" (8 tokens). Candidate: "the cat is sitting on the mat" (7 tokens).

Unigrams: all 7 candidate words appear in the reference ("the" is used twice and the reference has it twice), so $p_1 = 7/7 = 1$.

Bigrams: the candidate has "the cat, cat is, is sitting, sitting on, on the, the mat". Only "the cat" is missing from the reference (which says "small cat"), so $p_2 = 5/6$.

Trigrams: "the cat is" misses; "cat is sitting", "is sitting on", "sitting on the", "on the mat" match, so $p_3 = 4/5$.

4-grams: "the cat is sitting" misses; the other three match, so $p_4 = 3/4$.

Geometric mean: $(1 \times 0.8333 \times 0.8 \times 0.75)^{1/4} = 0.5^{1/4} = 0.841$.

Brevity penalty: $c = 7 \le r = 8$, so $\text{BP} = e^{1 - 8/7} = e^{-0.143} = 0.867$.

BLEU $= 0.867 \times 0.841 = 0.729$, often reported as 72.9.

</div>

~~~dg-fig 06.07-bleu.svg
Left: the four clipped precisions from the worked example and their geometric mean. Right: the brevity penalty is 1 at or above the reference length and drops steeply below it; dropping one word cost this candidate 13%.
~~~

BLEU is designed for the **corpus level**: sum match and total counts over all sentences, then combine once. Sentence-level BLEU needs smoothing, since one sentence with no matching 4-gram gives $p_4 = 0$ and BLEU 0. Scores also depend on tokenisation, so sacreBLEU fixes the tokeniser and prints a signature; never compare BLEU across tokenisers.

## Step 7: score summaries with ROUGE

**ROUGE** (Lin, 2004), built for summarisation, is recall-oriented: how much of the reference does the candidate *cover*?

- **ROUGE-N** is n-gram recall: matched n-grams divided by the number of n-grams in the reference. ROUGE-1 and ROUGE-2 are the common ones.
- **ROUGE-L** uses the **longest common subsequence** (LCS): the longest list of tokens that appears in both texts in the same order, not necessarily side by side. It is usually reported as an F-measure of LCS precision and recall.

On the same pair the whole candidate is a subsequence of the reference, so LCS = 7. Precision $7/7 = 1$, recall $7/8 = 0.875$, and $F = 2 \times 1 \times 0.875 / 1.875 = 0.933$.

Both metrics measure surface overlap: a correct paraphrase ("the kitten sits on the rug") can score poorly and a fluent wrong output can score well. Use them to track progress within one system and dataset, and pair them with human or model-based judges for decisions that matter.

## Decoding and evaluation in code

The sampler applies temperature, then top-k, then top-p:

```python
import numpy as np, math
from collections import Counter
rng = np.random.default_rng(0)

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    z = np.asarray(logits, float) / max(temperature, 1e-6)
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(p)[::-1]                       # most likely first
    keep = len(p) if top_k is None else top_k
    if top_p is not None:                             # smallest prefix with mass >= top_p
        keep = min(keep, int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1)
    q = np.zeros_like(p); q[order[:keep]] = p[order[:keep]]
    return rng.choice(len(p), p=q / q.sum())

logits = [3.0, 2.2, 2.0, 1.5, 1.0, 0.6, -0.5, -2.0]  # sunny, cold, warm, nice, ...
for kw in [dict(temperature=1e-6), dict(temperature=1.0), dict(top_k=3), dict(top_p=0.9)]:
    draws = [sample_next(logits, **kw) for _ in range(2000)]
    print(kw, np.bincount(draws, minlength=8))
# {'temperature': 1e-06} [2000    0    0    0    0    0    0    0]
# {'temperature': 1.0}   [ 886  390  315  208   97   74   22    8]
# {'top_k': 3}           [1080  487  433    0    0    0    0    0]
# {'top_p': 0.9}         [ 940  389  338  201  132    0    0    0]
```

BLEU with clipping and the brevity penalty, matching the hand example:

```python
def bleu(cand, ref, N=4):
    grams = lambda t, n: Counter(zip(*[t[i:] for i in range(n)]))
    logp = 0.0
    for n in range(1, N + 1):
        c, r = grams(cand, n), grams(ref, n)
        match = sum(min(k, r[g]) for g, k in c.items())          # clipped counts
        logp += math.log(match / max(sum(c.values()), 1)) / N    # breaks if match == 0
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * math.exp(logp)

ref  = "the small cat is sitting on the mat".split()
cand = "the cat is sitting on the mat".split()
print(round(bleu(cand, ref), 4))                                 # 0.729

```

This is a teaching version for one sentence and one reference (ROUGE-L is a similar short dynamic program over an LCS table); for reported numbers use sacreBLEU at corpus level and a maintained ROUGE package.

## Step 8: keep the test set clean with deduplication

Web corpora are full of repeats. Lee et al. (2021) found substantial train-test overlap in standard datasets and showed that training on deduplicated data made models emit memorised text far less often, with no loss in quality. Duplicates also waste compute.

- **Exact duplicates**: normalise each document (lowercase, collapse whitespace), hash it (for example `hashlib.sha1(" ".join(doc.lower().split()).encode())`), and keep the first occurrence of each hash. One pass, linear time, original order preserved.
- **Near duplicates** differ by a date, header or a few words. Represent each document by its set of **shingles** (overlapping word or character n-grams, e.g. 5-grams) and measure overlap with the **Jaccard similarity** $J(A, B) = |A \cap B| / |A \cup B|$.

**MinHash** estimates Jaccard cheaply: hash every shingle with a random hash function and keep the minimum. Two documents' minima are equal with probability exactly $J$, so with about 100 independent hash functions the fraction of matching minima estimates $J$.

To avoid comparing all pairs of signatures, **locality-sensitive hashing (LSH)** splits a 100-value signature into $b$ bands of $r$ rows. Two documents become **candidates** if all $r$ values match in at least one band, which happens with probability $1 - (1 - s^r)^b$ for Jaccard $s$. Only candidates are checked exactly.

~~~dg-fig 06.07-minhash.svg
Left: banding turns similarity into an S-shaped yes/no filter; more rows per band moves the threshold right. Right: over 60 synthetic document pairs, 100-hash MinHash estimates sit close to the true Jaccard (mean absolute error 0.03).
~~~

With $b = 20$, $r = 5$: at $s = 0.8$ one band matches with probability $0.8^5 = 0.328$, so the pair is a candidate with probability $1 - (1 - 0.328)^{20} = 0.9996$; at $s = 0.3$ only 5% of the time. Deduplicate before splitting into train and test, or explicitly check the test set against the training set.

## Reading the result

- **Sweep beam width and $\alpha$ together** on one dev set, plotting BLEU next to the length ratio $c/r$ (sacreBLEU prints it). A BLEU drop with shorter outputs is length bias, not a worse model.
- **For sampled text, read samples and measure diversity**: the fraction of distinct n-grams exposes loops that a single score hides.
- **Audit overlap**: if scores fall sharply once test items found in training data are removed, they were inflated.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Outputs get shorter as beam width grows | Length bias in the raw log-probability score | Length normalisation, tune $B$ and $\alpha$ on a dev set
| Empty outputs at wide beams | Search found the high-probability early end | Normalise, add a minimum length, keep $B$ around 4 to 8
| Chatbot repeats the same sentence | Greedy or beam decoding on open-ended text | Sample with top-p around 0.9 and moderate temperature
| Sentence BLEU is 0 for a decent output | No matching 4-gram, so the geometric mean collapses | Report corpus BLEU, or use smoothing for sentence scores
| BLEU differs from a paper's number | Different tokeniser or casing | Use sacreBLEU and compare signatures
| Test scores suspiciously high | Test examples duplicated in training data | Exact and MinHash deduplication across the split

## Going deeper

**Search error versus model error.** A bad output comes either from the search missing the model's best sequence (search error) or from the model's best being bad (model error). Exact-search studies on translation models (Stahlberg and Byrne, 2019) found the true most probable output is very often the empty sentence, so beam search works partly *because* it fails to find the mode. Minimum Bayes risk decoding (pick the sample most similar on average to the other samples) targets quality more directly.

**Why truncated sampling works.** Each tail token is unlikely, but the long tail together holds noticeable mass, so plain sampling eventually picks a bad token and the error compounds. Top-p cuts the tail adaptively.

**MinHash correctness.** Under a random permutation of shingles, the first element of $A \cup B$ is equally likely to be any of its elements, and the two minima agree exactly when it lies in $A \cap B$: probability $|A \cap B| / |A \cup B|$. With $m$ hashes the estimate's standard deviation is $\sqrt{J(1 - J)/m}$, about 0.05 at $J = 0.5$, $m = 100$. The LSH threshold, where the S-curve is steepest, sits near $(1/b)^{1/r}$: about 0.55 for $b = 20$, $r = 5$.

## Key terms

| Term | Meaning
| --- | --- |
| Decoding rule | How tokens are chosen from the model's next-token probabilities
| Greedy decoding | Always take the single most likely next token
| Beam search | Keep the $B$ best partial sequences by total log-probability at every step
| Length normalisation | Divide the score by $|y|^{\alpha}$ so short outputs are not unfairly favoured
| Temperature | Divide logits by $\tau$ before softmax: below 1 sharpens, above 1 flattens
| Top-p (nucleus) | Sample only from the smallest set of tokens holding probability mass $p$
| BLEU | Geometric mean of clipped 1- to 4-gram precisions times a brevity penalty
| ROUGE-L | Overlap based on the longest common subsequence, reported as an F-measure
| MinHash | Signature whose matching rate between two documents estimates Jaccard similarity

<div class="take">
<div class="lab">Key takeaway</div>

the decoding rule shapes the output as much as the model does: beam search with length normalisation for single-answer tasks, truncated sampling for open-ended ones, and never assume the most probable sequence is the best. BLEU and ROUGE measure n-gram overlap under fixed tokenisation, and they are only honest on deduplicated data.

</div>

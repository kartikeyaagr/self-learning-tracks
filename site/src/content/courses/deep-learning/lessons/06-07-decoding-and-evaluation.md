---
id: "06.07"
course: "deep-learning"
module: "06"
slug: "decoding-and-evaluation"
title: "Decoding and evaluation"
keyIdeas: ["Greedy decoding takes the most likely token at each step; beam search keeps the $B$ best partial sequences by total log-probability and approximates the most likely complete output.", "Summed log-probabilities favour short outputs, so beam search needs length normalisation. Very wide beams often lower quality: the most probable sequence is not the best one.", "For open-ended generation, likelihood-maximising search produces dull, repetitive text; sampling with temperature, top-k or top-p (nucleus) is the norm.", "BLEU is clipped n-gram precision (n = 1 to 4, geometric mean) times a brevity penalty, computed at corpus level; ROUGE is recall-oriented overlap, with ROUGE-L based on the longest common subsequence. Both measure surface overlap, not meaning.", "Deduplicate training data, exactly by hash and approximately with MinHash and LSH. Duplicates waste compute, increase memorisation, and leak test examples into training so scores are inflated."]
question: "You raise the beam width of a translation model from 5 to 50. Model log-probability of the outputs goes up, but BLEU goes down and outputs get noticeably shorter. Explain why, and what you would change."
answer: "Beam search maximises the total log-probability, a sum of negative terms, so every extra token lowers the score; short hypotheses that end early have an advantage. A narrow beam rarely finds those short, high-probability hypotheses, but a 50-wide beam explores enough to find them. It is succeeding at search while exposing a mismatch between the model's probability and translation quality. BLEU drops because the outputs omit content (lower n-gram matches) and because the brevity penalty e^(1 - r/c) applies once candidates are shorter than the references. Fixes: add length normalisation (divide by length to the power α, about 0.6-1.0, or GNMT's penalty), possibly a coverage penalty, and tune both the beam width and α on a dev set; often a beam of 4-8 with normalisation is best."
sources: [{"title": "The Curious Case of Neural Text Degeneration (Holtzman et al., 2019)", "url": "https://arxiv.org/abs/1904.09751", "note": "Why likelihood-maximising decoding degenerates on open-ended text, and nucleus sampling."}, {"title": "BLEU: a Method for Automatic Evaluation of Machine Translation (Papineni et al., 2002)", "url": "https://aclanthology.org/P02-1040/", "note": "Clipped n-gram precision and the brevity penalty, defined in sections 2.1 to 2.3."}, {"title": "ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004)", "url": "https://aclanthology.org/W04-1013/", "note": "Definitions of ROUGE-N, ROUGE-L and their variants."}, {"title": "Deduplicating Training Data Makes Language Models Better (Lee et al., 2021)", "url": "https://arxiv.org/abs/2107.06499", "note": "Exact and MinHash near-duplicate removal at scale, and its effect on memorisation and train-test overlap."}]
practice: ["beam-search-decoding", "evaluation-metrics-bleu-rouge", "text-generation-sampling", "text-deduplication-minhash", "perform-stable-exact-deduplication"]
tags: ["beam search", "BLEU", "deduplication"]
tools: ["Python", "NumPy"]
---

## From probabilities to text

A sequence model (05.03, 05.04) gives $P(y_t \mid y_{<t}, x)$. Producing text means choosing tokens one at a time, and the choice rule matters as much as the model. This lesson covers the rules, the classic metrics for scoring the result, and the data hygiene that keeps those scores honest. Sampling mechanics for LLM serving (temperature, top-k, top-p, speculative decoding) are covered in more depth in the Inference Engineering course; here the focus is on search and evaluation.

~~~dg-flow Every generated output passes through a decoding rule, and its score depends on references and on the test set being clean.
Model logits -> *Decoding rule (greedy, beam, sampling) -> Output text -(compare with references)-> BLEU / ROUGE
~~~

## Greedy and beam search

Greedy decoding picks $\arg\max$ at every step. It is fast and deterministic but myopic: a token that looks best now can lead to a poor continuation.

Beam search keeps the $B$ highest-scoring partial hypotheses. At each step it expands each by every vocabulary token, scores by cumulative log-probability $\sum_t \log P(y_t \mid y_{<t}, x)$, and keeps the top $B$ of the $B \times V$ candidates. Hypotheses that emit the end token move to a finished list; search stops when no live beam can beat the best finished one, or at a length limit. Typical widths are 4 to 5 for translation and summarisation.

Two known problems:

- **Length bias.** Every token adds a negative log-probability, so shorter hypotheses score higher. Normalise by length, $\text{score} / |y|^{\alpha}$ with $\alpha$ between about 0.6 and 1.0 (GNMT used $((5 + |y|)/6)^{\alpha}$).
- **Wider is not better.** Raising $B$ finds higher-probability outputs, but for translation these are often shorter and worse; at extreme widths the empty output can win. The model's highest-probability sequence is not the best translation, so beam width is a hyperparameter to tune on a dev set, not a quality knob to max out.

## Sampling for open-ended text

For stories, dialogue and chat, maximising likelihood produces generic and repetitive text that falls into loops. Holtzman et al. (2019) showed that human text is not the most probable text under a model: it regularly contains lower-probability tokens. Sampling restores that variety, with controls on the tail:

- **Temperature** $\tau$: sample from $\text{softmax}(z / \tau)$. $\tau < 1$ sharpens, $\tau > 1$ flattens, $\tau \to 0$ approaches greedy.
- **Top-k**: sample only among the $k$ most likely tokens (e.g. 40).
- **Top-p (nucleus)**: sample from the smallest set of tokens whose probabilities sum to at least $p$ (e.g. 0.9). The set adapts: a few tokens when the model is confident, many when it is not.

Use beam search where there is roughly one right answer (translation, extraction), and sampling where variety is the point.

## BLEU

BLEU (Papineni et al., 2002) scores a candidate translation against one or more references. For each $n$ from 1 to 4 it computes a modified precision: n-gram matches divided by candidate n-grams, where each n-gram's count is clipped at the most times it appears in any single reference. The score is

$$\text{BLEU} = \text{BP} \cdot \exp\Big(\sum_{n=1}^{4} \tfrac{1}{4} \log p_n\Big), \qquad \text{BP} = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}$$

with $c$ the total candidate length and $r$ the reference length. Precision alone rewards short, safe outputs, so the brevity penalty punishes outputs shorter than the reference.

<div class="ex">
<div class="lab">Worked example</div>

**clipping and the brevity penalty**

Candidate "the the the the the the the" against reference "the cat is on the mat": unclipped unigram precision is 7/7; "the" appears at most twice in the reference, so the clipped precision is 2/7.

A 10-token candidate against a 12-token reference gets $\text{BP} = e^{1 - 12/10} = e^{-0.2} = 0.82$, an 18% penalty regardless of quality.

</div>

```python
from collections import Counter
def clipped_precision(cand, refs, n):
    grams = lambda t: Counter(zip(*[t[i:] for i in range(n)]))
    c = grams(cand)
    max_ref = Counter()
    for r in refs: max_ref |= grams(r)                  # elementwise max over references
    return sum(min(k, max_ref[g]) for g, k in c.items()), max(sum(c.values()), 1)
```

BLEU is designed for corpus level: sum matches and totals over all sentences, then combine. Sentence-level BLEU is noisy and needs smoothing, since a single missing 4-gram makes the geometric mean zero. Scores also depend on tokenisation, which is why sacreBLEU standardises it and prints a signature; never compare BLEU numbers computed with different tokenisers.

## ROUGE

ROUGE (Lin, 2004) was built for summarisation and is recall-oriented: how much of the reference does the candidate cover? ROUGE-N is n-gram recall (ROUGE-1, ROUGE-2). ROUGE-L uses the longest common subsequence, which rewards in-order matches without requiring them to be contiguous, and is usually reported as an F-measure combining LCS-based precision and recall.

Both metrics measure surface overlap. A correct paraphrase can score poorly and a fluent but wrong output can score well. Use them to track progress within one system and dataset, and pair them with human evaluation or model-based judges for decisions that matter.

## Deduplication

Evaluation is only meaningful if test examples were not in the training data, and web-scale corpora are full of repeats. Lee et al. (2021) found substantial train-test overlap in standard datasets and showed that deduplicated training data led models to emit memorised text far less often, with no loss in quality.

- **Exact duplicates**: normalise (lowercase, collapse whitespace), hash each document, keep the first occurrence of each hash. Linear time, stable order.
- **Near duplicates**: represent each document by its set of word or character shingles (e.g. 5-grams) and estimate Jaccard similarity with MinHash: the probability that two documents share the minimum hash under a random hash function equals their Jaccard similarity. With LSH, split a 100-value signature into $b$ bands of $r$ rows; two documents become candidates if any band matches exactly, which happens with probability $1 - (1 - s^r)^b$.

With $b = 20$, $r = 5$: documents with Jaccard 0.8 become candidates with probability 0.9996; documents at 0.3 only 0.05. Candidates are then checked exactly. Deduplicate before splitting into train and test, or check the test set against the training set explicitly.

<div class="take">
<div class="lab">Key takeaway</div>

the decoding rule shapes the output as much as the model does: beam search with length normalisation for single-answer tasks, sampling for open-ended ones. BLEU and ROUGE measure n-gram overlap under fixed tokenisation, and they are only honest on deduplicated data.

</div>

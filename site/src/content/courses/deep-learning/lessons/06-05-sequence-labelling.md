---
id: "06.05"
course: "deep-learning"
module: "06"
slug: "sequence-labelling"
title: "Sequence labelling"
keyIdeas: ["Sequence labelling assigns one label per token. Part-of-speech tags are single-token labels; named entities use the **BIO** scheme (B- begins an entity, I- continues it, O is outside) so that multi-word spans fit one label per token.", "Greedy per-token argmax can output invalid sequences such as O I-PER or B-LOC I-PER. The **Viterbi** algorithm finds the best valid sequence exactly in $O(TK^2)$ by keeping, for every position and tag, the best path score plus a back-pointer.", "A **CRF** layer learns a $K \\times K$ matrix of transition scores and trains on whole-sequence likelihood; structured decoding helps most when the model is uncertain, and constrained decoding can be added to any trained tagger for free.", "With subword tokenisers, label only the first subword of each word, give the rest the ignore label -100, and map predictions back to words through word ids and character offsets.", "Evaluate NER with entity-level F1 (exact span and type match), not token accuracy, which mostly counts easy O tokens. Extractive QA is span labelling: score start and end per token and pick the best pair with start ≤ end inside the passage."]
question: "A NER model's per-token argmax output for five words is: O, B-LOC, I-PER, I-PER, O. How would a lenient BIO decoder, a strict decoder, and Viterbi decoding with transition constraints each handle this, and how does each outcome affect entity-level F1 if the gold answer is a single PER entity over words 2-4?"
answer: "A lenient decoder closes LOC at word 2 and starts a new PER entity at word 3 from the stray I-PER, giving LOC(2) and PER(3-4). A strict decoder keeps LOC(2) and discards the I-PER tokens, since they lack a B-PER, giving only LOC(2). Viterbi with constraints forbids B-LOC followed by I-PER (and O followed by I-PER) and searches for the best *valid* sequence using the model's scores: if the PER evidence on words 3-4 is strong, the best valid option is usually B-PER I-PER I-PER on words 2-4, paying a small penalty for overriding the LOC score on word 2; if the LOC evidence on word 2 dominates, it may choose B-LOC O O, or B-LOC B-PER I-PER. Against the gold PER(2-4): lenient produces two predicted entities, neither exactly matching, so 2 false positives and 1 false negative (P = 0, R = 0, F1 = 0). Strict produces 1 false positive and 1 false negative (also F1 = 0). Only the constrained decode can produce the exact span PER(2-4), giving P = R = F1 = 1 for this sentence; its other outcomes again score zero for this entity. Note that token accuracy for the greedy output is 4 of 5 words correct (80%) even though entity F1 is zero. The general lesson: structured decoding turns per-token evidence into consistent spans, and should share its stray-I policy with the evaluation script; if invalid outputs like this are common, the model is uncertain and a CRF layer or more in-domain data is worth considering."
sources: [{"title": "Neural Architectures for Named Entity Recognition (Lample et al., 2016)", "url": "https://arxiv.org/abs/1603.01360", "note": "BiLSTM-CRF for NER and the IOBES scheme; section 2 explains the CRF layer and Viterbi decoding."}, {"title": "Bidirectional LSTM-CRF Models for Sequence Tagging (Huang et al., 2015)", "url": "https://arxiv.org/abs/1508.01991", "note": "Compares LSTM, BiLSTM and BiLSTM-CRF taggers on POS, chunking and NER."}, {"title": "Introduction to the CoNLL-2003 Shared Task", "url": "https://aclanthology.org/W03-0419/", "note": "Defines the standard NER dataset and the exact-match entity F1 evaluation."}, {"title": "SQuAD: 100,000+ Questions for Machine Comprehension", "url": "https://arxiv.org/abs/1606.05250", "note": "The extractive QA task framing that start/end span prediction solves."}]
practice: ["named-entity-recognition", "greedy-sequence-tagging", "token-classification", "span-extraction", "pos-tagging"]
tags: ["NER", "BIO tagging", "Viterbi"]
tools: ["Python", "NumPy"]
---

## What you will learn

- What sequence labelling is and the tasks built on it: POS tagging, named-entity recognition, extractive question answering.
- How the BIO scheme gives multi-word entities one label per token, and how to turn tags back into spans.
- Why greedy per-token argmax can produce invalid output, and how the Viterbi algorithm fixes it (worked by hand).
- What a CRF layer adds, and how to align word labels with subword tokens.
- How to measure a tagger with entity-level F1, and how start/end scoring turns QA into span labelling.

*Before you start:* you need softmax and cross-entropy (module 02), recurrent encoders and bidirectional LSTMs (05.02), subword tokenisation and character offsets (06.01), and word vectors (06.04). Dynamic programming is explained from scratch here.

## From word vectors to word labels

Lesson 06.04 gave every word a dense vector. So far such vectors fed classifiers that label a *whole* text. Many tasks need a decision for *each* word instead. The models are ordinary classifiers applied at every position; the twist is that neighbouring labels depend on each other, and ignoring that produces broken output.

## The idea in one picture

Take the sentence "Barack Obama visited New York yesterday". We want to mark the person and the place. A tagger reads the sentence and outputs, for every word, a probability for each possible label:

| Word | Barack | Obama | visited | New | York | yesterday
| --- | --- | --- | --- | --- | --- | --- |
| Gold tag | B-PER | I-PER | O | B-LOC | I-LOC | O

~~~dg-fig 06.05-tagprobs.svg
A tagger's output for six words. Each column is one softmax over the five tags; the outlined cell is the most likely tag for that word. Reading the outlined cells left to right spells out the two entities.
~~~

**Sequence labelling** (also called **token classification**) means: one input sequence, one label per token, same length in and out. The tags in the table use the **BIO scheme**: **B-PER** means "the first word of a person name", **I-PER** means "inside (continuing) a person name", and **O** means "outside any entity". Grouping the tags back together gives two **entities** (labelled spans of text): PER over words 0-1 and LOC over words 3-4.

Here the most likely tag per word is right. The rest of the lesson is about when it is not.

## Step 1: three tasks, one model shape

Three common tasks share this form:

- **Part-of-speech (POS) tagging**: noun, verb, adjective and so on. Universal Dependencies uses 17 coarse tags; the Penn Treebank uses 45 finer ones.
- **Named-entity recognition (NER)**: find spans naming people, organisations, places, dates, drugs, product codes.
- **Span extraction**: find the answer to a question as a contiguous span of a passage.

They share one model shape. An **encoder** produces a context-aware vector $h_t$ for each token $t$ (a bidirectional LSTM from 05.02, or a pretrained transformer from the next lesson). A linear layer maps each vector to $K$ scores, one per tag, and a decoding step turns scores into tags and spans.

~~~dg-flow The token-classification pipeline. Decoding and evaluation happen at the word or span level, not the subword level.
Words -(tokeniser, keep offsets)-> Subwords -> Encoder -> *Per-token logits -(argmax or Viterbi)-> BIO tags -(group)-> Entity spans
~~~

In symbols, for token $t$ the scores and probabilities are

$$z_t = W h_t + b, \qquad p(y_t = k \mid x) = \frac{e^{z_{t,k}}}{\sum_{j=1}^{K} e^{z_{t,j}}},$$

where $x$ is the whole sentence, $h_t$ is the encoder's vector for position $t$ (it can depend on every word, which is why a bidirectional encoder helps), $W$ is a $K \times d$ weight matrix shared by all positions, and $z_{t,k}$ is the score (**logit**) of tag $k$ at position $t$. Training minimises cross-entropy summed over labelled positions.

How good is "good"? For English POS tagging, the dumbest reasonable baseline (give each word its most frequent training tag, and unknown words the most common open-class tag) already reaches about 92% token accuracy on newswire; strong taggers reach about 97% (standard published figures).

## Step 2: tagging schemes for multi-word entities

An entity can cover several tokens, so a label has to say not just *what* type but *where in the entity* the token sits. That is the job of BIO: B-X begins an entity of type X, I-X continues it, O is outside.

The B tag separates adjacent entities of the same type: in the headline "Paris London rail strike", B-LOC B-LOC marks two cities, while B-LOC I-LOC would merge them.

**BIOES** (or BILOU) adds E-X for the last token of a multi-token entity and S-X for a single-token one; the explicit boundary signal sometimes helps a little. Neither scheme can represent **nested entities**, such as the location "England" inside the organisation "Bank of England"; those need span-based models that score every candidate span directly.

Turning tags into spans needs a rule for misfits such as I-PER right after O. A **lenient** decoder starts a new entity at a stray I-X; a **strict** one drops it. Either is fine if conversion, decoding and evaluation all use the same rule.

## Step 3: why per-token argmax breaks

The simplest decoder is **greedy**: at each position pick the tag with the highest score, $\hat y_t = \arg\max_k z_{t,k}$. It is fast, but each decision ignores its neighbours, so the result can violate the grammar of BIO. An I-PER after O, or a B-LOC followed by I-PER, is not a valid sequence.

This happens when the model is unsure: rare names, words that can be a person or a place ("Jordan"), early training. The fix is to choose the whole sequence at once:

$$\hat y_{1:T} = \arg\max_{\text{valid } y_{1:T}} \sum_{t=1}^{T} s_t(y_t),$$

where $s_t(k)$ is the score of tag $k$ at position $t$ (use log-probabilities, so summing scores multiplies probabilities) and "valid" means no forbidden pair of neighbouring tags. Trying every sequence is hopeless: with $K$ tags and $T$ tokens there are $K^T$ of them, about $10^{11}$ for 7 tags and 13 words. Dynamic programming solves it exactly in a fraction of that time.

## Step 4: the Viterbi algorithm

The **Viterbi algorithm** walks left to right and keeps one number per (position, tag) cell:

$$\delta_t(k) = \text{best total score of any valid tag sequence for tokens } 1..t \text{ that ends in tag } k.$$

The update says: to end in tag $k$ at position $t$, come from the best allowed predecessor $j$, then add this position's score:

$$\delta_t(k) = s_t(k) + \max_{j} \big[\delta_{t-1}(j) + A_{jk}\big].$$

Here $A_{jk}$ is the **transition score** for moving from tag $j$ to tag $k$. For pure constraints it is $0$ for allowed pairs and $-\infty$ for forbidden ones (O followed by I-PER, for example). Alongside each $\delta_t(k)$ we store a **back-pointer**: which $j$ won the max. At the end, take the best final cell and follow back-pointers to recover the whole path. The start is handled the same way: a sentence may not begin with an I- tag.

<div class="ex">
<div class="lab">Worked example</div>

**Viterbi by hand on "told Van Dyke"**

Three tags, O, B-PER, I-PER. The tagger's scores per word (O, B-PER, I-PER) are: told $(2.0, 0.0, 0.0)$, Van $(0.5, 1.0, 1.2)$, Dyke $(0.0, 0.2, 1.5)$. "Van" looks like a name continuation, so its highest score is I-PER.

Greedy argmax picks O, I-PER, I-PER with total $2.0 + 1.2 + 1.5 = 4.7$. But I-PER right after O is forbidden, so this is invalid.

Position 1 (told): $\delta_1 = (2.0,\ 0.0,\ -\infty)$, since a sentence cannot start with I-PER.

Position 2 (Van): O can follow anything, best predecessor is O: $\delta_2(\text{O}) = 2.0 + 0.5 = 2.5$. B-PER can follow anything: $\delta_2(\text{B}) = 2.0 + 1.0 = 3.0$ (from O). I-PER may only follow B-PER or I-PER: $\max(0.0, -\infty) + 1.2 = 1.2$ (from B-PER).

Position 3 (Dyke): $\delta_3(\text{O}) = \max(2.5, 3.0, 1.2) + 0.0 = 3.0$. $\delta_3(\text{B}) = 3.0 + 0.2 = 3.2$. $\delta_3(\text{I}) = \max(\delta_2(\text{B}), \delta_2(\text{I})) + 1.5 = \max(3.0, 1.2) + 1.5 = 4.5$, from B-PER.

The best final cell is I-PER with 4.5. Back-pointers: I-PER at Dyke came from B-PER at Van, which came from O at told. Answer: O, B-PER, I-PER, the entity "Van Dyke".

The valid path scores 0.2 less than the invalid greedy one (4.5 versus 4.7): the cheapest price of obeying the rules.

</div>

~~~dg-fig 06.05-viterbi.svg
The worked example as a trellis. Each node holds δ, the best valid score so far; grey arrows are back-pointers. The highlighted path is Viterbi's answer. The dashed greedy path jumps from O straight to I-PER, which no back-pointer allows.
~~~

Each position compares $K$ predecessors for each of $K$ tags, so the cost is $O(T K^2)$: about 600 additions for 13 words and 7 tags, versus $10^{11}$ sequences.

## Step 5: learn the transitions with a CRF

Hard constraints only forbid impossible pairs. A **conditional random field (CRF)** layer *learns* a full $K \times K$ transition matrix $A$ (for example, that two PER entities rarely touch) and trains on the score of the whole sequence:

$$\text{score}(x, y) = \sum_{t=1}^{T} z_{t, y_t} + \sum_{t=2}^{T} A_{y_{t-1} y_t}, \qquad p(y \mid x) = \frac{e^{\text{score}(x, y)}}{\sum_{y'} e^{\text{score}(x, y')}}.$$

The numerator rewards the gold sequence; the denominator $Z(x)$ sums over every possible sequence and is computed with the **forward algorithm**, which is Viterbi with "max" replaced by "log-sum-exp". The loss is $\log Z(x) - \text{score}(x, y_{\text{gold}})$. At test time Viterbi finds the best sequence under the learned $A$. BiLSTM-CRF was the standard NER architecture before transformers (Huang et al., 2015; Lample et al., 2016).

How much does this help? The next figure trains a small BiLSTM tagger on a synthetic NER task (1,500 training sentences; 500 test sentences in which about half the entity names are unseen, so context matters), decodes it three ways, and trains a BiLSTM-CRF alongside.

~~~dg-fig 06.05-training.svg
Left: entity F1 on the test set while training. Right: share of test sentences where greedy argmax emits a forbidden tag pair. Constrained Viterbi never does, by construction, and it is ahead while the model is still unsure (steps 60-130); once the model is confident, all four methods converge.
~~~

After 80 mini-batches, greedy decoding reaches entity F1 0.40 (lenient grouping) or 0.44 (strict), constrained Viterbi on the same model 0.57, and the BiLSTM-CRF 0.58; a quarter of greedy outputs contain an invalid pair. By step 300 all sit at about 0.96. Structure matters most when the model is uncertain: small datasets, new domains, rare types. Constrained decoding costs nothing to add, so use it. The CRF still emits a few invalid pairs early because its transitions are learned, not hard-coded.

## Step 6: words versus subword tokens

A transformer tokeniser (06.01) splits words into pieces: "Kaczynski" might become "Ka", "##czy", "##nski". Labels, however, belong to words. The standard recipe:

1. Keep the tokeniser's word ids (which word each subword came from).
2. Give each word's label to its **first** subword.
3. Give later subwords and special tokens ([CLS], [SEP]) the ignore label **-100**, which PyTorch's cross-entropy skips by default.
4. At prediction time, read each word's tag from its first subword; character offsets map spans back to the string.

| Token | [CLS] | Ann | Ka | ##czy | ##nski | visited | Oslo | [SEP]
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Word id | none | 0 | 1 | 1 | 1 | 2 | 3 | none
| Training label | -100 | B-PER | I-PER | -100 | -100 | O | B-LOC | -100

Getting this wrong shifts every label by one or more positions. Nothing crashes; accuracy simply collapses, so decode a few training examples back to words and eyeball them first.

## Step 7: question answering as span labelling

In extractive question answering (SQuAD-style), the input is `[CLS] question [SEP] passage [SEP]` and the answer is a span of the passage. Instead of BIO tags, the model outputs two scores per token: a start score $s_i$ ("the answer begins at token $i$") and an end score $e_j$ ("the answer ends at token $j$"). The answer is the pair

$$(\hat i, \hat j) = \arg\max_{i \le j,\ j - i < L_{\max}} \; s_i + e_j,$$

with both positions inside the passage, not the question, and $L_{\max}$ a length cap such as 30 tokens.

~~~dg-fig 06.05-span.svg
Start and end scores for "when did it open?" over a short passage. The highest start score is on "1950" (3.4) but the highest end score is on "1937" (3.6), which comes before it. Searching valid pairs instead picks "may 1937" with total 3.2 + 3.6 = 6.8.
~~~

Independent argmaxes can put the end before the start, as here; searching valid pairs costs only $O(T \cdot L_{\max})$. For unanswerable questions (SQuAD 2.0), pointing both at [CLS] means "no answer": answer only if the best span beats the null score $s_0 + e_0$ (here 0.8) by a tuned margin.

## Sequence labelling in code

The core pieces fit in a few dozen lines of NumPy: span grouping, constrained Viterbi and entity-level scoring.

```python
import numpy as np

def bio_to_spans(tags, strict=False):
    # BIO tags -> [(start, end_exclusive, type)]
    spans, start, typ = [], None, None
    for i, t in enumerate(tags + ["O"]):              # sentinel closes a trailing span
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                spans.append((start, i, typ))
            start, typ = None, None
            if t.startswith("B-") or (t.startswith("I-") and not strict):
                start, typ = i, t[2:]                 # lenient: a stray I- opens a span
    return spans

def allowed(prev, cur):                               # I-X may only follow B-X or I-X
    return not cur.startswith("I-") or prev in ("B-" + cur[2:], cur)

def viterbi(E, tags):
    # E: (T, K) per-token scores (log-probs). Returns the best *valid* sequence.
    T_, K = E.shape
    trans = np.array([[0.0 if allowed(p, c) else -np.inf for c in tags] for p in tags])
    delta = E[0] + np.array([-np.inf if c.startswith("I-") else 0.0 for c in tags])
    back = np.zeros((T_, K), dtype=int)
    for t in range(1, T_):
        cand = delta[:, None] + trans                 # (prev, cur)
        back[t] = cand.argmax(0)
        delta = cand.max(0) + E[t]
    path = [int(delta.argmax())]
    for t in range(T_ - 1, 0, -1):
        path.append(back[t, path[-1]])
    return [tags[i] for i in path[::-1]], float(delta.max())

def entity_prf(gold_tags, pred_tags):
    g, p = set(bio_to_spans(gold_tags)), set(bio_to_spans(pred_tags))
    tp = len(g & p)
    P, R = tp / max(len(p), 1), tp / max(len(g), 1)
    return P, R, 2 * P * R / max(P + R, 1e-12)

tags = ["O", "B-PER", "I-PER"]
E = np.array([[2.0, 0.0, 0.0],      # told
              [0.5, 1.0, 1.2],      # Van
              [0.0, 0.2, 1.5]])     # Dyke
greedy = [tags[i] for i in E.argmax(1)]
print(greedy)                        # ['O', 'I-PER', 'I-PER']   <- invalid
print(viterbi(E, tags))              # (['O', 'B-PER', 'I-PER'], 4.5)
print(bio_to_spans(greedy), bio_to_spans(greedy, strict=True))   # [(1, 3, 'PER')] []

gold, pred = ["O", "B-PER", "I-PER"], ["O", "B-PER", "O"]
print(entity_prf(gold, pred))        # (0.0, 0.0, 0.0)  although 2 of 3 tokens are right
```

Lenient grouping happens to rescue this greedy output; strict finds nothing; with a type clash such as B-LOC I-PER only Viterbi can help. In real projects, Hugging Face tokenisers expose `word_ids()` and `offset_mapping` for alignment and `seqeval` computes CoNLL-style entity scores.

## Checking the result: entity-level F1

Token accuracy is the wrong headline for NER: most tokens are O, so a model that finds nothing still looks good.

~~~dg-fig 06.05-metrics.svg
Five simulated taggers on a 2,000-token test set where 82% of tokens are O and there are 202 entities. Predicting O everywhere scores 0.82 token accuracy and 0 entity F1. Off-by-one boundaries on 40% of entities barely dent accuracy (0.96) but cut F1 to 0.59.
~~~

The CoNLL standard is **entity-level precision, recall and F1**: a predicted entity counts as correct only if its start, end and type all match a gold entity exactly.

<div class="ex">
<div class="lab">Worked example</div>

**token accuracy versus entity F1**

A test set has 10 gold entities. The model predicts 8, of which 6 match exactly; the other 2 have one boundary token wrong.

Precision $= 6/8 = 0.75$, recall $= 6/10 = 0.60$.

F1 $= 2 \times 0.75 \times 0.60 / (0.75 + 0.60) = 0.9 / 1.35 = 0.667$.

Token accuracy could still be 97%, because the two near-misses are each a single wrong token among hundreds of correct O tags. Each near-miss counts twice against F1: once as a false positive (the wrong span) and once as a false negative (the missed gold span).

</div>

Beyond the headline, report F1 **per entity type** (the rare type you care about may sit far below the average), split errors into missed, spurious, wrong-type and wrong-boundary, and read 20 errors by hand: many turn out to be inconsistent gold annotation.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| High token accuracy, low entity F1 | Metric dominated by O tokens | Evaluate entity-level F1 per type
| Output contains O I-PER or B-LOC I-PER | Greedy argmax ignores neighbours | Constrained Viterbi or a CRF layer
| F1 near zero after switching to a transformer | Subword labels misaligned with words | Use word ids, label first subword, -100 the rest
| Different F1 from your script and a colleague's | Different stray-I or BIO/BIOES policy | Use one shared decoder and `seqeval`-style scoring
| QA answer ends before it starts or lies in the question | Independent argmax of start and end | Search valid pairs with a length cap and passage mask
| Long documents lose entities at the edges | Truncation at the model's maximum length | Slide overlapping windows and merge predictions

## Going deeper

**Why Viterbi is exact.** The score is a sum of terms that each involve at most two neighbouring tags. So the best sequence ending in tag $k$ at position $t$ must extend the best sequence ending in some tag $j$ at $t-1$: any worse prefix could be swapped for the better one without changing the remaining terms. This "optimal substructure" is exactly what dynamic programming needs. Memory is $O(TK)$ for the back-pointers.

**The forward algorithm and CRF training.** Replacing $\max$ by $\log\sum\exp$ in the Viterbi update gives $\alpha_t(k) = z_{t,k} + \log\sum_j \exp(\alpha_{t-1}(j) + A_{jk})$, and $\log Z(x) = \log \sum_k \exp \alpha_T(k)$. Autograd differentiates through it, so a CRF is about 30 lines of PyTorch. The recursion is sequential over $t$, so a CRF head is slower than a plain softmax head.

**Why a softmax head is not enough in principle, and often enough in practice.** A per-token softmax treats tags as independent given the encoder states. With weak encoders that was badly wrong and CRFs gave large gains; a deep bidirectional transformer already sees the whole sentence in every $h_t$, so the CRF gain is usually small, while constrained decoding stays a free fix.

**Global normalisation.** Older maximum-entropy Markov models normalised each step separately, which lets a state with few successors ignore the evidence (the *label bias* problem). CRFs normalise once over whole sequences, which avoids it.

**Beyond flat BIO.** Nested entities, discontinuous entities ("left and right ventricle") and overlapping types break the one-label-per-token assumption. Span-based models instead enumerate all spans up to a length cap and classify each one, $O(T \cdot L_{\max})$ per sentence. Extractive QA is the simplest span model: one span, scored as start plus end.

**The HMM connection.** A hidden Markov model is the generative cousin of a linear-chain CRF: its Viterbi decode is the same algorithm with $A_{jk} = \log p(k \mid j)$ and $s_t(k) = \log p(x_t \mid k)$.

## Key terms

| Term | Meaning
| --- | --- |
| Sequence labelling | Predicting one label for every token of the input
| BIO / BIOES | Tag schemes marking the beginning, inside (and end, single) of entities, plus O for outside
| Named-entity recognition | Finding spans that name people, places, organisations and other types
| Greedy decoding | Taking the highest-scoring tag at each position independently
| Viterbi algorithm | Dynamic programming that finds the best tag sequence in $O(TK^2)$
| Transition score | Score $A_{jk}$ for tag $k$ following tag $j$; $-\infty$ forbids the pair
| CRF layer | Learns transition scores and trains on whole-sequence likelihood
| Ignore label (-100) | Label that excludes a position, such as a later subword, from the loss
| Entity-level F1 | Harmonic mean of precision and recall over exactly matching spans
| Span extraction | Choosing a start and end token, as in extractive question answering

<div class="take">
<div class="lab">Key takeaway</div>

sequence labelling is per-token classification plus structure. Encode the tokens, decode with constrained Viterbi or a CRF so the tags form valid spans, align subword tokens to words with the -100 ignore label, and judge the model by entity-level F1, never by token accuracy.

</div>

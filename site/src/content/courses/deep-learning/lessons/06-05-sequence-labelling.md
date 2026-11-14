---
id: "06.05"
course: "deep-learning"
module: "06"
slug: "sequence-labelling"
title: "Sequence labelling"
keyIdeas: ["Sequence labelling assigns one label per token: part-of-speech tags, or named-entity tags in the BIO scheme (B- begins an entity, I- continues it, O is outside).", "Per-token argmax can output invalid sequences (I-PER right after O or after B-LOC). Constrained or CRF decoding with the Viterbi algorithm finds the best valid sequence in $O(TK^2)$.", "With subword tokenisers, label only the first subword of each word and mask the rest out of the loss; map predictions back to words via offsets.", "Evaluate NER with entity-level F1 (exact span and type match), not token accuracy, which is inflated by the many O tokens.", "Extractive question answering is span labelling: predict start and end logits per token, then choose the best valid pair with start ≤ end and a length cap."]
question: "A NER model's per-token argmax output for five words is: O, B-LOC, I-PER, I-PER, O. How would a lenient BIO decoder, a strict decoder, and Viterbi decoding with transition constraints each handle this, and how does each outcome affect entity-level F1 if the gold answer is a single PER entity over words 2-4?"
answer: "A lenient decoder closes LOC at word 2 and starts a new PER entity at word 3 from the stray I-PER, giving LOC(2) and PER(3-4). A strict decoder keeps LOC(2) and discards the I-PER tokens, since they lack a B-PER. Viterbi with constraints forbids B-LOC followed by I-PER and finds the best valid sequence; if the PER logits on words 3-4 are strong, it is likely to choose B-PER I-PER I-PER for words 2-4, or B-LOC O O if the LOC evidence on word 2 dominates. Against the gold PER(2-4): lenient produces two false positives and one miss (P = 0, R = 0); strict produces one false positive and one miss; only the constrained decode can produce an exact match, giving P = R = F1 = 1 for this sentence. Structured decoding converts per-token evidence into consistent spans."
sources: [{"title": "Neural Architectures for Named Entity Recognition (Lample et al., 2016)", "url": "https://arxiv.org/abs/1603.01360", "note": "BiLSTM-CRF for NER and the IOBES scheme; section 2 explains the CRF layer and Viterbi decoding."}, {"title": "Bidirectional LSTM-CRF Models for Sequence Tagging (Huang et al., 2015)", "url": "https://arxiv.org/abs/1508.01991", "note": "Compares LSTM, BiLSTM and BiLSTM-CRF taggers on POS, chunking and NER."}, {"title": "Introduction to the CoNLL-2003 Shared Task", "url": "https://aclanthology.org/W03-0419/", "note": "Defines the standard NER dataset and the exact-match entity F1 evaluation."}, {"title": "SQuAD: 100,000+ Questions for Machine Comprehension", "url": "https://arxiv.org/abs/1606.05250", "note": "The extractive QA task framing that start/end span prediction solves."}]
practice: ["named-entity-recognition", "greedy-sequence-tagging", "token-classification", "span-extraction", "pos-tagging"]
tags: ["NER", "BIO tagging", "Viterbi"]
tools: ["Python", "NumPy"]
---

## One label per token

Many NLP tasks label every token rather than the whole text:

- **Part-of-speech tagging**: noun, verb, adjective and so on, with 17 tags in Universal Dependencies or 45 in the Penn Treebank.
- **Named-entity recognition (NER)**: find spans that name people, organisations, locations, dates, drugs.
- **Span extraction**: find the answer to a question as a contiguous span of a passage.

All share one model shape: an encoder produces a vector per token (bidirectional LSTM from 05.02, or a pretrained transformer from the next lesson), a linear layer maps each to $K$ label logits, and a decoding step turns logits into labels or spans.

~~~dg-flow The token-classification pipeline. Decoding and evaluation happen at the word or span level, not the subword level.
Words -(tokeniser, keep offsets)-> Subwords -> Encoder -> *Per-token logits -(argmax or Viterbi)-> BIO tags -(group)-> Entity spans
~~~

## Tagging schemes

Entities span several tokens, so labels encode position within the entity. In BIO, B-X starts an entity of type X, I-X continues it, O is outside:

| Token | Barack | Obama | visited | New | York | yesterday |
| --- | --- | --- | --- | --- | --- | --- |
| Tag | B-PER | I-PER | O | B-LOC | I-LOC | O |

The B tag is what separates two adjacent entities of the same type. BIOES adds E (end) and S (single-token) tags, which gives the model more explicit boundary signals and sometimes helps a little. Plain BIO cannot represent nested entities ("Bank of [England]" inside an organisation); those need span-based models that score every candidate span.

```python
def bio_to_spans(tags):
    spans, start, typ = [], None, None
    for i, t in enumerate(tags + ["O"]):              # sentinel closes a trailing span
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None: spans.append((start, i, typ))
            start, typ = (i, t[2:]) if t != "O" else (None, None)   # stray I- starts a span
    return spans                                      # [(start, end_exclusive, type)]
```

How to treat an I- tag that follows O or a different type is a policy choice; the lenient policy above starts a new entity, a strict one discards it. Use the same policy as your evaluation script.

## Baselines and decoding

For POS tagging, simply assigning each word its most frequent training tag (and the most common open-class tag to unknown words) already gets about 92% accuracy on English newswire; good models reach about 97%. Most of the remaining errors are genuinely ambiguous words such as "that" or "back".

Greedy decoding takes the argmax at each token independently. It is fast but can emit invalid sequences like `O I-PER` or `B-LOC I-PER`, because nothing ties neighbouring decisions together. Two fixes:

- **Constrained decoding**: forbid invalid transitions, then find the highest-scoring valid sequence.
- **A CRF layer** on top of the encoder (BiLSTM-CRF was the standard NER architecture before transformers): learn a $K \times K$ transition score matrix and train on whole-sequence likelihood.

Both use the Viterbi algorithm: dynamic programming that keeps, for each position and each tag, the best score of any path ending there, plus a back-pointer. Cost is $O(T K^2)$, trivial for a dozen tags. The same algorithm decodes hidden Markov models, the classical tagger.

## Subword alignment

A transformer tokeniser splits "Kaczynski" into several pieces, but the labels belong to words. The standard recipe: give the word's label to its first subword, and give the remaining subwords the ignore label (-100 in PyTorch's cross-entropy) so they are excluded from the loss. At prediction time read the label from each word's first subword. Character offsets from the tokeniser (lesson 06.01) map spans back to the original string. Getting this alignment wrong shifts every label by a token, and accuracy collapses without any error being raised.

## Evaluating NER

Token accuracy is misleading: most tokens are O, so a model that predicts O everywhere can score above 80%. The CoNLL standard is entity-level precision, recall and F1, where a predicted entity counts only if both its boundaries and its type match exactly.

<div class="ex">
<div class="lab">Worked example</div>

**token accuracy versus entity F1**

A test set has 10 gold entities. The model predicts 8, of which 6 match exactly; the other 2 have one boundary token wrong.

Precision $= 6/8 = 0.75$, recall $= 6/10 = 0.60$, F1 $= 2 \times 0.75 \times 0.6 / 1.35 = 0.667$.

Token accuracy could still be 97%, because the two near-misses are each a single wrong token among hundreds of correct O tags.

</div>

## Span extraction

For extractive question answering (SQuAD-style), the input is `[CLS] question [SEP] passage [SEP]`, and the model outputs two logits per token: start score $s_i$ and end score $e_j$. The answer is the pair maximising $s_i + e_j$ subject to $i \le j$, $j - i < L_{\max}$ (say 30 tokens), and both positions inside the passage rather than the question. Taking the two argmaxes independently can produce an end before the start; search over valid pairs, which is $O(T \cdot L_{\max})$. If the dataset allows unanswerable questions, compare the best span's score against the score of pointing both at `[CLS]`.

<div class="take">
<div class="lab">Key takeaway</div>

sequence labelling is per-token classification plus structure. Encode tokens, constrain or Viterbi-decode so labels form valid spans, align subwords to words carefully, and measure with entity-level F1.

</div>

---
id: "05.03"
course: "deep-learning"
module: "05"
slug: "sequence-to-sequence-and-attention"
title: "Sequence-to-sequence and attention"
keyIdeas: ["An encoder-decoder maps an input sequence to an output sequence of a different length: the encoder reads the source, the decoder generates the target one token at a time conditioned on it.", "Without attention, everything about the source must pass through one fixed-size vector, and quality drops sharply on long inputs.", "Attention lets each decoder step compute a weighted average of all encoder states, with weights from a learned score between the decoder state and each encoder state.", "Additive (Bahdanau) attention scores with a small MLP, $v^\\top \\tanh(W_s s + W_h h_i)$; multiplicative (Luong) attention uses $s^\\top W h_i$ or a plain dot product, which is cheaper and is what transformers scale up.", "Training uses teacher forcing (feed the true previous token); inference feeds the model's own predictions. Mask padded source positions before the softmax."]
question: "A seq2seq LSTM translator without attention handles 10-word sentences well but produces poor translations for 50-word sentences. Explain the failure mechanistically, and explain what attention changes in both the forward pass and the backward pass."
answer: "Forward: the decoder sees the source only through the encoder's final state, a fixed-size vector. A 10-word sentence fits; a 50-word sentence must be compressed into the same capacity, so details, especially from early words that have been overwritten by many updates, are lost. Backward: the loss at an early target word must reach the corresponding early source word through roughly 50 encoder steps plus the decoder steps, and the gradient weakens over that path, so the alignment is hard to learn. With attention, each decoder step reads a weighted average of all encoder states, so capacity grows with source length and the decoder can attend to whichever word it needs. The gradient reaches each encoder state directly through its attention weight, so the path length no longer depends on the distance between source and target positions."
sources: [{"title": "Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)", "url": "https://arxiv.org/abs/1409.3215", "note": "The encoder-decoder LSTM and the source-reversal trick."}, {"title": "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)", "url": "https://arxiv.org/abs/1409.0473", "note": "Additive attention; figure 2 shows the length-robustness result."}, {"title": "Effective Approaches to Attention-based NMT (Luong et al., 2015)", "url": "https://arxiv.org/abs/1508.04025", "note": "Dot, general and concat scores, and global versus local attention."}, {"title": "Dive into Deep Learning: Bahdanau Attention", "url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html", "note": "Runnable encoder-decoder with additive attention."}]
practice: ["sequence-to-sequence-forward-pass", "attention-mechanism", "scaled-dot-product-attention"]
tags: ["seq2seq", "attention"]
tools: ["NumPy", "PyTorch"]
---

## Mapping one sequence to another

Translation, summarisation, speech recognition and code generation all map an input sequence to an output sequence whose length differs and is not known in advance. The sequence-to-sequence architecture (Sutskever et al., 2014; Cho et al., 2014) handles this with two recurrent networks:

- The **encoder** reads the source $x_1, \dots, x_S$ and produces hidden states $h_1, \dots, h_S$.
- The **decoder** is a language model over the target that is conditioned on the source. It starts from a summary of the encoder and emits $y_1, y_2, \dots$ until it produces an end token.

In the original form the conditioning is just the encoder's final state, used to initialise the decoder. Sutskever et al. reached strong translation results with 4-layer LSTMs and one surprising trick: reversing the source sentence, which puts the first source words close to the first target words and shortens the gradient path between them.

## Teacher forcing and exposure bias

During training the decoder at step $t$ receives the true previous token $y_{t-1}$, not its own prediction. This is teacher forcing: every step's loss is computed in parallel from ground truth inputs, and training is stable. At inference there is no ground truth, so the decoder consumes its own outputs, and an early mistake puts it in states it never saw during training. The mismatch is called exposure bias. Mitigations include scheduled sampling (sometimes feeding model predictions during training) and better search at decoding time (lesson 06.07).

## The bottleneck

Compressing a 50-word sentence into one 1,000-dimensional vector loses information, and the encoder's early states have the longest path to the decoder's outputs. Bahdanau et al. (2014) showed that BLEU for a basic encoder-decoder falls steeply as sentence length grows past about 30 words, while their attention model stays roughly flat.

## Attention: a soft lookup into the source

Instead of one summary, keep all encoder states and let each decoder step choose what to read. At step $t$, with decoder state $s_{t-1}$:

$$e_{t,i} = \text{score}(s_{t-1}, h_i), \qquad \alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}, \qquad c_t = \sum_i \alpha_{t,i}\, h_i$$

The context vector $c_t$ is fed into the decoder along with the previous token, and the decoder produces $s_t$ and the next-token distribution.

~~~dg-seq One decoder step with attention over the encoder states.
Decoder > Attention: query s(t-1)
Attention > Encoder states: score every h(i)
Encoder states -> Attention: softmax weights α(t,i)
Attention -> Decoder: context c(t) = Σ α h
Decoder > Decoder: update state, emit y(t)
~~~

Score functions:

- **Additive** (Bahdanau): $\text{score}(s, h) = v^\top \tanh(W_s s + W_h h)$, a one-hidden-layer MLP. Works well when $s$ and $h$ have different sizes.
- **General / bilinear** (Luong): $s^\top W h$.
- **Dot** (Luong): $s^\top h$. No parameters, and it turns a whole step into one matrix multiply. Scaled by $1/\sqrt{d}$, this is the scaled dot-product attention of transformers.

Bahdanau used a bidirectional encoder so each $h_i$ summarises context on both sides of source word $i$, and scored with the previous decoder state $s_{t-1}$; Luong scored with the current state $s_t$. The differences are small in practice.

```python
import numpy as np
def additive_attention(s, H, Ws, Wh, v, mask):   # s: (D,), H: (S, E), mask: (S,) bool
    e = np.tanh(H @ Wh.T + Ws @ s) @ v            # (S,) one score per source position
    e = np.where(mask, e, -np.inf)                # padded positions get zero weight
    a = np.exp(e - e.max()); a /= a.sum()
    return a @ H, a                               # context (E,), weights (S,)
```

<div class="ex">
<div class="lab">Worked example</div>

**three attention weights by hand**

Scores for three source words: $e = (2.0, 0.5, -1.0)$.

Exponentials: $7.389, 1.649, 0.368$, total $9.406$. Weights: $\alpha = (0.786, 0.175, 0.039)$.

The context vector is 79% the first encoder state. If a fourth position were padding and not masked, it would steal weight from real words.

</div>

## What attention changes

- **Gradient path**: the decoder's loss at step $t$ reaches encoder state $h_i$ directly through $\alpha_{t,i}$, in one hop, instead of through every intervening recurrent step.
- **Interpretability**: the matrix of $\alpha_{t,i}$ is a soft alignment; in translation it is roughly diagonal, with visible reorderings such as adjective-noun swaps between English and French.
- **Cost**: computing all weights is $O(S \cdot T)$ for source length $S$ and target length $T$, negligible next to the recurrent work at these lengths.

Once every decoder step can look anywhere in the source, the next question is why the encoder and decoder need recurrence at all. If each position can attend to every other position, including within the same sequence, the sequential loop can go. That is the transformer, next lesson.

<div class="take">
<div class="lab">Key takeaway</div>

seq2seq models separate reading from writing. Attention replaces a single fixed-size summary with a learned, per-step weighted average of all encoder states, removing the length bottleneck and giving gradients a one-hop path to every source position.

</div>

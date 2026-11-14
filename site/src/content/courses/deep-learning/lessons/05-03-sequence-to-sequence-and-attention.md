---
id: "05.03"
course: "deep-learning"
module: "05"
slug: "sequence-to-sequence-and-attention"
title: "Sequence-to-sequence and attention"
keyIdeas: ["An encoder-decoder maps an input sequence to an output sequence of a different length: the encoder reads the source into hidden states, and the decoder, a language model conditioned on the source, writes the target one token at a time until it emits an end token.", "Without attention, everything about the source must pass through one fixed-size vector. Quality drops sharply on long inputs: in our reversal experiment, whole-sequence accuracy fell from 96% at 4 digits to under 2% at 12.", "Attention scores each encoder state against the decoder state, turns the scores into weights with a softmax, and feeds the weighted average (the context vector) to the decoder. The weights are recomputed at every output step.", "Additive (Bahdanau) attention scores with a small MLP, $v^\\top \\tanh(W_s s + W_h h_i)$; multiplicative (Luong) attention uses $s^\\top W h_i$ or a plain dot product, which is cheaper and, scaled by $1/\\sqrt{d}$, is what transformers use.", "Training uses teacher forcing (feed the true previous token); inference feeds the model's own predictions, so early mistakes compound (exposure bias). Always evaluate free-running, and mask padded source positions to $-\\infty$ before the softmax."]
question: "A seq2seq LSTM translator without attention handles 10-word sentences well but produces poor translations for 50-word sentences, even though its teacher-forced validation loss looks fine. Explain the failure mechanistically, explain what attention changes in both the forward pass and the backward pass, and say how you would measure the improvement honestly."
answer: "Forward: the decoder sees the source only through the encoder's final state, a fixed-size vector. A 10-word sentence fits; a 50-word sentence must be compressed into the same capacity, so details, especially from early words that have been overwritten by many updates, are lost. This is the bottleneck, and its signature is quality that falls steadily with length. Backward: the loss at an early target word must reach the corresponding early source word through roughly 50 encoder steps plus the decoder steps, and the gradient weakens over that path, so the alignment is hard to learn. The teacher-forced validation loss hides part of the problem: during evaluation the decoder is still fed the true previous words, while in deployment it consumes its own outputs, so one early error pushes it into unfamiliar states and errors compound (exposure bias); the bottleneck makes those early errors more likely. With attention, each decoder step scores every encoder state against its current state, normalises the scores with a softmax and reads the weighted average of all encoder states, so the information available grows with source length and the decoder can look up whichever word it needs at that step. The gradient reaches each encoder state directly through its attention weight and the context vector, so the path length no longer depends on the distance between source and target positions. To measure the gain honestly, decode free-running (greedy or beam search, not teacher-forced), report BLEU or accuracy per sentence-length bucket rather than one average, make sure padded positions are masked so results do not depend on batch composition, and inspect a few attention maps to confirm a sensible, roughly diagonal alignment."
sources: [{"title": "Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)", "url": "https://arxiv.org/abs/1409.3215", "note": "The encoder-decoder LSTM and the source-reversal trick."}, {"title": "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)", "url": "https://arxiv.org/abs/1409.0473", "note": "Additive attention; figure 2 shows the length-robustness result."}, {"title": "Effective Approaches to Attention-based NMT (Luong et al., 2015)", "url": "https://arxiv.org/abs/1508.04025", "note": "Dot, general and concat scores, and global versus local attention."}, {"title": "Dive into Deep Learning: Bahdanau Attention", "url": "https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html", "note": "Runnable encoder-decoder with additive attention."}]
practice: ["sequence-to-sequence-forward-pass", "attention-mechanism", "scaled-dot-product-attention"]
tags: ["seq2seq", "attention"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- How an **encoder-decoder** (seq2seq) model turns one sequence into another sequence of a different length, and why the decoder is really a language model with extra input.
- Why squeezing the whole input into one vector breaks down on long inputs, shown on a small model you could train on a laptop.
- How attention works, step by step: scores, softmax weights, and the context vector, computed by hand on three words.
- The common score functions (additive, general, dot, scaled dot) and when each is used.
- How teacher forcing trains the decoder, what exposure bias is, and why padded positions must be masked.

*Before you start:* you need recurrent networks and backpropagation through time (05.01), LSTM/GRU cells and bidirectional encoders (05.02), and the softmax and cross-entropy loss from earlier modules. Lesson 05.04 (transformers) builds directly on the attention step introduced here.

Lessons 05.01 and 05.02 built recurrent networks that read a sequence and keep a memory over dozens of steps, producing one output per step or one label per sequence. Many tasks instead need to read a whole sequence, then *write* a new one whose length is different and not known in advance. Translation, summarisation and speech-to-text all have this shape. This lesson builds the standard recurrent answer and then fixes its biggest weakness with attention, the idea that later grew into the transformer.

## The idea in one picture

Take a toy task that is easy to state but needs the whole input: **reverse a string of digits**. The input `4 7 0 2` must become `2 0 7 4`. The first output digit depends on the *last* input digit, and the last output depends on the first, so the model cannot write anything useful until it has read everything.

| Input (source) | 3 1 4 1 5 | 0 4 7 0 4 4 5 4 7 6 2 5
| --- | --- | --- |
| Output (target) | 5 1 4 1 3 | 5 2 6 7 4 5 4 4 0 7 4 0
| Length | 5 | 12

We trained two small models on this task with lengths 3 to 20 (same encoder, same decoder size, same 2,000 training steps; the setup is given in the code section). The only difference: one model passes the input to the decoder as a single summary vector, the other uses attention.

~~~dg-fig 05.03-length.svg
Same task, same budget. Without attention, accuracy falls from length 6 on, and whole-sequence accuracy is near zero by 12 digits. With attention the model is essentially perfect across the shaded training range and only fails on lengths it never saw.
~~~

On 4-digit inputs both models are fine (98.8% and 100% of digits correct). By 12 digits the model without attention gets only 54% of digits right and almost never (1.5%) a whole sequence; the attention model still gets at least 99.5% of whole sequences right up to length 20. Both curves eventually drop past the training range, which is a separate problem (length generalisation) discussed under Going deeper.

Terms: the **source** is the input $x_1, \dots, x_S$ and the **target** the output $y_1, \dots, y_T$ ($S \ne T$ in general). The **encoder** reads the source; the **decoder** writes the target one token at a time; **attention** lets the decoder look back at any part of the source at every step.

## Step 1: the encoder reads the source

The encoder is an ordinary recurrent network (an LSTM or GRU from 05.02). It reads the source one token at a time and produces a hidden state at every position:

$$h_i = \text{RNN}_{\text{enc}}(h_{i-1}, \text{emb}(x_i)), \qquad i = 1, \dots, S.$$

Here $\text{emb}(x_i)$ is the learned embedding of token $x_i$, and $h_i$ (say 64 or 1,000 numbers) summarises the source up to position $i$. Encoders are usually **bidirectional**: a second RNN reads right to left and the states are concatenated, so each $h_i$ describes word $i$ with context on both sides.

## Step 2: the decoder writes, one token at a time

The decoder is a **conditional language model**: it predicts the next target token from the target tokens so far *and* the source. It factorises the probability of the whole output as a chain of next-token predictions:

$$p(y_1, \dots, y_T \mid x) = \prod_{t=1}^{T} p(y_t \mid y_1, \dots, y_{t-1}, x).$$

Each factor is one softmax over the vocabulary. The decoder keeps its own state $s_t$; at each step it takes the previous token, updates $s_t$, and outputs a distribution over the next token. A start token `<bos>` is fed at step 1, and generation stops when the decoder emits `<eos>`, so the model itself decides the output length.

In the original seq2seq models (Sutskever et al., 2014; Cho et al., 2014) the only link between the two halves is the encoder's final state: it initialises the decoder, $s_0 = h_S$, and the decoder never sees the source again. Sutskever et al. used 4-layer LSTMs and one surprising trick: feeding the source *reversed*, which puts the first source words next to the first target words in time and shortens the gradient path between them.

## Step 3: the bottleneck

Look at what Step 2 asks of $h_S$. One vector of fixed size has to carry every detail of the source: every digit, every word, its order. A 5-digit input fits easily. A 20-digit input must be squeezed into the same 64 numbers, and the early digits have been overwritten by many later updates. This is the **bottleneck**.

The no-attention curve in the first figure is the bottleneck made visible: the decoder simply does not get the information. Bahdanau et al. (2014) showed the same pattern on real translation: the BLEU score of a basic encoder-decoder falls steeply once sentences pass roughly 20-30 words, while their attention model stays roughly flat.

There is also a training-side problem: the loss on the last output digit must reach the *first* source digit through all $S$ encoder steps and $T$ decoder steps, and (as 05.01 showed) long gradient paths weaken the signal.

## Step 4: training with teacher forcing

To train, we minimise the cross-entropy of the true target, summed over steps:

$$\mathcal{L} = -\sum_{t=1}^{T} \log p(y_t \mid y_1, \dots, y_{t-1}, x).$$

The condition uses the *true* previous tokens: during training the decoder is fed the correct previous token at every step, not its own guess. This is **teacher forcing**, and it keeps training stable because one early mistake cannot derail the rest of the sequence.

At inference there is no correct answer, so the decoder consumes its own predictions. After one mistake it is in a state it never saw in training, and mistakes compound. This mismatch is called **exposure bias**.

~~~dg-fig 05.03-exposure.svg
The no-attention model on 20-digit inputs. Dashed: fed the true previous digit (teacher forcing). Solid: fed its own guesses, as at inference. The shaded gap is accuracy lost to errors compounding. The attention model (flat line at the top) makes no first mistake, so it never pays this cost.
~~~

Both no-attention curves start at 89% for the first digit (the inputs are identical there). By position 5 the teacher-forced model is right 70% of the time and the free-running one only 48%; by position 10, 54% versus 24%. Mitigations include **scheduled sampling** (Bengio et al., 2015), which sometimes feeds the model's own predictions during training, and better search at decoding time (beam search, lesson 06.07).

## Step 5: attention, a soft lookup into the source

The fix for the bottleneck is simple to say: **keep all the encoder states** $h_1, \dots, h_S$, and let each decoder step choose which ones to read. The choice is made in three small moves.

**Move 1: score.** Compare the decoder's current state (the "question" it is asking) with each encoder state, giving one number per source position:

$$e_{t,i} = \text{score}(s_{t-1}, h_i).$$

Here $s_{t-1}$ is the decoder state before producing token $t$, and $e_{t,i}$ is a raw score: high means "source position $i$ looks relevant right now". Scores can be any real number.

**Move 2: normalise.** Turn the scores into positive weights that sum to 1 with a softmax over source positions:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{S} \exp(e_{t,j})}.$$

$\alpha_{t,i}$ is the **attention weight**: the fraction of step $t$'s attention spent on source position $i$.

**Move 3: average.** The **context vector** is the weighted average of the encoder states:

$$c_t = \sum_{i=1}^{S} \alpha_{t,i}\, h_i.$$

It has the same size as one $h_i$: if one weight is near 1, $c_t$ is essentially that state. The decoder uses $c_t$ with the previous token to update its state and predict $y_t$. The weights are recomputed at every step, so step 1 can read the end of the source and step 12 the beginning.

~~~dg-fig 05.03-softmax.svg
The softmax step for the worked example below. Scores (left) become weights that sum to 1. If a padding position with score 0 is not masked (second bar in each pair), it takes 9.6% of the weight away from the real words.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one attention step by hand**

Three source words with 2-D encoder states $h_1 = (1, 1)$, $h_2 = (0, 1)$, $h_3 = (-1, 1)$. The decoder state is $s = (1.5, 0.5)$. Use the plain dot product as the score.

Scores: $e_1 = 1.5 \cdot 1 + 0.5 \cdot 1 = 2.0$; $e_2 = 1.5 \cdot 0 + 0.5 \cdot 1 = 0.5$; $e_3 = 1.5 \cdot (-1) + 0.5 \cdot 1 = -1.0$.

Exponentials: $e^{2.0} = 7.389$, $e^{0.5} = 1.649$, $e^{-1.0} = 0.368$. Their total is $9.406$.

Weights: $\alpha = (7.389, 1.649, 0.368) / 9.406 = (0.786, 0.175, 0.039)$. They sum to 1.

Context: $c = 0.786 \cdot (1, 1) + 0.175 \cdot (0, 1) + 0.039 \cdot (-1, 1) = (0.786 - 0.039,\ 0.786 + 0.175 + 0.039) = (0.747, 1.000)$. The second coordinate is exactly 1 because every $h_i$ has a 1 there, and a weighted average of equal numbers is that number: a quick sanity check.

So the context is 79% word 1. If a fourth position were padding with score 0 and *not* masked, the total would grow to $10.406$ and the real weights would shrink to $(0.710, 0.158, 0.035)$, with $0.096$ wasted on padding.

</div>

## Step 6: choosing the score function

Move 1 needs a function that compares a decoder state $s$ with an encoder state $h$. Three classic choices:

| Name | Formula | Learned parameters | Notes
| --- | --- | --- | --- |
| Additive (Bahdanau) | $v^\top \tanh(W_s s + W_h h)$ | $W_s, W_h, v$ | A one-hidden-layer MLP; works when $s$ and $h$ have different sizes
| General / bilinear (Luong) | $s^\top W h$ | $W$ | One matrix maps $h$ into the space of $s$
| Dot (Luong) | $s^\top h$ | none | Needs equal sizes; a whole step is one matrix multiply
| Scaled dot | $s^\top h / \sqrt{d}$ | none | Divides by the square root of the vector size $d$; the transformer's choice

Bahdanau et al. scored with the previous decoder state $s_{t-1}$, as in Step 5; Luong et al. (2015) scored with the current state $s_t$. The difference matters little in practice. The dot-product family won because attention over all positions becomes one matrix multiplication, which is what the transformer (05.04) scales up.

~~~dg-seq One decoder step with attention over the encoder states.
Decoder > Attention: query s(t-1)
Attention > Encoder states: score every h(i)
Encoder states -> Attention: softmax weights α(t,i)
Attention -> Decoder: context c(t) = Σ α h
Decoder > Decoder: update state, emit y(t)
~~~

## Step 7: masking padding

Sentences in a batch have different lengths, so shorter ones are padded with a dummy token. Padded positions have encoder states too and would get a score; as the worked example showed, weight then leaks onto them, by an amount that depends on how much padding a sentence happened to get.

The fix is a **mask**: before the softmax, set the score of every padded position to $-\infty$ (in practice a very large negative number). Since $e^{-\infty} = 0$, those positions get exactly zero weight, and the real positions share the full 1.

## Reading the attention weights

The matrix of $\alpha_{t,i}$ can be drawn as a heatmap: rows are output steps, columns are source positions. In translation this is a soft **alignment**, usually close to diagonal with visible swaps where languages order words differently (for example adjective-noun order between English and French).

~~~dg-fig 05.03-heatmap.svg
Attention weights from our trained model reversing 12 digits. Each row sums to 1. The bright band runs along the anti-diagonal: the first output reads the last input, the last output reads the first.
~~~

The model discovered the reversal on its own. Notice that the weight is often split between two neighbouring positions (for example 0.39 and 0.52 in the second row). With a bidirectional encoder each $h_i$ already carries information about its neighbours, so reading either one works. This is a useful warning: attention weights show where the model *reads*, which is not always where the information *originated*.

## Attention in code

A single additive-attention step in NumPy, with a mask. Three real words and one padding row:

```python
import numpy as np

def additive_attention(s, H, Ws, Wh, v, mask):   # s: (D,), H: (S, E), mask: (S,) bool
    e = np.tanh(H @ Wh.T + Ws @ s) @ v            # (S,) one score per source position
    e = np.where(mask, e, -np.inf)                # padded positions get zero weight
    a = np.exp(e - e.max()); a /= a.sum()         # stable softmax
    return a @ H, a                               # context (E,), weights (S,)

rng = np.random.default_rng(0)
H = np.array([[1., 1], [0, 1], [-1, 1], [0, 0]])  # 3 words + 1 padding row
s = np.array([1.5, 0.5])
Ws, Wh, v = rng.normal(size=(3, 2)), rng.normal(size=(3, 2)), rng.normal(size=3)
c, a = additive_attention(s, H, Ws, Wh, v, np.array([True, True, True, False]))
print(a.round(3))    # [0.158 0.151 0.69  0.   ]   (random W, so these are not the dot-product weights)
print(c.round(3))    # [-0.532  1.   ]
```

The same step for a whole batch in PyTorch, with scaled dot-product scores and a length mask:

```python
import torch
torch.manual_seed(0)
B, S, D = 2, 4, 8
H = torch.randn(B, S, D)                          # encoder states
s = torch.randn(B, D)                             # decoder state (the query)
lengths = torch.tensor([4, 2])                    # 2nd sentence has 2 padding positions
mask = torch.arange(S)[None, :] < lengths[:, None]           # (B, S), True = real token
e = torch.einsum('bd,bsd->bs', s, H) / D**0.5                # (B, S) scores
e = e.masked_fill(~mask, float('-inf'))
alpha = e.softmax(dim=-1)                                    # rows sum to 1
c = torch.einsum('bs,bsd->bd', alpha, H)                     # (B, D) context vectors
print(alpha.round(decimals=3))
# tensor([[0.1580, 0.4770, 0.0800, 0.2850],
#         [0.5210, 0.4790, 0.0000, 0.0000]])
```

Teacher forcing lives in the decoder loop: at step $t$ the next input is `tgt[:, t+1]` (the true token) during training and `logits.argmax(-1)` (the model's own guess) at inference. The full experiment (`exp_0503.py`): a bidirectional GRU encoder with 64 units per direction, a 64-unit GRU decoder, Adam with learning rate 0.003, batch 64, gradient clipping at 1.0, 2,000 steps per model on random digit strings of length 3 to 20. Each model trains in one to two CPU minutes.

## Checking that it worked

- **Evaluate by length bucket.** An overall average hides the bottleneck; plot accuracy (or BLEU) per length band, as in the first figure.
- **Evaluate free-running, not teacher-forced.** Teacher-forced accuracy flatters the model: at 20 digits our no-attention model scored 56% of digits correct when fed true tokens but only 31% when generating on its own. Report the deployment number.
- **Look at a few attention maps.** Expect a sensible alignment (diagonal for translation). Uniform rows (every weight about $1/S$) mean attention is unused: check that $c_t$ really reaches the decoder. Rows should sum to 1 and padded columns be exactly 0.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Quality falls steadily with input length | Fixed-size bottleneck (no attention) | Add attention; use a bidirectional encoder
| Results change with batch composition or padding amount | Padding not masked before the softmax | Set padded scores to $-\infty$ before the softmax
| Great validation loss, poor generated outputs | Exposure bias: loss measured with teacher forcing | Evaluate free-running; scheduled sampling; beam search
| Output repeats a phrase or skips part of the source | Attention returns to the same positions or skips some | Coverage penalty or coverage features; beam search with length normalisation
| NaN in the weights | A row that is entirely masked: softmax of all $-\infty$ | Make sure every row has at least one real position

## Going deeper

**The gradient path.** With attention, the loss at output step $t$ reaches encoder state $h_i$ directly through $\alpha_{t,i}$ and $c_t$, in one hop, instead of through every recurrent step between them. That is why the attention model in our figures learned the reversal in a few hundred steps (its batch loss was 0.08 after 500 steps, while the no-attention model's was still about 1.5).

**Capacity and cost.** With attention the decoder can read $S$ vectors instead of one, so usable memory grows with the input. The price is $S$ scores at each of $T$ output steps, $O(S \cdot T)$, small next to the recurrent work for sentence-length inputs.

**Why scale the dot product.** If the entries of $s$ and $h$ are roughly independent with variance 1, their dot product has variance $d$. For $d = 512$ the scores spread over tens of units, and the softmax saturates into a near one-hot vector with tiny gradients. Dividing by $\sqrt{d}$ restores unit variance. You can see the effect as a temperature: dividing the example scores $(2.0, 0.5, -1.0)$ by $\sqrt 2$ softens the weights from $(0.786, 0.175, 0.039)$ to $(0.682, 0.236, 0.082)$.

**Attention as a lookup.** The decoder state is a **query**; each encoder state is both a **key** (compared with the query) and a **value** (averaged). Give keys and values separate projections and let a sequence attend to *itself*, and you have self-attention. Once every position sees every other directly, the recurrent loop is unnecessary and can be dropped so all positions run in parallel: the transformer, next lesson.

**Length generalisation.** Our attention model was near perfect up to length 22 (98.5% exact matches) and fell to 54% exact matches at 26 and near zero by 30. A likely reason is that the decoder must track how far it has got, and it never learned to count past about 20. Long documents pose the same problem for real systems (see 05.04 on position schemes).

## Key terms

| Term | Meaning
| --- | --- |
| Encoder-decoder (seq2seq) | Two networks: one reads the source into states, the other writes the target token by token
| Conditional language model | A decoder that predicts the next token from the previous tokens and the source
| Bottleneck | Forcing the whole source through one fixed-size vector
| Teacher forcing | Training the decoder on the true previous token instead of its own prediction
| Exposure bias | The mismatch when, at inference, the decoder must consume its own (possibly wrong) outputs
| Attention weight $\alpha_{t,i}$ | Softmax-normalised share of step $t$'s attention placed on source position $i$
| Context vector $c_t$ | Weighted average of encoder states used by decoder step $t$
| Additive / dot-product score | MLP-based versus inner-product way to compare decoder and encoder states
| Mask | Setting scores of padding (or forbidden) positions to $-\infty$ so their weight is exactly zero

<div class="take">
<div class="lab">Key takeaway</div>

seq2seq models separate reading from writing. Attention replaces a single fixed-size summary with a learned, per-step weighted average of all encoder states (score, softmax, average), which removes the length bottleneck, gives gradients a one-hop path to every source position, and is the seed of the transformer.

</div>

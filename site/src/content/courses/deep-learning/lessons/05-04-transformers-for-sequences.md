---
id: "05.04"
course: "deep-learning"
module: "05"
slug: "transformers-for-sequences"
title: "Transformers for sequences"
keyIdeas: ["A transformer drops recurrence: self-attention, $\\text{softmax}(QK^\\top/\\sqrt{d})\\,V$, connects every pair of positions in one step, so a whole training sequence is processed in parallel at $O(T^2)$ attention cost.", "An encoder block is self-attention plus a position-wise feed-forward network, each wrapped in a residual connection and layer norm. A decoder block adds a causal mask (future scores set to $-\\infty$) and a cross-attention sublayer that reads the encoder's output.", "Pre-LN, $x + \\text{Sublayer}(\\text{LN}(x))$, trains stably without long warmup; the original post-LN layout normalises after the residual add and can get stuck when deep unless the learning rate is warmed up.", "Attention is permutation-equivariant, so order must be injected. Sinusoidal and learned embeddings are added to the inputs; rotary embeddings (RoPE) rotate queries and keys so that attention scores depend only on the relative offset.", "Three families follow from which blocks you keep: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5, the original transformer)."]
question: "RoPE is applied to queries and keys but not to values, and not added to the token embeddings like sinusoidal encodings. Why is that design chosen, and what would be lost if you instead added the rotation result to the input embeddings once at the bottom of the network?"
answer: "Rotating q and k makes each attention score a function of content and the relative offset m - n, because the two rotations combine into a single rotation by the difference: for each 2-D pair, $(R_{m\\theta} q)^\\top (R_{n\\theta} k) = q^\\top R_{(n-m)\\theta} k$. That gives every layer a clean relative-position signal exactly where it is used, in the scores, and it is reapplied in every layer so it cannot be washed out. Values are left unrotated so the information attention moves between positions is not tagged with an absolute position, and the residual stream stays position-free. If you added a positional vector to the input embeddings once, position would be mixed into the content representation, would have to survive many layers of updates, and the dot product between two such vectors, $(x_m + p_m)\\cdot(x_n + p_n)$, contains content-position cross terms such as $x_m \\cdot p_n$ that depend on absolute positions, not just their difference. The same two words the same distance apart would then score differently at the start and the end of a document. You would lose the exact relative-offset property, the model would have to learn translation invariance from data, and length generalisation beyond the trained context would be weaker."
sources: [{"title": "Attention Is All You Need (Vaswani et al., 2017)", "url": "https://arxiv.org/abs/1706.03762", "note": "The original encoder-decoder transformer; section 3 for blocks and 3.5 for sinusoidal positions."}, {"title": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "url": "https://arxiv.org/abs/2104.09864", "note": "Derivation of RoPE and its relative-position property, section 3.4."}, {"title": "The Annotated Transformer (Harvard NLP)", "url": "https://nlp.seas.harvard.edu/annotated-transformer/", "note": "Line-by-line PyTorch implementation of the full encoder-decoder."}, {"title": "On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)", "url": "https://arxiv.org/abs/2002.04745", "note": "Why post-LN needs warmup and pre-LN does not."}]
practice: ["transformer-encoder-block", "transformer-decoder-block", "rotary-position-embedding-rope", "sinusoidal-positional-encoding", "masked-causal-attention"]
tags: ["transformer", "positional encoding", "RoPE"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why the transformer drops recurrence, what it gains (parallel training, one-step paths between tokens) and what it pays ($T^2$ attention cost).
- How self-attention works as three matrix multiplies and a softmax, checked by hand on four tokens.
- What goes inside encoder and decoder blocks, and the masks that decide who sees whom.
- Why the position of layer norm (pre-LN versus post-LN) decides whether a deep model trains at all.
- Why attention has no sense of word order on its own, and the three standard fixes: sinusoidal, learned and rotary (RoPE) position encodings.
- How encoder-only, decoder-only and encoder-decoder models follow from which blocks you keep.

*Before you start:* you need attention as a soft lookup with softmax weights (05.03), residual connections (03.04) and layer normalisation (02.02).

## From attention-plus-RNN to attention only

Lesson 05.03 ended with a decoder that attends over all encoder states at every step. The recurrent loop was still there: the encoder read the source one token at a time.

The transformer (Vaswani et al., 2017) removes that loop. Every position in a sequence attends to every position in the *same* sequence, which is called **self-attention**. Two consequences follow:

- **Parallel training.** All $T$ positions are computed together with matrix multiplies, instead of $T$ dependent steps.
- **Constant path length.** Any two positions interact within one layer. In an RNN, information from token 1 reaches token 50 only after 49 updates, losing detail on the way. The price is that every layer compares all pairs, so attention compute and memory grow as $O(T^2)$.

This lesson stays at block level; multi-head shape tracing and building a full GPT are in the AI Engineering and AI Research courses.

## The idea in one picture

Take the four-token sentence "the dog bit him". Suppose each token has already been turned into a **query** vector (what this token is looking for), a **key** vector (what it offers to others) and a **value** vector (what it hands over if chosen). Two dimensions suffice:

| Token | Query q | Key k | Value v
| --- | --- | --- | --- |
| the | (0, 0) | (0, 0) | (0, 0)
| dog | (1, 1) | (2, 0) | (1, 0)
| bit | (2, 0) | (1, 1) | (0, 1)
| him | (0, 2) | (1, 0) | (1, 1)

Every token scores every token with a dot product $q \cdot k$, divided by $\sqrt{2}$; a softmax turns each row into weights that sum to 1. The result is a 4 by 4 grid of **attention weights**:

~~~dg-fig 05.04-attn.svg
Left: in an encoder every token spreads its attention over all four tokens. Right: a decoder's causal mask blanks the upper triangle, so each token can only look at itself and earlier tokens. Rows still sum to 1.
~~~

Read a row as "where this token looks". The row for "bit" puts 0.65 of its weight on "dog": the query of "bit" lines up best with the key of "dog". The row for "the" is flat (0.25 each), because its query is the zero vector and scores everything equally.

The right panel sets scores for later tokens to $-\infty$ before the softmax, so they get weight exactly 0. That is the **causal mask**, the only difference between how an encoder and a decoder attend to their own input.

## Step 1: self-attention is three matrix multiplies

In a real model, queries, keys and values are computed from the token vectors. Stack the $T$ token vectors as rows of a matrix $X$ of shape $T \times D$, where $D$ is the **model width** (for example 512). Three learned weight matrices project it:

$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V.$$

Then all outputs at once are

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V.$$

What each piece is:

- $Q K^\top$ is a $T \times T$ grid of scores: entry $(i, j)$ is how well query $i$ matches key $j$.
- $d$ is the length of each query and key vector. Dividing by $\sqrt{d}$ keeps the scores from growing with $d$; without it, the softmax saturates to a near one-hot vector and its gradients vanish.
- The softmax runs along each row, so row $i$ becomes a set of weights over all positions.
- Multiplying by $V$ gives each position a weighted average of the value vectors.

**Multi-head attention** runs $h$ of these in parallel, each with its own smaller projections of size $d = D/h$, then concatenates the $h$ results and mixes them with one more matrix $W_O$.

<div class="ex">
<div class="lab">Worked example</div>

**one row of self-attention by hand**

Take the query of "bit", $q = (2, 0)$, against the four keys. Raw dot products: the $0$, dog $2 \cdot 2 + 0 \cdot 0 = 4$, bit $2 \cdot 1 + 0 \cdot 1 = 2$, him $2 \cdot 1 + 0 \cdot 0 = 2$.

Divide by $\sqrt{2} = 1.414$: scores $(0, 2.828, 1.414, 1.414)$.

Exponentiate: $(1, 16.92, 4.11, 4.11)$, which add up to $26.15$. Divide each by the total: weights $(0.038, 0.647, 0.157, 0.157)$. They sum to 1.

Output = weighted sum of values: $0.647 \cdot (1, 0) + 0.157 \cdot (0, 1) + 0.157 \cdot (1, 1) = (0.804, 0.315)$.

Now apply the causal mask: "him" comes after "bit", so its score becomes $-\infty$ and its exponential 0. The total is $1 + 16.92 + 4.11 = 22.03$, the weights are $(0.045, 0.768, 0.187, 0)$ and the output is $(0.768, 0.187)$. These match the third row of each panel in the figure.

</div>

## Step 2: masks decide who sees whom

Both kinds of mask add $-\infty$ to a score before the softmax.

- A **padding mask** hides the filler tokens added so that sentences of different lengths fit in one batch. Without it, real words leak attention onto padding.
- A **causal mask** hides every position $j > i$ from position $i$. This is what lets a decoder train in parallel: the whole target sentence goes in at once (teacher forcing, 05.03), every position predicts its next token, and none of them can peek at the answer.

## Step 3: the encoder block

An **encoder block** wraps two sublayers, and each one is added back onto its input through a residual connection:

1. Multi-head self-attention, every position attending to every non-padding position. This mixes information *between* tokens.
2. A position-wise **feed-forward network (FFN)**, typically $D \to 4D \to D$ with a GELU or a gated variant, applied to each token separately. This processes information *within* each token.

In the **pre-LN** layout used by almost every modern model, each sublayer reads a normalised copy of its input:

$$x \leftarrow x + \text{Attn}(\text{LN}(x)), \qquad x \leftarrow x + \text{FFN}(\text{LN}(x)).$$

Here $x$ is the **residual stream**, the running $T \times D$ representation that flows from the bottom of the network to the top, and LN is layer norm. Every sublayer reads from the stream and writes a correction back into it. A rough parameter count per block is $4D^2$ for the attention projections plus $8D^2$ for the FFN, about $12D^2$ in total; with $D = 64$ that is 49,984 parameters including biases, as the code below prints.

## Step 4: where the layer norm goes

The 2017 paper used **post-LN**, which normalises *after* the residual add: $x \leftarrow \text{LN}(x + \text{Sublayer}(x))$. The stream is re-normalised at every layer, so there is no clean identity path from output to input. Xiong et al. (2020) showed that at initialisation post-LN produces large gradients for the parameters near the output, which is why it needs a careful learning-rate **warmup** (start tiny, ramp up over the first thousands of steps) and becomes unstable as depth grows.

~~~dg-fig 05.04-ln.svg
A 12-layer causal transformer learning to copy 8 tokens, trained with Adam at learning rate 0.003. Post-LN without warmup never leaves the guessing level; adding a 100-step warmup rescues it; pre-LN trains without any warmup.
~~~

In this small real run, post-LN without warmup sits at a loss of about 2.64 (uniform guessing among 14 tokens, $\ln 14$) for all 300 steps. With warmup it learns; pre-LN learns with no warmup at all. Pre-LN needs one extra layer norm after the last block, before the output head, because the stream itself is never normalised.

## Step 5: the decoder block and the three families

A **decoder block** has three sublayers, each with its own residual connection and layer norm:

1. Masked self-attention with the causal mask from step 2.
2. **Cross-attention**: queries come from the decoder's stream, keys and values come from the encoder's final output. This is the seq2seq attention of 05.03.
3. The feed-forward network.

~~~dg-flow Original encoder-decoder transformer, one block of each. Every sublayer has a residual connection and layer norm.
[Encoder block] Source tokens -> *Self-attention -> Feed-forward -> Encoder output
[Decoder block] Target so far -> *Masked self-attention -(K, V from encoder)-> Cross-attention -> Feed-forward -> Next-token logits
~~~

Which blocks you keep gives three model families:

| Family | Blocks | Attention | Typical use
| --- | --- | --- | --- |
| Encoder-only | encoder | bidirectional | classification, tagging, retrieval (BERT, lesson 06.06)
| Decoder-only | decoder without cross-attention | causal | generation, chat (GPT and most LLMs)
| Encoder-decoder | both | bidirectional encoder, causal decoder | translation, summarisation (T5, original transformer)

A decoder-only model simply drops sublayer 2: with no separate source, the prompt and the answer are one sequence under one causal mask.

## Step 6: giving attention a sense of order

Nothing in the attention formula knows where a token sits. Shuffle the input tokens and the scores, weights and outputs are shuffled in exactly the same way: self-attention is **permutation-equivariant**. Without extra help, "dog bites man" and "man bites dog" contain the same tokens and produce the same set of outputs.

How much this matters: a tiny two-layer transformer (mean-pooled for the answer) must decide whether token A appears before token B in a 10-token sequence, a question that is purely about order.

~~~dg-fig 05.04-order.svg
Test accuracy on "does A come before B?". With no position information the model is stuck at a coin flip for all 400 steps; with sinusoidal codes or RoPE it reaches 100%.
~~~

There are three standard ways to inject position.

**Sinusoidal codes** (the original paper). Add a fixed vector to each token embedding:

$$PE_{(p, 2i)} = \sin\!\big(p / 10000^{2i/D}\big), \qquad PE_{(p, 2i+1)} = \cos\!\big(p / 10000^{2i/D}\big),$$

where $p$ is the position, $i$ indexes pairs of dimensions, and $D$ is the model width. Each pair of dimensions is a clock hand turning at its own speed: low dimensions turn fast and separate neighbours, high ones turn slowly and separate distant regions.

~~~dg-fig 05.04-sinus.svg
Left: the sinusoidal code for 48 positions in 16 dimensions; each column is the vector added at one position. Right: three single dimensions of a 32-wide code, with periods of about 6, 63 and 628 positions.
~~~

**Learned absolute embeddings** (BERT, GPT-2, ViT). Keep a trainable vector per position and add it, just like a token embedding. Simple, but there is no vector for position 513 if training stopped at 512.

**Rotary embeddings (RoPE).** Instead of adding anything to the residual stream, rotate each query and key just before the dot product. Split a $d$-dimensional head vector into $d/2$ pairs and rotate pair $i$ at position $m$ by the angle $m\theta_i$, with $\theta_i = 10000^{-2i/d}$.

The point of RoPE is that rotations compose. Rotating $q$ by $m\theta$ and $k$ by $n\theta$ and taking the dot product gives the same number as rotating only one of them by $(m - n)\theta$. So the score depends on the two contents and on their **relative offset**, never on where in the sequence they sit. Values are not rotated, so what attention moves between positions carries no absolute position tag. RoPE is the default in most current LLMs.

~~~dg-fig 05.04-rope.svg
One fixed query and key, always 4 positions apart, slid along a sequence. With RoPE the score is the same everywhere; with sinusoidal codes added to the inputs, the score drifts between about 1 and 9 depending on absolute position.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**RoPE depends only on the offset**

One 2-D pair, $\theta = 0.5$. Query $q = (1, 0)$ at position 3 is rotated by $3 \times 0.5 = 1.5$ rad to $(\cos 1.5, \sin 1.5) = (0.071, 0.997)$. Key $k = (1, 0)$ at position 1 is rotated by 0.5 rad to $(0.878, 0.479)$.

Dot product: $0.071 \times 0.878 + 0.997 \times 0.479 = 0.062 + 0.478 = 0.540$, which is $\cos(1.5 - 0.5) = \cos(1.0)$.

Move both to positions 10 and 8: the angles are 5.0 and 4.0, and the dot product is $\cos(5.0 - 4.0) = 0.540$ again. Same offset, same score.

</div>

## Transformer blocks in code

A pre-LN block is a dozen lines of PyTorch. The checks after it test the two properties this lesson relies on: the causal mask really hides the future, and without positions the block is permutation-equivariant.

```python
import torch, torch.nn as nn
torch.manual_seed(0)

class PreLNBlock(nn.Module):
    def __init__(self, D=64, H=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, causal=False):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) if causal else None  # True = blocked
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]   # sublayer 1 + residual
        return x + self.ff(self.ln2(x))                                       # sublayer 2 + residual

blk = PreLNBlock()
x = torch.randn(2, 5, 64)                         # (batch, tokens, model width)
print(blk(x).shape)                               # torch.Size([2, 5, 64])
print(sum(p.numel() for p in blk.parameters()))   # 49984  (about 12 * 64**2)

# Causality: change the LAST token; earlier outputs must not move
y1 = blk(x, causal=True)
x2 = x.clone(); x2[:, -1] += 1.0
y2 = blk(x2, causal=True)
print((y1 - y2)[:, :-1].abs().max().item())       # 0.0
print((y1 - y2)[:, -1].abs().max().item() > 0)    # True

# No positions -> shuffling the input just shuffles the output
perm = torch.tensor([3, 0, 4, 1, 2])
print(torch.allclose(blk(x)[:, perm], blk(x[:, perm]), atol=1e-6))   # True
```

The built-in `nn.TransformerEncoderLayer(..., norm_first=True)` is the same pre-LN block. Note the two mask conventions: a boolean mask where `True` means "blocked", and a float mask (from `nn.Transformer.generate_square_subsequent_mask`) of 0 and $-\infty$ added to the scores. Mixing them up silently inverts the mask.

RoPE in NumPy, with a check that the score depends only on the offset:

```python
import numpy as np
def rope(x, pos, base=10000.0):           # x: (T, d) queries or keys, d even; pos: (T,)
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)          # (d/2,) one frequency per pair
    ang = pos[:, None] * theta[None, :]                # (T, d/2) angle = position * frequency
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]                    # interleaved pairs (x0,x1), (x2,x3), ...
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=(1, 64)), rng.normal(size=(1, 64))
for m, n in [(3, 1), (10, 8), (500, 498)]:
    print(m, n, round((rope(q, np.array([m])) @ rope(k, np.array([n])).T).item(), 4))
# 3 1 -8.1158
# 10 8 -8.1158
# 500 498 -8.1158
```

Two layouts exist: interleaved pairs (as above, the original paper) and split halves (dimension $j$ paired with $j + d/2$, used by many Hugging Face models). Weights trained with one layout give garbage with the other.

## Checking that it works

Cheap checks to run before training anything large:

- **Causality.** Perturb token $t$; outputs before $t$ must not change at all.
- **Order sensitivity.** If shuffling the full model's input merely shuffles its output, positions never arrived.
- **Loss at step 0.** It should be near $\ln(\text{vocabulary size})$; a loss that never leaves that value, like post-LN without warmup above, means optimisation is stuck.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Training loss near zero, generation is nonsense | causal mask missing or inverted; the model copies the answer | test that perturbing the future leaves earlier outputs unchanged
| Loss stuck at $\ln V$ in a deep model | post-LN without warmup, or learning rate too high | use pre-LN with a final LN, add warmup
| Model ignores word order | positions never added, or added after attention | add embeddings at the input, or apply RoPE to q and k in every layer
| Padding changes predictions | padding mask missing | pass a key padding mask so pads get zero weight
| Quality collapses beyond training length | positions never seen in training | RoPE scaling (position interpolation, NTK-aware)

## Going deeper

**Why RoPE gives relative positions.** Write each 2-D pair as a complex number. Rotation by angle $\phi$ is multiplication by $e^{i\phi}$. The real part of $(q e^{im\theta}) \overline{(k e^{in\theta})} = q \bar k\, e^{i(m - n)\theta}$ is the pair's contribution to the dot product, and it depends on $m$ and $n$ only through $m - n$. Summing over pairs gives the full score. Additive codes do not have this property: $(x_m + p_m) \cdot (x_n + p_n)$ contains cross terms like $x_m \cdot p_n$ that depend on absolute positions, which is the drift in the RoPE figure.

**Cost.** One layer costs about $O(T^2 D)$ for scores and weighted sums and $O(T D^2)$ for projections and the FFN, so the quadratic part takes over once $T$ exceeds roughly $D$. An RNN is $O(T D^2)$ but sequential.

**Training is parallel, generation is not.** At inference a decoder still emits one token at a time, caching keys and values of the prefix (the KV cache, covered in the Inference Engineering course).

**Length generalisation** is the weak spot. Models degrade beyond their trained context because they never saw those angles. Position interpolation scales $m$ down into the trained range; NTK-aware scaling raises the base so slow clocks stretch while fast ones barely change.

## Key terms

| Term | Meaning
| --- | --- |
| Self-attention | Every position of a sequence attends to every position of the same sequence
| Query, key, value | Learned projections: what a token looks for, what it offers, what it hands over
| Causal mask | Sets scores for future positions to $-\infty$ so a decoder cannot peek
| Cross-attention | Decoder queries attending over encoder keys and values
| Residual stream | The running per-token representation that every sublayer reads from and adds to
| Pre-LN / post-LN | Layer norm before each sublayer (stable) versus after the residual add (needs warmup)
| Permutation-equivariant | Shuffling inputs shuffles outputs the same way; attention alone ignores order
| Positional encoding | Sinusoidal or learned vectors added to the input to supply order
| RoPE | Rotating queries and keys by position-dependent angles so scores depend on relative offset

<div class="take">
<div class="lab">Key takeaway</div>

a transformer is a stack of residual blocks of attention and feed-forward layers, with masks deciding who sees whom, pre-LN keeping deep stacks trainable, and a position scheme supplying order. RoPE encodes position by rotating queries and keys, so attention depends only on relative distance.

</div>

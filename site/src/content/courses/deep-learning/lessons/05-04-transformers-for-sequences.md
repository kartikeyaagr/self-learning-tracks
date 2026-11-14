---
id: "05.04"
course: "deep-learning"
module: "05"
slug: "transformers-for-sequences"
title: "Transformers for sequences"
keyIdeas: ["A transformer drops recurrence: self-attention connects every pair of positions in one step, so a whole training sequence is processed in parallel at $O(T^2)$ attention cost.", "An encoder block is self-attention plus a position-wise MLP, each wrapped in a residual connection and layer norm. A decoder block adds a causal mask and a cross-attention sublayer that reads the encoder's output.", "Pre-LN (normalise before each sublayer) trains stably without long warmup; the original post-LN layout normalises after the residual add and is fragile when deep.", "Attention is permutation-equivariant, so order must be injected. Sinusoidal and learned embeddings are added to the inputs; rotary embeddings (RoPE) rotate queries and keys so that attention scores depend on the relative offset.", "Three families follow from which blocks you keep: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5, the original transformer)."]
question: "RoPE is applied to queries and keys but not to values, and not added to the token embeddings like sinusoidal encodings. Why is that design chosen, and what would be lost if you instead added the rotation result to the input embeddings once at the bottom of the network?"
answer: "Rotating q and k makes each attention score a function of content and the relative offset m - n, because the two rotations combine into a single rotation by the difference. That gives every layer a clean relative-position signal exactly where it is used, in the scores. Values are left unrotated so the information attention moves between positions is not tagged with an absolute position, and the residual stream stays position-free. If you added a positional vector to the input embeddings once, position would be mixed into the content representation, would have to survive many layers of updates, and the dot product between two such vectors contains content-position cross terms that depend on absolute positions, not just their difference. You would lose the exact relative-offset property and get weaker length generalisation."
sources: [{"title": "Attention Is All You Need (Vaswani et al., 2017)", "url": "https://arxiv.org/abs/1706.03762", "note": "The original encoder-decoder transformer; section 3 for blocks and 3.5 for sinusoidal positions."}, {"title": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "url": "https://arxiv.org/abs/2104.09864", "note": "Derivation of RoPE and its relative-position property, section 3.4."}, {"title": "The Annotated Transformer (Harvard NLP)", "url": "https://nlp.seas.harvard.edu/annotated-transformer/", "note": "Line-by-line PyTorch implementation of the full encoder-decoder."}, {"title": "On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)", "url": "https://arxiv.org/abs/2002.04745", "note": "Why post-LN needs warmup and pre-LN does not."}]
practice: ["transformer-encoder-block", "transformer-decoder-block", "rotary-position-embedding-rope", "sinusoidal-positional-encoding", "masked-causal-attention"]
tags: ["transformer", "positional encoding", "RoPE"]
tools: ["NumPy", "PyTorch"]
---

## From attention-plus-RNN to attention only

Lesson 05.03 ended with a decoder attending over encoder states. The transformer (Vaswani et al., 2017) asks every position in a sequence to attend to every other position in the same sequence (self-attention), and removes the recurrent loop entirely. Two consequences:

- **Parallel training**: all $T$ positions are computed at once with matrix multiplies, instead of $T$ dependent steps. This is why transformers scale to large datasets.
- **Constant path length**: any two positions interact within one layer, compared with $O(T)$ steps in an RNN. The price is $O(T^2)$ attention compute and memory per layer.

This lesson stays at block level. For the mechanics of queries, keys, values, multi-head attention and shape tracing, see the AI Engineering course (LLM internals and the transformer walkthrough), and the AI Research course for building GPT from scratch.

## Encoder and decoder blocks

An **encoder block** has two sublayers, each wrapped as $x + \text{Sublayer}(\text{LN}(x))$ in the pre-LN layout:

1. Multi-head self-attention, every position attending to every non-padding position.
2. A position-wise feed-forward network, typically $D \to 4D \to D$ with GELU or a gated variant.

A **decoder block** has three:

1. Masked self-attention: a causal mask sets scores for future positions to $-\infty$, so position $t$ sees only positions up to $t$. This is what makes teacher-forced training parallel: all target positions are predicted at once without peeking.
2. Cross-attention: queries come from the decoder, keys and values from the final encoder output. This is exactly the seq2seq attention of the previous lesson.
3. The feed-forward network.

~~~dg-flow Original encoder-decoder transformer, one block of each. Every sublayer has a residual connection and layer norm.
[Encoder block] Source tokens -> *Self-attention -> Feed-forward -> Encoder output
[Decoder block] Target so far -> *Masked self-attention -(K, V from encoder)-> Cross-attention -> Feed-forward -> Next-token logits
~~~

**Pre-LN vs post-LN.** The 2017 paper used post-LN, $\text{LN}(x + \text{Sublayer}(x))$. Xiong et al. (2020) showed that post-LN produces large gradients near the output at initialisation, which is why it needs a careful learning-rate warmup and becomes unstable when deep. Pre-LN keeps an un-normalised residual stream from input to output and trains reliably; almost all large models use it, with a final layer norm before the output head.

## Three families

| Family | Blocks | Attention | Typical use |
| --- | --- | --- | --- |
| Encoder-only | encoder | bidirectional | classification, tagging, retrieval (BERT, lesson 06.06) |
| Decoder-only | decoder without cross-attention | causal | generation, chat (GPT and most LLMs) |
| Encoder-decoder | both | bidirectional encoder, causal decoder | translation, summarisation (T5, original transformer) |

## Giving attention a sense of order

Self-attention treats its input as a set: permute the tokens and the outputs are permuted identically. Without position information, "dog bites man" and "man bites dog" look the same.

- **Sinusoidal** (original): add a fixed vector with $PE_{(p, 2i)} = \sin(p / 10000^{2i/D})$ and $PE_{(p, 2i+1)} = \cos(p / 10000^{2i/D})$ to each token embedding. Different dimensions oscillate at geometrically spaced frequencies.
- **Learned absolute**: a trainable embedding per position (BERT, GPT-2, ViT). Simple, but undefined beyond the trained length.
- **Rotary (RoPE)**: instead of adding anything to the residual stream, rotate each query and key. Split a $d$-dimensional head vector into $d/2$ pairs; rotate pair $i$ at position $m$ by angle $m\theta_i$ with $\theta_i = 10000^{-2i/d}$.

The point of RoPE is that rotations compose. If $q$ is rotated by $m\theta$ and $k$ by $n\theta$, their dot product equals the dot product of the unrotated vectors with one of them rotated by $(m - n)\theta$. Attention scores therefore depend on content and relative offset only, and the same offset means the same thing anywhere in the sequence. Values are not rotated, so the information moved by attention is position-free. RoPE is the default in most current LLMs (the Inference Engineering course covers fusing it into kernels).

```python
import numpy as np
def rope(x, pos, base=10000.0):           # x: (T, d) queries or keys, d even
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)          # (d/2,)
    ang = pos[:, None] * theta[None, :]                # (T, d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]                    # interleaved pairs
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out
```

Two layouts exist: interleaved pairs (as above, the original paper) and split halves (first half paired with second half, used by many Hugging Face models). Weights trained with one layout give garbage with the other.

<div class="ex">
<div class="lab">Worked example</div>

**RoPE depends only on the offset**

One 2-D pair, $\theta = 0.5$. Query $q = (1, 0)$ at position 3 is rotated by 1.5 rad to $(0.071, 0.997)$. Key $k = (1, 0)$ at position 1 is rotated by 0.5 rad to $(0.878, 0.479)$.

Dot product: $0.071 \times 0.878 + 0.997 \times 0.479 = 0.540 = \cos(1.0)$.

Move both to positions 10 and 8: angles 5.0 and 4.0, dot product $\cos(1.0) = 0.540$ again. Same offset, same score.

</div>

## What changes in practice

Training is parallel, but generation is still one token at a time; the KV cache that makes that efficient is covered in the Inference Engineering course. Length generalisation is the weak spot: models trained on 4k tokens degrade beyond that, and RoPE-scaling methods (position interpolation, NTK-aware scaling) stretch the angles so longer contexts map into the trained range.

<div class="take">
<div class="lab">Key takeaway</div>

a transformer is a stack of residual blocks of attention and feed-forward layers, with masks deciding who sees whom and a position scheme supplying order. RoPE encodes position by rotating queries and keys, so attention depends on relative distance.

</div>

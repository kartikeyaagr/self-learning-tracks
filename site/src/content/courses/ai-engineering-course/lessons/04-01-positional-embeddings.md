---
id: "04.01"
course: "ai-engineering-course"
module: "04"
slug: "positional-embeddings"
title: "Positional embeddings"
keyIdeas: ["Attention compares every token with every other token and ignores order. Without positions, 'dog bites man' and 'man bites dog' look the same.", "Original transformer: add fixed sinusoidal vectors (different frequencies per dimension) to token embeddings.", "GPT-2 style: learned absolute position embeddings, one vector per position up to a maximum length.", "RoPE: rotate query and key vectors by an angle proportional to position. Their dot product then depends only on relative distance.", "RoPE is used by Llama, Mistral, Qwen and most open models. ALiBi adds a distance penalty to attention scores instead.", "Context extension tricks (position interpolation, NTK scaling, YaRN) rescale RoPE frequencies so models work beyond their training length."]
question: "Why does a transformer need positional information at all, and what is the main advantage of RoPE over learned absolute position embeddings?"
answer: "Self-attention is permutation-invariant: it compares tokens by content only, so without position information word order is lost. RoPE rotates query and key vectors by angles proportional to their positions, so their dot product depends only on relative distance. Unlike learned absolute embeddings it adds no parameters, generalises better to relative structure, and can be rescaled (interpolation, NTK, YaRN) to extend context beyond the training length."
sources: [{"title": "Vaswani et al.: Attention Is All You Need (2017)", "url": "https://arxiv.org/abs/1706.03762", "note": "Section 3.5: sinusoidal positional encodings."}, {"title": "Su et al.: RoFormer / RoPE (2021)", "url": "https://arxiv.org/abs/2104.09864", "note": "Rotary position embeddings, used by most modern LLMs."}, {"title": "EleutherAI: Rotary Embeddings, a relative revolution", "url": "https://blog.eleuther.ai/rotary-embeddings/", "note": "Intuitive explanation with code."}]
tools: ["RoPE"]
---

## Attention has no sense of order

Self-attention computes a weighted mix of all tokens, where weights come from dot products between query and key vectors. Shuffle the input tokens and each token gets the same set of scores, just in a different order. On its own, attention treats a sentence as a bag of words. Position has to be injected.

## Absolute positions

**Sinusoidal (Vaswani et al., 2017)**: add to each token embedding a fixed vector made of sines and cosines at different frequencies. Low dimensions change quickly with position, high dimensions slowly, like the hands of a clock. No parameters, and in principle it works for any length.

**Learned (GPT-2, BERT)**: a trainable vector per position, up to a maximum (1,024 for GPT-2). Simple, but the model has never seen position 1,025, so it can't go beyond the training length.

## Rotary position embeddings (RoPE)

Su et al.'s RoPE changes where position goes. Instead of adding to the input, it **rotates** each query and key vector by an angle proportional to its position. Dimensions are paired, and each pair rotates at its own frequency.

~~~dg-flow RoPE: rotation turns absolute positions into relative distance.
Query at position m -(rotate by m x theta)-> q'
Key at position n -(rotate by n x theta)-> k'
q' . k' -(depends only on)-> *m - n
~~~

Because rotating both vectors and taking a dot product depends only on the angle between them, the attention score depends on the **relative distance** m - n, not on absolute positions. EleutherAI's write-up gives the geometry with code. RoPE has no learned parameters, works with the KV cache, and is used by Llama, Mistral, Qwen and most open models.

| Method | Where applied | Relative? | Used in |
| --- | --- | --- | --- |
| Sinusoidal | Added to input | Implicitly | Original transformer |
| Learned absolute | Added to input | No | GPT-2, BERT |
| RoPE | Rotates Q and K | Yes | Llama, Mistral, Qwen |
| ALiBi | Bias on attention scores | Yes | BLOOM, MPT |

## Extending context

A model trained on 8k tokens sees unfamiliar rotation angles beyond 8k and degrades. Position interpolation squeezes new positions into the trained range; NTK-aware scaling and YaRN rescale frequencies unevenly so local detail survives. With a short fine-tune these push context to 32k-128k and beyond. This is how many "long context" versions of open models are made.

## Rotate and compare

RoPE encodes position by rotating each query and key vector by an angle proportional to its position. The figure shows one pair of dimensions. Move the query position m and key position n, and watch the attention score.

~~~dg-raw
%%widget rope Rotating a query and a key by their positions; the score depends only on the offset.
~~~

The key property: if you rotate q by angle mθ and k by angle nθ, their dot product depends only on (m − n)θ. Move m and n together and the score stays fixed. Attention therefore sees relative distance for free, without any learned position table.

## The full recipe

A head of dimension d is split into d/2 pairs. Pair i rotates at its own frequency:

```
theta_i = base^(-2i / d)        base = 10,000 in the original paper, 500,000 in Llama 3
angle for pair i at position m = m * theta_i
```

The first pairs rotate fast (they distinguish neighbouring tokens), the last pairs rotate very slowly (they distinguish positions thousands of tokens apart). It is the same idea as the hands of a clock: seconds, minutes and hours together pin down a time.

## Why context extension is possible

A model trained on 8k tokens has never seen the slow pairs reach the angles that 128k positions produce. Run it at 128k directly and attention breaks down. The fixes all change the angles, not the weights:

| Method | Idea |
| --- | --- |
| Position interpolation | Divide positions by the stretch factor, so 128k positions map onto the trained 0 to 8k range |
| NTK-aware scaling | Raise the base instead, stretching slow pairs more than fast ones |
| YaRN | Treat each frequency band differently and adjust attention temperature; needs only a short fine-tune |

All of them work best with a short fine-tune on long documents. Advertised context lengths also don't guarantee the model uses the whole window well; test retrieval of facts placed at different depths before relying on it.

<div class="take">
<div class="lab">Key takeaway</div>

attention is order-blind, so position must be added. Modern LLMs use RoPE, which rotates queries and keys so attention depends on relative distance and can be stretched to longer contexts.

</div>

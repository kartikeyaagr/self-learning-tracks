---
id: "04.01"
course: "ai-engineering-course"
module: "04"
slug: "positional-embeddings"
title: "Positional embeddings"
keyIdeas: ["Attention compares every token with every other token and ignores order. Without positions, 'dog bites man' and 'man bites dog' look the same.", "Absolute schemes add a position vector to the input: fixed sinusoids (original transformer) or a learned table (GPT-2, BERT).", "RoPE rotates query and key vectors by an angle proportional to position, so their dot product depends only on the relative offset.", "RoPE is used by Llama, Mistral, Qwen, DeepSeek and most open models. ALiBi adds a distance penalty to attention scores instead.", "Context extension (position interpolation, NTK scaling, YaRN) rescales RoPE angles so a model works beyond its training length, usually after a short fine-tune."]
question: "Why does a transformer need positional information at all, and what is the main advantage of RoPE over learned absolute position embeddings?"
answer: "Self-attention is permutation-invariant: each token's output is a weighted mix of all tokens, with weights from content-only dot products, so shuffling the input just shuffles the output and word order is lost. Position has to be injected. RoPE rotates each query and key vector by angles proportional to its position, so the query-key dot product depends only on the relative offset between the two tokens. Unlike a learned absolute table it adds no parameters, has no hard maximum position, captures relative structure directly, works naturally with a KV cache (each key is rotated once, when it is written), and its angles can be rescaled with interpolation, NTK scaling or YaRN to extend context beyond the training length."
rubric: ["States that attention is permutation-invariant without position information", "Explains RoPE as rotating queries and keys by position-dependent angles", "States the score depends only on relative offset", "Gives at least one advantage over learned absolute embeddings (no table limit, relative structure, extendable, no parameters)"]
sources: [{"title": "Vaswani et al.: Attention Is All You Need (2017)", "url": "https://arxiv.org/abs/1706.03762", "note": "Section 3.5: sinusoidal positional encodings."}, {"title": "Su et al.: RoFormer / RoPE (2021)", "url": "https://arxiv.org/abs/2104.09864", "note": "Rotary position embeddings, used by most modern LLMs."}, {"title": "Peng et al.: YaRN, Efficient Context Window Extension (2023)", "url": "https://arxiv.org/abs/2309.00071", "note": "Per-frequency RoPE rescaling for long context with a short fine-tune."}, {"title": "EleutherAI: Rotary Embeddings, a relative revolution", "url": "https://blog.eleuther.ai/rotary-embeddings/", "note": "Intuitive explanation with code."}]
practice: ["rotary-position-embedding-rope", "rope-scores-with-context-extension", "sinusoidal-positional-encoding", "rotary-position-embeddings-for-decoding", "gpt2-token-and-position-embeddings"]
tools: ["RoPE"]
---

## Attention has no sense of order

Self-attention computes a weighted mix of all tokens, with weights from dot products between query and key vectors ([Attention](lesson:aieng:04.02)). Shuffle the input tokens and each token gets the same set of scores, just in a different order. On its own, attention treats a sentence as a bag of words. Position has to be injected.

## Absolute positions

**Sinusoidal (Vaswani et al., 2017).** Add to each token embedding a fixed vector of sines and cosines at different frequencies. Low dimensions change quickly with position, high dimensions slowly, like the hands of a clock. No parameters, and in principle it works for any length.

**Learned (GPT-2, BERT).** A trainable vector per position, up to a maximum (1,024 for GPT-2). Simple, but the model has never seen position 1,025, so it cannot go beyond the training length.

## Rotary position embeddings (RoPE)

Su et al.'s RoPE changes where position goes. Instead of adding to the input, it **rotates** each query and key vector by an angle proportional to its position. A head of dimension $d$ is split into $d/2$ pairs, and pair $i$ rotates at its own frequency:

$$
\theta_i = b^{-2i/d}, \qquad \text{angle of pair } i \text{ at position } m = m\,\theta_i
$$

with base $b = 10{,}000$ in the original paper and 500,000 in Llama 3. Write $R(\phi)$ for rotation by angle $\phi$. Because rotations compose, $R(m\theta)q \cdot R(n\theta)k = q \cdot R((n-m)\theta)k$: the score depends on the content of $q$ and $k$ and on the **offset** $m - n$, never on the absolute positions.

~~~dg-flow RoPE: rotation turns absolute positions into relative distance.
Query at position m -(rotate by m x theta)-> q'
Key at position n -(rotate by n x theta)-> k'
q' . k' -(depends only on)-> *m - n
~~~

Move the query and key positions below and watch the score: move both together and it stays fixed.

~~~dg-raw
%%widget rope Rotating a query and a key by their positions; the score depends only on the offset.
~~~

RoPE has no learned parameters and fits the KV cache well: each key is rotated once, when it is written, and never touched again. It is used by Llama, Mistral, Qwen, DeepSeek and most open models.

<div class="ex">
<div class="lab">Worked example</div>

**one pair of dimensions, and the full frequency range**

Take one pair with $\theta = 0.5$ rad per token, query content $q = (1, 0.5)$ and key content $k = (0.8, -0.2)$. Unrotated, $q \cdot k = 0.7$.

- Query at position 3, key at 1: rotate $q$ by 1.5 rad and $k$ by 0.5 rad. The dot product is $-0.127$.
- Query at 10, key at 8: rotate by 5.0 and 4.0 rad. Still $-0.127$: same offset, same score.
- Query at 1, key at 3: offset $-2$ instead of $+2$, and the score is $0.883$. Direction matters, not just distance.
- Same position: no relative rotation, back to $0.7$.

Now a real head, $d = 128$. The fastest pair has $\theta_0 = 1$ rad per token and repeats every $2\pi \approx 6.3$ tokens, so it separates neighbours. With $b = 10{,}000$ the slowest pair repeats every 54,000 tokens; with Llama 3's $b = 500{,}000$, every 2.6 million. The slow pairs are what tell position 3,000 from position 30,000, and why raising the base is the first step towards long context.

</div>

| Method | Where applied | Relative? | Used in |
| --- | --- | --- | --- |
| Sinusoidal | Added to input | Implicitly | Original transformer |
| Learned absolute | Added to input | No | GPT-2, BERT |
| RoPE | Rotates Q and K | Yes | Llama, Mistral, Qwen, DeepSeek |
| ALiBi | Linear distance penalty on attention scores | Yes | BLOOM, MPT |

## Extending context

A model trained on 8k tokens has never seen its slow pairs reach the angles that 128k positions produce. Run it at 128k directly and attention breaks down. The fixes change the angles, not the weights:

| Method | Idea |
| --- | --- |
| Position interpolation | Divide positions by the stretch factor, so 32k positions map onto the trained 0 to 8k range (position 20,000 behaves like 5,000) |
| NTK-aware scaling | Raise the base instead, stretching slow pairs more than fast ones so neighbour detail survives |
| YaRN | Treat each frequency band differently (leave fast pairs alone, interpolate slow ones) and adjust attention temperature |

All work best with a short fine-tune on long documents; YaRN (Peng et al., 2023) needs only a small fraction of the original training tokens. This is how many long-context versions of open models are made, for example Qwen3 extending its native 32k window with YaRN. Labs now do it as a mid-training stage ([How LLMs learn](lesson:aieng:03.02)).

## Where this breaks

- **Advertised is not usable.** A stretched window does not guarantee the model uses all of it well; test retrieval of facts placed at different depths before relying on it.
- **Mismatched scaling at inference.** Serving a YaRN-extended model without its RoPE scaling config (or with a different factor) silently degrades long inputs. Check the server's rope-scaling settings match the model card.
- **Short-text cost of static scaling.** Some servers apply the scaling factor to every request, which can slightly hurt short prompts; enable it only when you need the long window.

<div class="take">
<div class="lab">Key takeaway</div>

Attention is order-blind, so position must be added. Modern LLMs use RoPE, which rotates queries and keys so attention depends on relative offset, adds no parameters, and can be stretched to longer contexts by rescaling its angles.

</div>

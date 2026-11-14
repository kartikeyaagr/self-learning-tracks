---
id: "06.02"
course: "ai-engineering-course"
module: "06"
slug: "mixture-of-experts"
title: "Mixture of experts"
keyIdeas: ["A mixture-of-experts (MoE) layer replaces one big MLP with many expert MLPs plus a small router.", "For each token the router picks the top-k experts (often k = 1 or 2); only those run.", "Total parameters (capacity) grow with the number of experts; active parameters (compute per token) stay small.", "Training needs a load-balancing loss so tokens don't all go to a few experts.", "Serving cost: all experts must sit in memory even though each token uses a few, so MoE is memory-heavy and compute-light.", "Expert parallelism spreads experts across GPUs; tokens are sent to the GPU holding their expert (all-to-all communication)."]
question: "Mixtral 8x7B has about 47B parameters but uses about 13B per token. Explain why, and what that means for GPU memory versus compute."
answer: "Mixtral has 8 expert MLPs per layer but routes each token to only the top 2. Attention and embedding weights are shared, so total parameters are ~47B (not 8x7B=56B), while each token only touches shared weights plus 2 experts, about 13B active parameters. All ~47B must be held in GPU memory, but compute (FLOPs) per token is similar to a 13B dense model."
sources: [{"title": "Hugging Face: Mixture of Experts Explained", "url": "https://huggingface.co/blog/moe", "note": "Routers, load balancing, and serving trade-offs."}, {"title": "Fedus et al.: Switch Transformers (2021)", "url": "https://arxiv.org/abs/2101.03961", "note": "Top-1 routing to scale to a trillion parameters."}, {"title": "Jiang et al.: Mixtral of Experts (2024)", "url": "https://arxiv.org/abs/2401.04088", "note": "8 experts, top-2 routing, 47B total and 13B active parameters."}]
---

## Capacity without the compute

In a dense model every parameter is used for every token. Making the model bigger makes every token more expensive. Mixture of experts breaks that link: the model has many more parameters, but each token only passes through a few of them.

## How an MoE layer works

The attention part of the block stays the same. The MLP is replaced by N expert MLPs and a router (a small linear layer).

~~~dg-flow Top-2 routing in one MoE layer.
Token vector -(router scores)-> Pick top 2 of 8 experts -(run)-> *Expert 3, Expert 6 -(weighted sum)-> Layer output
~~~

1. The router computes a score per expert for the token.
2. Keep the top-k (Switch Transformer: k = 1; Mixtral: k = 2).
3. Run only those experts and combine their outputs, weighted by the router's softmax scores.

Hugging Face's MoE explainer notes that "experts" don't specialise in human topics like biology or law; they tend to specialise in token types and patterns, and routing differs per token and per layer.

## Total vs active parameters

Mixtral 8x7B (Jiang et al., 2024) has 8 experts per layer and top-2 routing. Attention and embeddings are shared, so it's about 47B parameters total rather than 56B, and about 13B are used per token.

| Aspect | Dense 13B | Mixtral 8x7B |
| --- | --- | --- |
| Parameters in memory | 13B | ~47B |
| Parameters used per token | 13B | ~13B |
| Compute per token | ~same | ~same |
| Quality | Lower | Close to much larger dense models |

So an MoE model **costs memory like a big model and compute like a small one**. DeepSeek-V3 pushes this further: 671B total, 37B active.

## Training problems

- **Load imbalance**: the router can collapse onto a few favourite experts. An auxiliary load-balancing loss (Switch Transformers) or bias-based balancing (DeepSeek) spreads tokens out.
- **Capacity limits**: each expert processes a bounded number of tokens per batch; overflow tokens get dropped or rerouted.
- **Instability**: router logits can blow up; tricks like router z-loss help.

## Serving MoE

- All experts must be loaded, so you need enough total GPU memory even though compute is light.
- **Expert parallelism** places different experts on different GPUs; each layer does an all-to-all exchange to send tokens to their experts and back. Network bandwidth becomes a bottleneck.
- Small batches waste the advantage: with few tokens, many experts sit idle while weights still occupy memory.

## Route tokens to experts

In a mixture-of-experts layer, a small router scores every expert for each token, and only the top k experts run. Click through the tokens below and change k.

~~~dg-raw
%%widget moe Router scores, top-k selection and the active versus total parameter split.
~~~

The economics in one line: an MoE model has the knowledge capacity of its total parameters and roughly the per-token compute of its active parameters.

| Model | Total parameters | Active per token | Experts |
| --- | --- | --- | --- |
| Mixtral 8x7B | 46.7B | 12.9B | 8, top-2 |
| DeepSeek-V3 | 671B | 37B | 256 routed + 1 shared, top-8 |

## What the router learns, and what can go wrong

The router is a linear layer: logits = x · W_router, softmax, keep the top k, and use their weights to mix the expert outputs. Left alone, it collapses: a few experts get most tokens early, get better from the extra training, and attract even more tokens. Fixes:

- **Load-balancing loss**: an extra term that penalises uneven routing across a batch (Switch Transformer, Mixtral).
- **Capacity limits**: each expert takes at most a fixed number of tokens per batch; overflow tokens skip the layer through the residual.
- **Bias-based balancing**: DeepSeek-V3 adjusts a per-expert bias during training instead of adding a loss term, reported to keep balance with less quality cost.

## Serving consequences

Active parameters set compute, but total parameters set memory: every expert must be loaded, because the next token may need any of them. Mixtral needs memory for 46.7B weights while computing like a 13B model. In large deployments experts are spread across GPUs (expert parallelism), and tokens are sent across the network to their experts and back each layer, so interconnect speed matters.

<div class="ex">
<div class="lab">Worked example</div>

**is MoE cheaper for you?**

At high batch sizes on enough GPUs, an MoE model gives near-dense quality at a fraction of the FLOPs per token.

On a single GPU with a small batch, the bottleneck is reading weights from memory, and you still have to hold every expert. A dense model of the same total size would be slower; a dense model of the same active size would fit in a quarter of the memory.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

MoE gives each token a few expert MLPs chosen by a router, so parameter count rises without raising compute per token. You pay in memory, routing complexity and communication.

</div>

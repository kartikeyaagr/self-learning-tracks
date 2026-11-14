---
id: "19.02"
course: "inference-eng-course"
module: "19"
slug: "token-routing-expert-activation"
title: "Token Routing & Expert Activation"
keyIdeas: ["The router is a small linear layer producing one score per expert.", "Top-k selection picks experts; their outputs are combined with normalized weights.", "Some models add shared experts that every token uses.", "Expert usage is uneven and depends on the input domain.", "Log per-expert token counts per layer to understand load."]
question: "Describe top-k routing for one token, and what \"expert activation statistics\" you would collect during serving."
answer: "The router computes a score per expert from the token's hidden state (a small linear layer), applies softmax (or sigmoid), picks the top-k experts, and normalizes their weights; the token's MLP output is the weighted sum of those experts' outputs. Collect per-layer counts of tokens routed to each expert per step and over time, to see load balance, hot experts, and how activation depends on the workload."
sources: [{"title": "Mixtral of Experts", "url": "https://arxiv.org/abs/2401.04088", "note": "Top-2 routing."}, {"title": "DeepSeek-V3 Technical Report", "url": "https://arxiv.org/abs/2412.19437", "note": "Fine-grained experts, shared experts and auxiliary-loss-free balancing."}]
tags: ["moe", "routing"]
---

## Routing for one token

The router is a single linear layer from the hidden state to one score per expert.

```
scores = h @ W_router                # shape [num_experts]
topk_vals, topk_idx = scores.topk(k)
weights = softmax(topk_vals)         # some models use sigmoid and renormalize
out = sum(w * experts[i](h) for w, i in zip(weights, topk_idx))
out += shared_expert(h)              # only in models with a shared expert
```

The router is cheap: a 4096 × 64 matmul per token. What it decides is expensive, because it decides which weights must be read and which GPUs must talk.

## For a batch

Running each token through its experts one by one would be very slow. The engine instead groups tokens by expert.

~~~dg-flow Batch routing in one MoE layer.
Tokens -> Router -> *Permute by expert -> Expert GEMMs -> Unpermute and combine
~~~

1. **Route:** compute top-k for every token. A batch of T tokens produces T × k (token, expert) pairs.
2. **Permute:** sort the pairs by expert, so each expert's tokens sit next to each other in memory. Record the offsets.
3. **Compute:** run each expert's MLP on its contiguous group.
4. **Unpermute and combine:** scatter results back to token order and add them, scaled by the router weights.

<div class="ex">
<div class="lab">Worked example</div>

**a batch of 6 tokens, 4 experts, k = 2**

Routing gives: t0 → (E1, E3), t1 → (E1, E2), t2 → (E0, E1), t3 → (E3, E2), t4 → (E1, E0), t5 → (E2, E3).

After permutation by expert: E0 gets [t2, t4], E1 gets [t0, t1, t2, t4], E2 gets [t1, t3, t5], E3 gets [t0, t3, t5].

The four expert GEMMs have 2, 4, 3 and 3 rows. E1 is the busiest and sets the pace if each expert runs on its own GPU.

</div>

## Activation statistics to collect

| Statistic | Why it matters |
| --- | --- |
| Tokens per expert per step | load balance and step time |
| Max ÷ mean tokens per expert or GPU | how bad the imbalance is |
| Distribution by workload (code, chat, other languages) | placement decisions per traffic type |
| Co-activation (which experts are picked together) | grouping experts to reduce traffic |
| Stability over minutes and hours | whether static placement is enough |

Training adds load-balancing losses, so on average usage is fairly even. Serving traffic is not average. A deployment that serves mostly code, or mostly one language, often shows a few experts receiving two to four times the mean.

## Deep dive: the router in equations

For token representation ⟦m945⟧ and router weights ⟦m946⟧:

⟦M943⟧

Some models skip the renormalization or use a sigmoid instead of a softmax. An engine must match the checkpoint exactly, as with RoPE.

During training, an auxiliary loss pushes routing toward balance. With ⟦m947⟧ the share of tokens sent to expert ⟦m948⟧ and ⟦m949⟧ its mean gate value:

⟦M944⟧

It is minimized when both are uniform. Training-time balance does not guarantee balance on your traffic: a code-heavy workload can concentrate on a few experts that a general mix spreads out.

~~~dg-raw
%%grid Routing a batch of 6 tokens to 4 experts with k = 2. Each column is an expert; each row is a token.
cols = E1, E2, E3, E4
key A = selected, higher gate | c1
key B = selected, lower gate | c2
t1 | A B . .
t2 | A . B .
t3 | B A . .
t4 | . A . B
t5 | A . B .
t6 | B A . .
%%
~~~

Expert 1 gets 5 tokens, expert 2 gets 4, expert 3 gets 2 and expert 4 gets 1. The layer finishes when the busiest expert does, so the step costs as if every expert had 5.

<div class="take">
<div class="lab">Key takeaway</div>

the router sends each token to a few experts, and the engine sorts tokens by expert to compute efficiently. Measure how tokens spread over experts, because it drives placement and step time.

</div>

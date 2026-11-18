---
id: "07.07"
course: "ai-engineering-course"
module: "07"
slug: "lora-and-qlora-in-practice"
title: "LoRA and QLoRA in practice"
keyIdeas: ["LoRA freezes the base weights and trains a low-rank update scaled by alpha over r; apply it to all linear layers, especially the MLP, with a learning rate about 10x what full fine-tuning would use.", "QLoRA keeps the frozen base in 4-bit NF4 and trains BF16 adapters on top: an 8B model fine-tunes in about 10 to 13 GB, a 70B model in about 50 to 55 GB.", "Data quality beats quantity: tens of good examples show an effect, hundreds to a few thousand cover most format and narrow-skill tasks, and the loss should cover only assistant tokens.", "LoRA learns less and forgets less than full fine-tuning; still run the target eval, a general-skills eval and a safety eval before and after.", "One base model can serve many adapters per request (multi-LoRA), which makes per-customer fine-tunes cheap to host."]
question: "You need to fine-tune a 70B open model on 3,000 support conversations to follow your escalation policy and output format, with one 80 GB GPU available. Choose the method and hyperparameters, estimate the memory, and describe how you would show the result is better without breaking anything else."
answer: "Use QLoRA: freeze the base in NF4 with double quantization and train rank-16 adapters on every linear layer (attention and MLP), alpha 16 to 32, learning rate about 1e-4 with warmup and cosine decay, 2 to 3 epochs, loss only on assistant tokens, gradient checkpointing on. Memory: the quantized decoder weights are about 35 GB plus about 4 GB of BF16 embeddings and output head; 207M adapter parameters with gradients and Adam states add about 3.3 GB; checkpointed activations and FP32 logits for a 2,048-token sequence add about 5 GB, plus working memory, so about 50 GB. That fits on 80 GB and would need tricks on 48 GB. Hold out 10% of conversations plus real logged cases; compare base and tuned on policy and format accuracy, on a general instruction-following set and on safety refusals, and check per-slice results. If general skills drop, lower the learning rate or epochs or mix in general data. Then serve the adapter on the shared base with multi-LoRA."
rubric: ["Chooses QLoRA (or LoRA on a quantized base) and justifies it from the memory budget", "Gives sensible rank, alpha, target modules, learning rate and epochs, and says why all linear layers", "Breaks memory into base weights, adapter states, activations and logits with roughly correct numbers", "Evaluates base versus tuned on the target task, general skills and safety, on held-out data", "Mentions serving the adapter (merged or multi-LoRA) and a mitigation for forgetting"]
sources: [{"title": "Hu et al.: LoRA: Low-Rank Adaptation of Large Language Models (2021)", "url": "https://arxiv.org/abs/2106.09685", "note": "The method: frozen weights plus a trained low-rank update."}, {"title": "Dettmers et al.: QLoRA (2023)", "url": "https://arxiv.org/abs/2305.14314", "note": "NF4, double quantization and paged optimizers; 65B fine-tuned on one 48 GB GPU."}, {"title": "Biderman et al.: LoRA Learns Less and Forgets Less (2024)", "url": "https://arxiv.org/abs/2405.09673", "note": "LoRA trails full fine-tuning on code and maths but preserves out-of-domain skills better."}, {"title": "Schulman and Thinking Machines: LoRA Without Regret (2025)", "url": "https://thinkingmachines.ai/blog/lora/", "note": "All-layer LoRA, the 10x learning-rate rule, batch-size sensitivity, rank needs for RL."}]
practice: ["lora-finetune-memory", "lora-finetune-and-merge", "nf4-quantize", "build-assistant-only-sft-loss-mask", "sft-lora-step-capstone"]
tools: ["Hugging Face PEFT", "bitsandbytes", "vLLM", "Vertex AI tuning", "Amazon Bedrock", "Tinker"]
tags: ["Fine-tuning", "LoRA"]
---

## What you are training

LoRA freezes a weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ and learns an update made of two thin matrices, $B \in \mathbb{R}^{d_{out} \times r}$ and $A \in \mathbb{R}^{r \times d_{in}}$:

$$
W' = W + \frac{\alpha}{r} B A
$$

$B$ starts at zero, so training begins exactly at the base model. Each adapted matrix adds $r(d_{in} + d_{out})$ trainable parameters. After training you can merge $BA$ into $W$ (no inference overhead) or keep it separate and swap adapters per request. QLoRA keeps $W$ in 4-bit NormalFloat (NF4), dequantizes it on the fly in each forward pass, and backpropagates through it into BF16 adapters. [Making a model useful](lesson:aieng:03.04) covers why low rank works; [LoRA and QLoRA](lesson:air-fanout-v1:10.02) in AI Research covers the theory.

~~~dg-flow One QLoRA training step.
Batch -(forward)-> *Frozen NF4 base, dequantized per layer -(plus scaled BA)-> Loss on assistant tokens -(backward)-> Gradients for A and B only -(Adam step)-> Updated adapters
~~~

## The hyperparameters that matter

| Setting | Sensible start | Evidence and notes |
| --- | --- | --- |
| Target modules | All linear layers: q, k, v, o, gate, up, down | QLoRA needed all linear layers to match full fine-tuning; Thinking Machines found attention-only LoRA underperforms MLP-only even at matched parameter count |
| Rank $r$ | 16 to 64 for SFT; 1 to 8 is enough for RL | Rank matters when the dataset is large; RL carries little information per episode |
| Alpha $\alpha$ | Equal to $r$ or $2r$, or rsLoRA's $\alpha/\sqrt{r}$ scaling | Keep $\alpha/r$ fixed when sweeping rank so the learning rate stays comparable |
| Learning rate | About $1 \times 10^{-4}$ to $2 \times 10^{-4}$, warmup, cosine decay | The best LoRA learning rate is consistently about 10× the full fine-tuning one |
| Epochs | 1 to 3 | Watch held-out loss and task metrics; more epochs mostly buys forgetting |
| Effective batch | 16 to 64 sequences | LoRA pays a larger penalty than full fine-tuning for very large batches |
| Loss mask | Assistant tokens only | Training on user and system tokens wastes capacity on text you will never generate |
| Dropout | 0 to 0.05 | Rarely the deciding factor |

The rank question has a known ceiling. Biderman et al. found that full fine-tuning on code and maths learns weight changes of 10 to 100 times higher rank than typical LoRA settings. With about 100k instruction pairs or billions of tokens of continued pretraining, LoRA clearly trails full fine-tuning. App tasks (a format, a policy, a tone, a classifier) are far smaller than that, and in that regime LoRA matches full fine-tuning.

## Data: how much and what kind

Hosted fine-tuning docs report visible gains from 50 to 100 good examples; LIMA aligned a 65B model with 1,000 carefully chosen ones. For a format or policy task, a few hundred to a few thousand clean, consistent examples is the usual range. Quality problems hurt more than small size. Watch for contradictory labels, answers that break your own format, and duplicated prompts that leak across the train and test split. Use real inputs from logs wherever you can, and write or generate the outputs with the care of a golden set ([data for fine-tuning](lesson:aieng:07.08)). Hold out at least 10% plus a set of real production cases the model never sees.

## Memory arithmetic: 8B and 70B

<div class="ex">
<div class="lab">Worked example</div>

**QLoRA, rank 16 on all linear layers, batch of one sequence, gradient checkpointing.** Llama-3.1-8B-style shapes (32 layers, hidden 4,096, MLP 14,336, 8 KV heads of 128, vocabulary 128,256) and Llama-3.3-70B-style shapes (80 layers, hidden 8,192, MLP 28,672). NF4 with double quantization costs 4.127 bits per weight; embeddings and output head stay in BF16. Adapter parameters cost 16 bytes each (BF16 weight and gradient, FP32 master copy and two Adam moments).

| Component | 8B | 70B |
| --- | --- | --- |
| Frozen decoder weights (NF4) | 6.98B × 4.127/8 = 3.6 GB | 68.5B × 4.127/8 = 35.3 GB |
| Embeddings + LM head (BF16) | 1.05B × 2 = 2.1 GB | 2.1B × 2 = 4.2 GB |
| Adapter parameters | 41.9M (0.52%) | 207M (0.29%) |
| Adapter weights, grads, Adam | 0.67 GB | 3.3 GB |
| Checkpointed activations, 2k / 4k tokens | 0.5 / 1.1 GB | 2.7 / 5.4 GB |
| FP32 logits plus their gradient, 2k / 4k | 2.1 / 4.2 GB | 2.1 / 4.2 GB |
| Recompute and runtime overhead (rough) | 1 to 2 GB | 2 to 3 GB |
| **Total, 2k / 4k tokens** | **about 10.5 / 13 GB** | **about 50 / 55 GB** |
| Full fine-tuning, states alone | 8.03B × 16 = 128 GB | 70.6B × 16 = 1.13 TB |

The 8B run fits a 16 GB GPU at 2k tokens and is comfortable on 24 GB. The 70B run fits one 80 GB H100 and misses a 48 GB card by a few gigabytes. QLoRA's original 65B-on-48 GB result used a 32k vocabulary, shorter sequences and paged optimizers. The surprise line is the logits: a 128k vocabulary makes the output layer's activations bigger than every other activation combined at these lengths, which is why fused or chunked cross-entropy kernels are a standard memory trick. The adapter you ship is small: 84 MB for the 8B model in BF16, 414 MB for the 70B.

</div>

## Hosted fine-tuning

Hosted services take a JSONL file of chat examples, a base model and a few settings (epochs, a learning-rate multiplier, sometimes adapter rank). They bill per training token and return a model id. You usually pay a higher per-token price to call the tuned model, or for dedicated hosting. As of November 2026: Vertex AI offers supervised tuning for Gemini 2.5 and 3.x Flash-class models; Amazon Bedrock offers customization including reinforcement fine-tuning on a few models; open-model platforms run LoRA jobs on Qwen, Llama and gpt-oss and serve the adapters; Thinking Machines' Tinker exposes the training loop itself (forward-backward, optimizer step, sampling) over LoRA on open weights. OpenAI is winding its fine-tuning platform down.

The trade-off is lock-in. With a closed base model you never get the weights, and the tuned model is retired with its base. With open weights you keep the adapter, can retrain on a new base, and can serve it anywhere.

## Forgetting: measure it, then limit it

Fine-tuning moves weights the rest of the model relied on. Luo et al. measured forgetting of domain knowledge, reasoning and reading comprehension in 1B to 7B models during continual instruction tuning. Qi et al. showed that fine-tuning on benign data can weaken safety refusals. Biderman et al. found the LoRA trade-off that gives "learns less, forgets less" its name: LoRA holds out-of-domain performance better than full fine-tuning and better than weight decay or dropout used as regularisers, and keeps generations more diverse.

Run three evals on the base and the tuned model, on the same held-out data:

1. **Target task**, split by slice (per policy rule, per field).
2. **General skills** you rely on: instruction following, the other tasks this endpoint serves, long-input handling.
3. **Safety and refusals**, with the same prompts as before.

If the target improves and the others hold within noise ([regression testing](lesson:aieng:10.06)), ship. If general skills drop, the usual fixes are a lower learning rate, fewer epochs, a lower rank, or mixing 5 to 20% general instruction data into training.

## Serving many adapters

A merged adapter is just a new model: no overhead, but one deployment per fine-tune. Keeping adapters separate lets one base serve many: the engine batches requests for different adapters together and applies each request's $BA$ with dedicated kernels (Punica, S-LoRA). In vLLM that is `--enable-lora` with `--max-loras` (adapters resident per batch) and `--max-lora-rank`; requests choose an adapter by name. A per-customer fine-tune then costs storage plus a few percent of throughput, not a GPU each. [Multi-LoRA serving](lesson:ieng:26.02) covers the scheduling. One caution: serve a QLoRA-trained adapter on the same quantized base it was trained against, or re-check quality on BF16 weights, because the adapter learned to correct a specific quantization error.

## Where this breaks

- **Eval only on the training distribution.** A held-out split from the same synthetic generator flatters the model. Include real production cases.
- **Chat template mismatch.** Training on one template and serving with another silently degrades output. Use the tokenizer's own template in both places.
- **Learning rate copied from full fine-tuning.** At $10^{-5}$ LoRA barely moves; the 10× rule exists for a reason.
- **Fine-tuning MoE models.** Adapters must cover expert MLPs to work well, the router is usually frozen, and the checkpoint is as large as the total parameter count ([MoE for app engineers](lesson:aieng:06.06)).

## What interviewers probe

- Trainable parameters for a given rank and model, and memory for LoRA versus QLoRA versus full fine-tuning.
- Why target all linear layers, and how alpha interacts with rank and learning rate.
- How you show a fine-tune helped and did not break general or safety behaviour.
- Merged versus multi-LoRA serving, and when each fits.

<div class="take">
<div class="lab">Key takeaway</div>

For app-sized tasks, QLoRA on all linear layers at rank 16 to 64 with a learning rate near $10^{-4}$ matches full fine-tuning at a fraction of the memory: about 13 GB for 8B, about 55 GB for 70B at 4k tokens. The work that decides success is the data, an eval of the base and tuned model on the target task, general skills and safety, and a serving plan for the adapter.

</div>

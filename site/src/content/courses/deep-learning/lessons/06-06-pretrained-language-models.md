---
id: "06.06"
course: "deep-learning"
module: "06"
slug: "pretrained-language-models"
title: "Pretrained language models"
keyIdeas: ["Pretrain once on unlabelled text, fine-tune cheaply per task: a pretrained encoder plus a small head beats task-specific architectures trained from scratch on most NLP benchmarks.", "BERT's masked language modelling picks 15% of tokens and predicts the originals; of those, 80% become [MASK], 10% a random token and 10% stay unchanged, because [MASK] never appears at fine-tuning time.", "Next-sentence prediction asked whether segment B followed segment A. RoBERTa showed that dropping it, and training longer on more data with dynamic masking, works better.", "For classification, put a linear head on the final [CLS] vector (or mean-pooled tokens) and fine-tune all weights with a small learning rate (2e-5 to 5e-5) for 2 to 4 epochs.", "Encoders see both directions, so they suit understanding tasks; they do not generate text naturally, which is what decoder-only models are for."]
question: "Suppose you pretrain BERT but replace every chosen token with [MASK] (100/0/0 instead of 80/10/10). What goes wrong when you fine-tune and use the model, and what specific job does the 10% 'left unchanged' bucket do?"
answer: "With 100% [MASK], the model only ever has to make predictions at [MASK] positions, so it can learn that ordinary visible tokens never need a carefully built representation of their own identity: the useful computation is focused on [MASK] slots. At fine-tuning and inference there are no [MASK] tokens, so every input looks unlike pretraining, and the representations of real tokens are less suited to the downstream task (a pretrain-finetune mismatch). The 10% random replacement means any visible token might be wrong, so the model must keep a contextual representation for every position. The 10% unchanged bucket means a visible token is sometimes also the correct answer, which teaches the model that the observed word is informative and anchors each position's representation toward the actual token, which is exactly the situation at fine-tuning time."
sources: [{"title": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)", "url": "https://arxiv.org/abs/1810.04805", "note": "MLM and NSP (section 3.1), the 80/10/10 ablation (appendix C.2) and fine-tuning hyperparameters (appendix A.3)."}, {"title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "url": "https://arxiv.org/abs/1907.11692", "note": "Ablations of NSP, dynamic masking, batch size and data scale."}, {"title": "Hugging Face Transformers: Fine-tuning a pretrained model", "url": "https://huggingface.co/docs/transformers/training", "note": "The practical classification fine-tuning loop with Trainer or plain PyTorch."}]
practice: ["masked-language-modeling", "mutation-trap", "next-sentence-prediction", "masked-mean-pooling"]
tags: ["BERT", "masked language modelling", "fine-tuning"]
tools: ["PyTorch", "Hugging Face Transformers"]
---

## Pretrain, then fine-tune

Until 2018, most NLP systems were trained from scratch per task, starting at best from static word vectors (lesson 06.04). BERT (Devlin et al., 2018) made a different recipe standard: pretrain a large transformer encoder on unlabelled text with a self-supervised objective, then fine-tune the whole network with a small task head on a few thousand labelled examples. Contextual vectors replaced static ones: "bank" gets a different representation in "river bank" and "bank account" because every layer attends over the whole sentence.

This lesson covers encoder pretraining and classic fine-tuning. Decoder-only LLMs and parameter-efficient fine-tuning (LoRA and friends) are covered in the AI Research course.

~~~dg-flow The two phases. Only the small head is new at fine-tuning time; every pretrained weight is updated too.
[Pretrain] Unlabelled text -(mask 15%)-> Corrupted tokens -> *Transformer encoder -> Predict original tokens
[Fine-tune] Labelled sentence -> Pretrained encoder -(final CLS vector)-> *Linear head -> Class label
~~~

## BERT's input format

- WordPiece vocabulary of about 30,000 tokens (lesson 06.01), maximum length 512.
- Input: `[CLS] segment A [SEP] segment B [SEP]`. Each token's input vector is the sum of a token embedding, a learned position embedding and a segment embedding (A or B).
- BERT-base: 12 layers, width 768, 12 heads, 110M parameters. BERT-large: 24 layers, width 1024, 16 heads, 340M.

## Masked language modelling

A left-to-right LM cannot use the right context, and a bidirectional encoder trained to predict the next token would trivially see the answer. MLM sidesteps this: corrupt some input tokens and train the encoder to recover them using context on both sides.

1. Choose 15% of token positions at random.
2. Of those, replace 80% with `[MASK]`, 10% with a random token, and leave 10% unchanged.
3. Compute cross-entropy only at the chosen positions; all others get the ignore label.

```python
import numpy as np
def mlm_corrupt(ids, mask_id, vocab_size, special, rng, p=0.15):
    ids = ids.copy()
    labels = np.full_like(ids, -100)                   # -100 = ignored by the loss
    cand = ~np.isin(ids, special)                      # never mask [CLS], [SEP], [PAD]
    chosen = cand & (rng.random(ids.shape) < p)
    labels[chosen] = ids[chosen]                       # copy labels before corrupting
    r = rng.random(ids.shape)
    ids[chosen & (r < 0.8)] = mask_id
    rand = chosen & (r >= 0.8) & (r < 0.9)
    ids[rand] = rng.integers(0, vocab_size, rand.sum())
    return ids, labels                                 # remaining 10% left unchanged
```

Why not always use `[MASK]`? Fine-tuning inputs never contain `[MASK]`, so a model that only ever predicted at `[MASK]` positions would learn representations specialised for a token it will never see again. The random 10% forces the model to keep a good representation of every token, since any token might be corrupted; the unchanged 10% biases representations toward the actual observed word. Copying labels before corrupting matters: overwrite first and the "labels" become the corrupted ids.

<div class="ex">
<div class="lab">Worked example</div>

**how much signal one sequence gives**

A 512-token sequence has about $0.15 \times 512 \approx 77$ prediction targets: about 61 `[MASK]`, 8 random replacements, 8 unchanged.

A causal LM on the same sequence predicts 511 targets. MLM learns from about 15% as many positions per sequence, one reason it needs long training, and the motivation for ELECTRA, which trains a discriminator on every position.

</div>

## Next-sentence prediction

BERT also trained a binary classifier on the final `[CLS]` vector: is segment B the actual next segment after A (50%), or a random segment from another document (50%)? The aim was to help sentence-pair tasks like entailment and QA. RoBERTa (Liu et al., 2019) found that removing NSP and packing full-length contiguous text matched or improved results, and that most of BERT's headroom came from training longer, on 10× more data, with bigger batches and dynamic masking (a fresh mask each time a sequence is seen). The lesson generalises: an auxiliary objective that looks sensible can be neutral or harmful, so ablate it.

## Fine-tuning for classification

Add a linear layer on top of the final `[CLS]` vector (or the masked mean of all token vectors) and train everything end to end with cross-entropy. The BERT paper's ranges remain good defaults:

- Learning rate 2e-5 to 5e-5 with AdamW, linear warmup over about 10% of steps, then linear decay.
- 2 to 4 epochs; batch size 16 or 32.
- A larger learning rate quickly destroys the pretrained features (catastrophic forgetting) and often gives a model that predicts one class.

On small datasets (a few thousand examples) fine-tuning is unstable: different random seeds, which change the head's initialisation and the data order, can give very different dev scores. Run several seeds and report the mean and spread; train a bit longer with warmup and bias-corrected Adam, both of which were found to reduce the variance.

Other heads use the same encoder: a per-token linear layer for tagging (lesson 06.05), start and end logits for span extraction, or two sentences in one input for pair classification.

## Encoders versus decoders

An encoder's bidirectional attention is ideal for understanding tasks: classification, tagging, retrieval embeddings, reranking. It has no natural way to generate text left to right. Decoder-only models trained with next-token prediction generate naturally and, at scale, solve many classification tasks by prompting. For a high-volume, fixed-label task, a fine-tuned 110M-parameter encoder is still often the cheapest accurate option.

<div class="take">
<div class="lab">Key takeaway</div>

masked language modelling turns unlabelled text into a bidirectional encoder whose contextual vectors transfer to almost any understanding task. Fine-tune all of it gently with a small head, and treat seed variance on small datasets as part of the result.

</div>

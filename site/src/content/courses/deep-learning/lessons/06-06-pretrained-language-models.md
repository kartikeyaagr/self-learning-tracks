---
id: "06.06"
course: "deep-learning"
module: "06"
slug: "pretrained-language-models"
title: "Pretrained language models"
keyIdeas: ["**Pretrain once, fine-tune cheaply:** a transformer encoder learns from unlabelled text, then a small head plus a short, gentle round of training adapts it to each labelled task, beating models trained from scratch, most of all when labels are scarce.", "BERT's **masked language modelling** picks 15% of tokens and predicts the originals; of those, 80% become `[MASK]`, 10% a random token and 10% stay unchanged, because `[MASK]` never appears at fine-tuning time.", "The output vectors are **contextual**: \"bank\" gets a different vector next to \"river\" than next to \"money\", because every layer attends over the whole sentence in both directions.", "Next-sentence prediction asked whether segment B followed segment A. RoBERTa showed that dropping it, and training longer on more data with dynamic masking, works better.", "For classification, put a linear head on the final `[CLS]` vector (or masked-mean-pooled tokens) and fine-tune all weights with a small learning rate (2e-5 to 5e-5), warmup, and 2 to 4 epochs; run several seeds, because small-data fine-tuning is unstable.", "Encoders see both directions, so they suit understanding tasks (classification, tagging, retrieval); they do not generate text naturally, which is what decoder-only models are for."]
question: "You pretrain a BERT-style encoder but replace every chosen token with [MASK] (100/0/0 instead of 80/10/10). You then fine-tune it on 2,000 labelled support tickets with a learning rate of 1e-3 and get a model that labels every ticket with the most common class. Explain what the 100/0/0 change does to the representations, what specific job the 10% 'left unchanged' bucket does, and what most likely caused the one-class model and how you would fix and verify it."
answer: "With 100% [MASK], the model only ever has to make predictions at [MASK] positions, so it can learn that ordinary visible tokens never need a carefully built representation of their own identity: the useful computation is focused on [MASK] slots. At fine-tuning and inference there are no [MASK] tokens, so every input looks unlike pretraining, and the representations of real tokens are less suited to the downstream task (a pretrain-finetune mismatch). The 10% random replacement means any visible token might be wrong, so the model must keep a contextual representation for every position. The 10% unchanged bucket means a visible token is sometimes also the correct answer, which teaches the model that the observed word is informative and anchors each position's representation toward the actual token, which is exactly the situation at fine-tuning time. The one-class model is a separate problem: 1e-3 is roughly 20 to 50 times the usual fine-tuning range. The randomly initialised head produces large, noisy gradients that, at that step size, overwrite the pretrained features (catastrophic forgetting), and the cheapest remaining solution is to predict the majority class. Fix it with AdamW at 2e-5 to 5e-5, linear warmup over about 10% of steps followed by linear decay, and 2 to 4 epochs at batch size 16 or 32. Verify by checking the predicted class distribution on the dev set (it should roughly match the true one), comparing against a TF-IDF plus logistic regression baseline, and running 3 to 5 seeds, reporting mean and spread because small-data fine-tuning varies a lot between seeds. Restoring 80/10/10 in pretraining would further improve how well the representations transfer."
sources: [{"title": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)", "url": "https://arxiv.org/abs/1810.04805", "note": "MLM and NSP (section 3.1), the 80/10/10 ablation (appendix C.2) and fine-tuning hyperparameters (appendix A.3)."}, {"title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "url": "https://arxiv.org/abs/1907.11692", "note": "Ablations of NSP, dynamic masking, batch size and data scale."}, {"title": "Hugging Face Transformers: Fine-tuning a pretrained model", "url": "https://huggingface.co/docs/transformers/training", "note": "The practical classification fine-tuning loop with Trainer or plain PyTorch."}]
practice: ["masked-language-modeling", "mutation-trap", "next-sentence-prediction", "masked-mean-pooling"]
tags: ["BERT", "masked language modelling", "fine-tuning"]
tools: ["PyTorch", "Hugging Face Transformers"]
---

## What you will learn

- Why "pretrain on raw text, then fine-tune per task" replaced training from scratch.
- How BERT's **masked language modelling** turns unlabelled text into a training signal, and why the 80/10/10 rule exists.
- What BERT's input looks like: `[CLS]`, `[SEP]` and three summed embeddings.
- What next-sentence prediction was, and what RoBERTa's ablations taught about it.
- How to fine-tune for classification: head, learning rate, warmup, seeds.
- When an encoder is the right tool and when you want a decoder-only model instead.

*Before you start:* subword tokenisation (06.01), static word embeddings (06.04), per-token heads (06.05), and the transformer encoder from module 05. Cross-entropy and AdamW come from modules 01 and 02.

## From static vectors to a pretrained encoder

Lesson 06.04 gave every word one vector, shared by all its uses: "bank" by a river and "bank" with money get the same input. Lesson 06.05 then trained a tagger on top, mostly from scratch.

This lesson pretrains the *whole network*, not just the first layer. A transformer encoder learns from billions of words of unlabelled text, and every task afterwards starts from those weights. This is **transfer learning**, the same move as starting an image classifier from ImageNet weights, and BERT (Devlin et al., 2018) made it the default in NLP.

This lesson covers encoder pretraining and classic fine-tuning. Decoder-only LLMs and parameter-efficient fine-tuning (LoRA and friends) are covered in the AI Research course.

## The idea in one picture

Hide a few words of an unlabelled sentence and ask a network to fill them in, using the words on *both* sides. To do this well across millions of sentences it must learn grammar, meaning and which words go together, without a single human label. That is **pretraining**. Then put one small new layer on top and train the whole thing briefly on your labelled task. That is **fine-tuning**.

~~~dg-flow The two phases. Only the small head is new at fine-tuning time; every pretrained weight is updated too.
[Pretrain] Unlabelled text -(mask 15%)-> Corrupted tokens -> *Transformer encoder -> Predict original tokens
[Fine-tune] Labelled sentence -> Pretrained encoder -(final CLS vector)-> *Linear head -> Class label
~~~

This lesson runs a miniature version of the recipe on a CPU. The "language" is a toy review language: 12 positive adjectives, 12 negative ones, 12 nouns, negation, and some sentences about river banks and money banks. A 2-layer encoder (width 64) is pretrained with masked language modelling on 40,000 unlabelled sentences, 2,000 steps that took about five and a half minutes on one shared CPU core. Then it is fine-tuned to label single clauses such as "the pasta was not fresh" as positive or negative, using only a handful of labelled sentences.

~~~dg-fig 06.06-transfer.svg
Same network, same labelled data, same training steps. The only difference is the starting point: weights from masked-language-model pretraining (accent) or random numbers (orange). Each faint dot is one random seed.
~~~

Read the figure from left to right. With 8 labelled sentences the pretrained model averages 85% test accuracy (seeds range from 54% to 100%), while the randomly initialised one is at 53%, barely above a coin flip. With 32 labels it is 99% against 65%; with 64 it is 100% against 73%. Only at 256 labels does the from-scratch network catch up (99%).

Why such a gap? To classify "the bread was not stale", the from-scratch model must learn from labels alone that "stale" is negative, and with 8 examples most adjectives never appear. The pretrained model already knows which adjectives behave alike, and how "not" flips them, from filling blanks. The labels only have to name the clusters. That is the promise of pretraining: **labels become cheap because most of the knowledge came from text that needed none.** Real BERT showed the same pattern at scale, beating task-specific architectures on most standard benchmarks, with the biggest gains on small datasets.

## Step 1: BERT's input format

- **Tokens.** BERT uses a **WordPiece** vocabulary of about 30,000 subword pieces (lesson 06.01), so rare words split into known pieces. The maximum length is 512 tokens.
- **Special tokens.** Every input starts with `[CLS]` ("classification") and each segment ends with `[SEP]` ("separator"): `[CLS] segment A [SEP] segment B [SEP]`. A single sentence uses only segment A. `[PAD]` fills short sequences to a common length and is hidden from attention.
- **Three embeddings, added.** Each position's input vector is the sum of a **token embedding** (which piece is here), a learned **position embedding** (where it is: 0 to 511), and a **segment embedding** (A or B). The width stays 768.

There is no causal mask: every position attends to every other position, left and right, in every layer.

~~~dg-fig 06.06-masks.svg
Which positions each token may look at. In an encoder (left) the row for "bank" covers "river", so its vector can depend on it. In a decoder (right) the same row stops at "bank" itself.
~~~

Two standard sizes, from the paper:

| Model | Layers | Width | Heads | Parameters
| --- | --- | --- | --- | --- |
| BERT-base | 12 | 768 | 12 | 110M
| BERT-large | 24 | 1024 | 16 | 340M

The output is one vector per input position. These are **contextual embeddings**: the vector at "bank" is computed from the whole sentence, so it differs between "river bank" and "bank account".

## Step 2: masked language modelling

The language model of lesson 06.02 predicts the next word from the words before it. A bidirectional encoder asked to do that would simply attend to the next word: the answer is visible. **Masked language modelling (MLM)** hides the answer instead. Corrupt some input tokens and train the encoder to recover the originals:

1. Choose 15% of token positions at random (never `[CLS]`, `[SEP]` or `[PAD]`).
2. Of the chosen positions, replace 80% with `[MASK]`, replace 10% with a random token, and leave 10% unchanged.
3. Compute the cross-entropy only at the chosen positions. Every other position gets the "ignore" label.

In symbols, with $M$ the set of chosen positions, $\tilde x$ the corrupted sequence and $x_i$ the original token at position $i$:

$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|}\sum_{i \in M} \log p_\theta(x_i \mid \tilde x).$$

Here $p_\theta(\cdot \mid \tilde x)$ is the encoder's softmax over the vocabulary at position $i$, and $\theta$ is all the weights. The output layer reuses the token embedding matrix (the weights are **tied**).

~~~dg-fig 06.06-mlm.svg
Left: three chosen positions in one sentence, one per bucket. Note the unchanged token: it still counts as a target. Right: how many loss terms one 512-token sequence produces. MLM gets about 77 (measured over 10,000 simulated sequences, split about 61 / 8 / 8 across the buckets); a next-token model gets 511.
~~~

Number check: with two special tokens, $0.15 \times 510 = 76.5$ targets on average; the simulation gave 76.5 with a standard deviation of 8.

**Why not always use `[MASK]`?** Real inputs never contain `[MASK]`. With only `[MASK]` corruption, the model learns that only `[MASK]` slots need careful prediction, and tunes its representations to a token it will never see again. The other buckets fix this:

- The 10% **random** replacements mean any visible token might be wrong, so the model must keep a good contextual representation of *every* position and check it against the context.
- The 10% **unchanged** tokens mean a visible word is sometimes exactly the answer, so representations stay anchored to the observed word, which is the situation at fine-tuning time.

In the BERT paper's ablation, differences after fine-tuning were small; 100% `[MASK]` hurt most when frozen BERT vectors fed a tagger directly, exactly this mismatch.

<div class="ex">
<div class="lab">Worked example</div>

**one MLM step by hand**

Sentence: `[CLS] the soup was not fresh and the staff were rude [SEP]`, 12 tokens of which 10 are ordinary. Expected targets: $0.15 \times 10 = 1.5$.

Suppose two positions are chosen (the figure shows three): "fresh" (position 5) and "rude" (position 10). A random draw puts "fresh" in the 80% bucket, so the input shows `[MASK]` there; "rude" lands in the 10% unchanged bucket, so the input still shows "rude".

Labels are $-100$ everywhere except positions 5 and 10, copied *before* the input was overwritten.

The encoder assigns probability 0.40 to "fresh" at position 5 and 0.25 to "rude" at position 10. The two loss terms are $-\ln 0.40 = 0.916$ and $-\ln 0.25 = 1.386$.

The loss averages over chosen positions only: $(0.916 + 1.386)/2 = 1.151$. The other ten contribute nothing.

For scale: a uniform guess over 30,522 pieces costs $\ln 30522 = 10.33$ per target.

</div>

What the toy model learned is visible in its weights. Nobody told it which adjectives are positive, but "the soup was tasty and the pizza was ..." only makes sense with a positive word in the gap.

~~~dg-fig 06.06-learned.svg
Left: the 24 adjective embeddings of the toy model after pretraining, projected to 2-D. The two polarities separate although no label was ever used. Right: the final-layer vector at the token "bank" in 120 sentences. One input embedding becomes two clouds depending on context.
~~~

Asked to fill "the soup was bland and the pizza was not `[MASK]`", it guessed lovely, excellent, good and pleasant: "not" flips polarity, learned from raw text. The "bank" vectors have mean cosine similarity 0.95 within a sense and 0.86 across senses; each sense forms three groups (one per toy sentence template) on its own side of the first principal direction. A static vector would be a single point.

## Step 3: next-sentence prediction, and what RoBERTa changed

BERT also trained **next-sentence prediction (NSP)**: in `[CLS] A [SEP] B [SEP]`, B was the real next segment half the time and a random one otherwise, and a classifier on the final `[CLS]` vector guessed which. It was meant to help sentence-pair tasks such as entailment and question answering.

RoBERTa (Liu et al., 2019) re-ran the recipe carefully and found:

- Removing NSP and packing full-length runs of contiguous text matched or slightly improved downstream results.
- **Dynamic masking**, drawing a fresh mask every time a sequence is seen, was as good as or slightly better than static masking, where the masks are drawn once during preprocessing (as in the original BERT code) and reused every epoch.
- Most of the gain came from training longer, on about 10× more text, with much bigger batches.

The general lesson outlives BERT: an auxiliary objective that sounds sensible can be neutral or harmful, so ablate it before you keep it.

## Step 4: fine-tuning for classification

Fine-tuning reuses the whole encoder and adds a **task head**:

1. Tokenise in the pretraining format: `[CLS] sentence [SEP]`, or `[CLS] A [SEP] B [SEP]` for a pair.
2. Take the final vector at `[CLS]`, $h_{\text{CLS}} \in \mathbb{R}^{768}$, or the masked mean of all token vectors (`[PAD]` excluded).
3. A new linear layer gives class scores $z = W h_{\text{CLS}} + b$, with $W$ of shape (classes × 768): for 2 classes, 1,538 new numbers on top of 110M.
4. Train everything end to end with cross-entropy, gently.

The BERT paper's ranges remain good defaults:

- Learning rate 2e-5 to 5e-5 with AdamW.
- Linear **warmup** over about the first 10% of steps, then linear decay to zero.
- 2 to 4 epochs, batch size 16 or 32.

Warmup matters because the head starts random: its first gradients are large and noisy, and at full learning rate they would push every pretrained weight in a random direction.

<div class="ex">
<div class="lab">Worked example</div>

**a fine-tuning schedule in numbers**

3,000 labelled sentences, batch 32, 3 epochs. Steps per epoch: $\lceil 3000/32 \rceil = \lceil 93.75 \rceil = 94$, so 282 steps in total.

Warmup is 10% of steps: $0.1 \times 282 = 28.2$, round to 28.

At step 14 (halfway through warmup) the learning rate is $2\times10^{-5} \times 14/28 = 1\times10^{-5}$.

At step 155 it is decaying: $2\times10^{-5} \times (282 - 155)/(282 - 28) = 2\times10^{-5} \times 127/254 = 1\times10^{-5}$ again. It reaches 0 at step 282.

</div>

~~~dg-fig 06.06-lr.svg
Left: the schedule from the example. Right: the toy encoder fine-tuned with 64 labels at different peak learning rates (tiny models tolerate larger rates than BERT-base, so the scale is shifted). Up to 1e-2 every run stays near 100%; above that, accuracy falls towards the from-scratch level (dashed).
~~~

A learning rate that is too large overwrites the pretrained features (**catastrophic forgetting**). In real BERT fine-tuning the typical symptom is a model that collapses to predicting one class. The toy model shows the trend: with 64 labels, a peak rate of 3e-2 cut mean accuracy to 88% and 0.1 cut it to 78%, close to the 73% the same network reaches from random weights. BERT-base's safe range is far lower, but the shape is the same: a broad plateau, then a cliff.

Other tasks swap the head: per-token logits for tagging (06.05), start and end scores for answer spans, or a sentence pair in one input.

## Pretraining and fine-tuning in code

The two core pieces, in PyTorch, on random token ids (no download needed). The encoder is a 2-layer, width-128 stand-in for BERT.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
V, D, LMAX = 30522, 128, 64                      # BERT vocabulary size, a small width
PAD, CLS, SEP, MASK = 0, 101, 102, 103           # BERT's ids for the special tokens

def mlm_corrupt(ids, p=0.15):
    ids = ids.clone()
    labels = torch.full_like(ids, -100)          # -100 = ignored by cross_entropy
    chosen = (ids > MASK) & (torch.rand(ids.shape) < p)   # never pick special tokens
    labels[chosen] = ids[chosen]                 # copy labels BEFORE corrupting
    r = torch.rand(ids.shape)
    ids[chosen & (r < 0.8)] = MASK
    rnd = chosen & (r >= 0.8) & (r < 0.9)
    ids[rnd] = torch.randint(1000, V, (int(rnd.sum()),))
    return ids, labels                           # remaining 10% left unchanged

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos, self.seg = nn.Embedding(V, D), nn.Embedding(LMAX, D), nn.Embedding(2, D)
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
    def forward(self, ids, seg):
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1])) + self.seg(seg)
        return self.layers(h, src_key_padding_mask=(ids == PAD))   # (batch, length, D)

enc = Encoder()
ids = torch.randint(1000, V, (8, 32)); ids[:, 0], ids[:, -1] = CLS, SEP
seg = torch.zeros_like(ids)
x, y = mlm_corrupt(ids)
mlm_head = nn.Linear(D, V)                                    # BERT ties this to enc.tok
logits = mlm_head(enc(x, seg))                                # (8, 32, V)
loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
print((y != -100).sum().item(), round(loss.item(), 2))        # 31 11.02
print(round(torch.log(torch.tensor(float(V))).item(), 2))     # 10.33 = chance level ln V

head = nn.Linear(D, 2)                                        # new, randomly initialised
cls_vec = enc(ids, seg)[:, 0]                                 # final [CLS] vectors
print(head(cls_vec).shape)                                    # torch.Size([8, 2])
opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=2e-5, weight_decay=0.01)
total, warm = 282, 28
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / warm if t < warm else (total - t) / (total - warm))
```

The untrained loss (11.02) sits near $\ln V = 10.33$, the sanity check that the loss is wired correctly, and 31 of 240 ordinary positions became targets, close to 15%.

With real weights, Hugging Face's `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)` loads the encoder with a fresh head, its tokenizer builds the `[CLS] ... [SEP]` ids and masks, and `DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)` does the 80/10/10 corruption (not run here: it downloads weights).

## Checking that it worked

During pretraining:

- The MLM loss should start near $\ln V$ and fall steadily; a loss near 0 almost at once means the answer leaks into the input.
- Ask it to fill blanks. Given "the soup was tasty and the pizza was `[MASK]`", put its top probabilities on excellent, friendly, good and great (about 0.10 each); after "the soup was bland and the pizza was `[MASK]`" its top guesses were dirty, greasy, awful and horrible. Its loss fell from 4.21 (near $\ln 79 = 4.37$, 79 words) to 0.94.

During fine-tuning:

- Beat a TF-IDF + logistic regression baseline (06.03); if you cannot, something is off.
- Run 3 to 5 seeds and report mean and spread. On a few thousand examples, runs differing only in seed (head init, data order) can give very different dev scores, and some fail entirely. Follow-up work found two fixes: train a little longer with warmup, and use Adam *with* bias correction (the original BERT optimiser omitted it). The faint dots in the transfer figure show the same spread at 8 labels: 54% to 100%.
- Check the predicted class distribution on the dev set.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| MLM loss near 0 within a few hundred steps | labels copied after corruption, or the target is visible to the model | copy labels first; check that masked positions really show `[MASK]` or a substitute
| MLM loss stuck near $\ln V$ | every label is $-100$, or special tokens are the only targets | count targets per batch (about 15% of real tokens)
| Fine-tuned model predicts one class | learning rate too large; forgetting | use 2e-5 to 5e-5 with warmup; check the dev label distribution
| Dev score jumps by several points between seeds | small dataset, unstable fine-tuning | several seeds, longer training, bias-corrected AdamW, report mean and spread
| Good dev score, poor in production | different tokeniser or casing, or inputs truncated at 512 | load the tokeniser saved with the model; chunk long texts
| Mean pooling worse than expected | `[PAD]` vectors included in the average | multiply by the attention mask before averaging and divide by the true length

## Encoders versus decoders

Bidirectional attention suits understanding tasks: classification, tagging, answer spans, retrieval embeddings, reranking. An encoder has no natural way to generate left to right, since it never learned to predict from the left context alone. Decoder-only models, trained on next tokens under a causal mask, generate naturally and at scale solve many classification tasks by prompting. For a high-volume task with fixed labels, a fine-tuned 110M-parameter encoder is still often the cheapest accurate option.

## Going deeper

**Where BERT-base's 110M parameters live.** The embedding block (token $30522 \times 768$, position $512 \times 768$, segment, layer norm) is 23.8M. Each layer has four $768 \times 768$ attention projections (2.36M with biases) and a $768 \to 3072 \to 768$ feed-forward block (4.72M): 7.09M per layer, 85.1M for 12. A $768 \times 768$ pooler adds 0.59M, for 109.5M in total.

**Sample efficiency and ELECTRA.** MLM learns from about 77 of 512 positions, a causal LM from 511. That is one reason BERT needed long training (published setup, from memory: about 1M steps at batch 256, on BooksCorpus plus English Wikipedia, roughly 3.3B words). ELECTRA replaces MLM with **replaced-token detection**: a small generator fills the masked slots with plausible tokens, and the main model classifies *every* position as original or replaced. Every position gives signal, the discriminator never sees `[MASK]`, and ELECTRA reached BERT-level results with considerably less compute.

**Why 15%?** Too little masking gives few targets; too much leaves too little context. Later work reported that larger models can do well with higher rates (around 40%), so 15% is a default, not a law. Masking whole words or spans (SpanBERT) makes the task usefully harder: predicting a word's last WordPiece when its first piece is visible is too easy.

**MLM is a denoising autoencoder.** It reconstructs clean input from a corrupted copy, predicting masked tokens independently of each other given the visible ones. So it is not a generative model of whole sentences, one more reason encoders do not generate fluent text.

## Key terms

| Term | Meaning
| --- | --- |
| Pretraining | Training on unlabelled text with a self-supervised objective before any task
| Fine-tuning | Continuing to train all pretrained weights, plus a small new head, on a labelled task
| Masked language modelling (MLM) | Corrupt 15% of tokens (80% `[MASK]`, 10% random, 10% unchanged) and predict the originals there
| Contextual embedding | A token's output vector, which depends on the whole sentence, unlike a static word vector
| `[CLS]` / `[SEP]` | Special tokens: the first position (used for classification) and the segment separator
| Next-sentence prediction (NSP) | BERT's binary task "does B follow A?"; RoBERTa showed it can be dropped
| Dynamic masking | Drawing a fresh random mask each time a sequence is used
| Warmup | Ramping the learning rate up from 0 over the first steps so the new head does not wreck pretrained weights
| Catastrophic forgetting | Losing pretrained knowledge because fine-tuning updates are too large

<div class="take">
<div class="lab">Key takeaway</div>

masked language modelling turns unlabelled text into a bidirectional encoder whose contextual vectors transfer to almost any understanding task. Fine-tune all of it gently, with a small head, a learning rate around 2e-5 to 5e-5 and warmup, and treat seed-to-seed variance on small datasets as part of the result.

</div>

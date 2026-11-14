---
id: "06.04"
course: "deep-learning"
module: "06"
slug: "word-embeddings"
title: "Word embeddings"
keyIdeas: ["A **word embedding** is a dense vector (typically 100 to 300 numbers) learned so that words used in similar contexts get similar vectors: the distributional hypothesis made trainable. An embedding layer is just a $V \\times D$ table whose row lookup has a sparse gradient.", "Skip-gram with negative sampling trains a logistic classifier to tell real (centre, context) pairs from $k$ random noise pairs, replacing a $V$-way softmax with $k + 1$ binary decisions per pair.", "Noise words are drawn from frequency to the 3/4 power so rare words are sampled more often than their raw share, and very frequent words are subsampled so 'the' stops dominating the updates.", "GloVe fits dot products to log co-occurrence counts by weighted least squares; fastText sums character n-gram vectors, so misspellings and unseen words still get vectors, though look-alike words with different meanings get close vectors too.", "Static embeddings give one vector per word type, so 'bank' (river) and 'bank' (money) share one blended vector, and they absorb social biases from the corpus. Check nearest neighbours and downstream accuracy, not just analogies."]
question: "You need word vectors for a domain of clinical notes full of rare drug names, abbreviations and misspellings, and a team member suggests downloading general-purpose word2vec vectors trained on news. Compare that option with training word2vec and fastText on your own notes, include one way fastText could mislead you, and say how you would check the result before using it in a model."
answer: "General news vectors are the weakest option: most drug names and clinical abbreviations are missing from their vocabulary, and the words that are present carry their news meaning (for example 'discharge' as an electrical or legal term rather than leaving hospital). Training on in-domain notes fixes the meaning problem. word2vec trained on the notes still learns one vector per vocabulary entry, so misspellings and rare drug names either fall below the minimum count and get no vector, or get noisy vectors from a handful of contexts; any unseen variant at test time has nothing at all. fastText builds each word from its character n-grams (3 to 6 characters with boundary markers), so 'amoxicilin' (misspelt) shares most of its n-grams with 'amoxicillin' and lands near it, inflections share pieces, and every new string gets a usable vector; rare words also borrow statistical strength from frequent words with common pieces. The risk is that surface similarity is not meaning: drug names that differ by a few letters but are clinically different, such as look-alike or sound-alike medications ('ampicillin' versus 'amoxicillin', or classic confusion pairs), will get similar vectors, and very short abbreviations have few n-grams and may collide with unrelated ones in the hashed buckets. Before using the vectors, list nearest neighbours by cosine for a set of critical terms and their known confusion pairs, have a clinician review them, compare against a word2vec baseline, and measure accuracy on the real downstream task (for example entity recognition) with and without the vectors. If ambiguity matters (one abbreviation with several expansions), consider a contextual model instead, since a static vector blends all senses."
sources: [{"title": "Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)", "url": "https://arxiv.org/abs/1301.3781", "note": "Introduces skip-gram and CBOW."}, {"title": "Distributed Representations of Words and Phrases (Mikolov et al., 2013)", "url": "https://arxiv.org/abs/1310.4546", "note": "Negative sampling, the 3/4-power noise distribution and frequent-word subsampling."}, {"title": "GloVe: Global Vectors for Word Representation (Pennington et al., 2014)", "url": "https://aclanthology.org/D14-1162/", "note": "The weighted least-squares objective and its motivation from co-occurrence ratios."}, {"title": "Enriching Word Vectors with Subword Information (Bojanowski et al., 2017)", "url": "https://arxiv.org/abs/1607.04606", "note": "fastText's character n-gram model and results on rare words."}]
practice: ["word2vec", "glove", "fasttext", "embedding-similarity", "cosine-similarity-matrix"]
tags: ["word embeddings", "word2vec"]
tools: ["NumPy"]
---

## What you will learn

- Why sparse count vectors cannot tell that "car" and "automobile" are related, and what a dense **word embedding** does instead.
- How word2vec's skip-gram with negative sampling turns "which words appear near which" into a small binary classification problem, with one update worked by hand.
- The tricks that make word2vec train well: 3/4-power noise sampling and frequent-word subsampling.
- How GloVe fits a global count matrix, and how fastText builds vectors for unseen words.
- How to train a tiny embedding model in PyTorch, check it, and spot the classic failure modes.

*Before you start:* tokenisation (06.01), bag-of-words vectors (06.03), the sigmoid and logistic loss, softmax, and a basic PyTorch training loop.

## From counts to meaning

Lesson 06.03 turned each document into a long, mostly-zero vector of word counts. That treats every word as its own isolated axis: "car" and "automobile" are as unrelated as "car" and "banana", because their one-hot vectors are perpendicular. A model that learns something about "automobile" learns nothing about "car".

Word embeddings give every word a short, dense vector (typically 100 to 300 numbers) and *learn* those numbers so that words used in similar ways end up close together. The guiding principle is the **distributional hypothesis**: words that occur in similar contexts tend to have similar meanings. Nobody tells the model that cats and dogs are animals; it notices that both are fed and both sleep on the grass.

## The idea in one picture

Take a toy corpus of 6,000 short sentences such as "the cat sleeps on the grass", "we buy rice at the market" and "he drives the bus to work". Count how often each word appears within two positions of a few context words:

| Word | sleeps | feeds | eat | buy | drives | fuel
| --- | --- | --- | --- | --- | --- | --- |
| cat | 67 | 72 | 0 | 0 | 0 | 0
| dog | 53 | 62 | 0 | 0 | 0 | 0
| bread | 0 | 0 | 60 | 49 | 0 | 0
| rice | 0 | 0 | 46 | 49 | 0 | 0
| car | 0 | 0 | 0 | 0 | 61 | 50

The rows for "cat" and "dog" point in almost the same direction (cosine similarity 0.999), while "cat" and "bread" share no contexts (cosine 0). The counts already carry meaning, but a real count matrix has one column per vocabulary word and is mostly zeros. An embedding method compresses it into a few dimensions.

We trained 16-dimensional word2vec vectors on this corpus (step 5 shows a slightly smaller version of the code) and projected them to two dimensions:

~~~dg-fig 06.04-map.svg
Learned word vectors projected to 2-D. Animals, foods and vehicles form three separate clusters without ever being labelled. The ambiguous word "bank", used in both river and money sentences, sits close to both context groups.
~~~

Three new terms:

- A **word embedding** is the learned vector for one word. All of them together form the **embedding matrix** $E$ of shape $V \times D$, where $V$ is the vocabulary size and $D$ the number of dimensions.
- **Cosine similarity** measures the angle between two vectors: $\cos(a, b) = a^\top b / (\|a\|\,\|b\|)$. It is 1 for the same direction, 0 for perpendicular, -1 for opposite, and ignores vector length.
- A **static** embedding gives each word type exactly one vector, whatever sentence it appears in.

~~~dg-fig 06.04-sim.svg
Cosine similarity between nine words. With one-hot vectors (left) only a word matches itself. With learned vectors (right) each group forms a bright block: cat and dog score 0.79, bread and rice 0.88, while cat and bread score 0.05.
~~~

## Step 1: an embedding is a lookup table

Mechanically an embedding layer is just the matrix $E$, and "embedding word number $i$" means taking row $i$. That is mathematically the same as multiplying a one-hot vector $e_i$ (all zeros except a 1 at position $i$) by the matrix:

$$e_i^\top E = E[i,:].$$

Selecting a row is far cheaper than multiplying, but the view shows that an embedding layer is a linear layer with one-hot inputs, trained by ordinary backpropagation.

The gradient is **sparse**: only rows used in the batch get a gradient, and a word that occurs twice in a batch gets both contributions added. That is why tables with millions of rows are cheap to train (PyTorch offers `nn.Embedding(..., sparse=True)` for huge vocabularies).

## Step 2: skip-gram, the prediction task

Skip-gram (Mikolov et al., 2013) slides a window over the text. Every position gives a **centre word** $c$, and each word within, say, 2 to 5 positions on either side is a **context word** $o$. In "the cat sleeps on the grass" with window 2, the centre "sleeps" gives the pairs (sleeps, the), (sleeps, cat), (sleeps, on), (sleeps, the). Our toy corpus of 34,197 tokens yields about 100,000 pairs.

The task is to predict the context from the centre. Each word gets *two* vectors: $v_w$ when it is the centre and $u_w$ when it is a context. The score of a pair is the dot product $u_o^\top v_c$, and the original formulation turns scores into probabilities with a softmax over the whole vocabulary:

$$P(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V} \exp(u_w^\top v_c)}.$$

Here $u_o$ is the true neighbour's context vector, $v_c$ the centre vector, and the denominator sums over the whole vocabulary. With $V = 10^6$ words, every training pair would touch a million vectors.

**CBOW** (continuous bag of words) is the mirror image: average the context vectors and predict the centre word. It is faster; skip-gram is better for rare words.

## Step 3: negative sampling, the binary shortcut

Negative sampling asks a cheaper question: "is this pair real, or made up?" For each real pair $(c, o)$, draw $k$ random **noise words** $n_1, \dots, n_k$ and train a logistic classifier to say yes to the real pair and no to the fake ones. The loss for one pair is

$$\ell = -\log \sigma(u_o^\top v_c) - \sum_{j=1}^{k} \log \sigma(-u_{n_j}^\top v_c),$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid. The first term rewards a high score for the true neighbour, the second low scores for noise words. One pair now costs $k + 1$ dot products instead of $V$.

~~~dg-flow One skip-gram negative-sampling update for the centre word "fox".
Centre "fox" -(lookup)-> v(fox) -(dot with u(quick), u(k noise words))-> *k+1 sigmoid scores -(1 real, k fake)-> Logistic loss
~~~

The gradients are those of logistic regression: for a score $s$ with label $t$ (1 real, 0 noise), $\partial \ell / \partial s = \sigma(s) - t$, "prediction minus target". The centre vector moves towards the real context vector and away from the noise vectors.

<div class="ex">
<div class="lab">Worked example</div>

**one negative-sampling update by hand**

Use 2-D vectors. Centre $v_c = (0.5, 1.0)$, true context $u_o = (1.0, 0.5)$, one noise word $u_n = (-0.5, 0.5)$, so $k = 1$, learning rate 0.5.

Scores: real $s_o = 0.5 \cdot 1.0 + 1.0 \cdot 0.5 = 1.0$; noise $s_n = 0.5 \cdot (-0.5) + 1.0 \cdot 0.5 = 0.25$.

Probabilities: $\sigma(1.0) = 0.731$ ("real" with 73%), $\sigma(-0.25) = 0.438$ ("fake" with only 44%). Loss: $-\ln 0.731 - \ln 0.438 = 0.313 + 0.826 = 1.139$.

Score gradients: real $\sigma(1.0) - 1 = -0.269$; noise $\sigma(0.25) - 0 = 0.562$.

Centre gradient: $-0.269\,u_o + 0.562\,u_n = (-0.269 - 0.281,\; -0.134 + 0.281) = (-0.550, 0.147)$.

Update: $v_c \leftarrow v_c - 0.5 \times (-0.550, 0.147) = (0.775, 0.927)$.

Check: the real score rises to $0.775 + 0.463 = 1.238$, the noise score falls to $-0.388 + 0.463 = 0.076$, and the loss drops from 1.139 to 0.986. (The context vectors also move, in real training.)

</div>

## Step 4: choosing the negatives and thinning frequent words

Which words should be the noise? Raw frequency would make "the" the negative most of the time; uniform sampling would waste updates on words that never matter. word2vec uses the **unigram distribution raised to the 3/4 power**: $P_{\text{noise}}(w) \propto f(w)^{0.75}$, where $f(w)$ is the word's frequency.

~~~dg-fig 06.04-noise.svg
How often each word of the toy corpus is drawn as a negative. The 0.75 power pulls the most frequent word down and lifts the rare tail, between raw frequency and uniform. Here "the" drops from 0.207 to 0.110 and the rarest word rises from 0.0030 to 0.0045.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what the 3/4 power does**

"the" has frequency 0.05; a rare word has frequency $10^{-6}$. Raw ratio: $0.05 / 10^{-6} = 50{,}000$ to 1.

After the 3/4 power: $50{,}000^{0.75} \approx 3{,}344$ to 1.

So, relative to "the", the rare word becomes $50{,}000 / 3{,}344 \approx 15$ times more likely to be drawn as a negative, and its vector gets pushed away from unrelated words more often.

</div>

The second trick is **subsampling frequent words**: the paper drops each occurrence of word $w$ before training with probability

$$P_{\text{drop}}(w) = 1 - \sqrt{t / f(w)}, \qquad t \approx 10^{-5}.$$

For "the" with $f = 0.05$, $\sqrt{10^{-5}/0.05} = 0.014$, so 98.6% of its occurrences are skipped; words with $f(w) \le t$ are always kept. Training gets faster, and because dropped words no longer fill window slots, the window reaches further to informative neighbours.

The last knobs are the window and $k$. Small windows (2 to 3) favour words that can replace each other; large windows (5 to 10) favour words about the same topic. The paper recommends $k$ = 5 to 20 for small corpora and 2 to 5 for large ones.

## Step 5: training it in PyTorch

The whole pipeline on a toy corpus: sentences, pairs, 3/4-power negatives, two embedding tables trained with the step 3 loss, then nearest neighbours.

```python
import random, numpy as np, torch
import torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0); rng = random.Random(0)

# 1. A toy corpus: 3 word groups used in the same sentence frames, plus a few word-specific facts
groups = {'cat dog horse cow': ['the {} sleeps on the grass', 'the farmer feeds the {}'],
          'bread rice apple cheese': ['i eat fresh {} for lunch', 'we buy {} at the market'],
          'car bus train bike': ['he drives the {} to work', 'the {} needs fuel']}
facts = ['the dog barks at night', 'the cow gives milk', 'the cheese is made from milk',
         'the train runs on rails', 'the bike has two wheels', 'an apple grows on a tree']
sents = []
for _ in range(6000):
    if rng.random() < 0.25: sents.append(rng.choice(facts).split()); continue
    words, frames = rng.choice(list(groups.items()))
    sents.append(rng.choice(frames).format(rng.choice(words.split())).split())

# 2. Vocabulary, (centre, context) pairs with window 2, noise distribution count^0.75
vocab = sorted({w for s in sents for w in s}); idx = {w: i for i, w in enumerate(vocab)}
pairs = np.array([(idx[s[i]], idx[s[j]]) for s in sents for i in range(len(s))
                  for j in range(max(0, i - 2), min(len(s), i + 3)) if j != i])
counts = np.bincount([idx[w] for s in sents for w in s], minlength=len(vocab))
noise = counts ** 0.75 / (counts ** 0.75).sum()
print(len(vocab), len(pairs))                       # 49 93776

# 3. Two embedding tables and the negative-sampling loss
V, D, k = len(vocab), 16, 5
v_in, u_out = torch.nn.Embedding(V, D), torch.nn.Embedding(V, D)
torch.nn.init.uniform_(v_in.weight, -0.5 / D, 0.5 / D); torch.nn.init.zeros_(u_out.weight)
opt = torch.optim.Adam([*v_in.parameters(), *u_out.parameters()], lr=0.01)
for epoch in range(5):
    perm, total = np.random.permutation(len(pairs)), 0.0
    for b in range(0, len(pairs), 512):
        c, o = map(torch.tensor, pairs[perm[b:b + 512]].T)
        neg = torch.tensor(np.random.choice(V, size=(len(c), k), p=noise))
        vc = v_in(c)                                             # (B, D)
        pos = (u_out(o) * vc).sum(-1)                            # (B,)   real-pair scores
        ngs = torch.einsum('bkd,bd->bk', u_out(neg), vc)         # (B, k) noise scores
        loss = -(F.logsigmoid(pos) + F.logsigmoid(-ngs).sum(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(c)
    print(epoch, round(total / len(pairs), 3))
# 0 2.501   1 1.498   2 1.409   3 1.399   4 1.398

# 4. Nearest neighbours by cosine similarity
E = F.normalize(v_in.weight.detach(), dim=1)
def neighbours(w, n=3):
    sims = E @ E[idx[w]]
    best = sims.argsort(descending=True)[1:n + 1]                # [0] is the word itself
    return [(vocab[i], round(sims[i].item(), 2)) for i in best]
for w in ['cat', 'bread', 'train']:
    print(w, neighbours(w))
# cat [('horse', 1.0), ('dog', 0.86), ('farmer', 0.81)]
# bread [('rice', 1.0), ('cheese', 0.91), ('apple', 0.87)]
# train [('bus', 0.88), ('car', 0.88), ('rails', 0.81)]
```

"cat" and "horse" reach 1.0 because they appear in exactly the same sentence frames, while "dog" also barks at night. `F.logsigmoid` stays finite for very negative scores, unlike `log(sigmoid(...))`. The loss never reaches zero, since some noise words are genuine neighbours. For real corpora use a tested library (gensim's `Word2Vec`, the `fasttext` package), which adds subsampling, learning-rate decay and multithreading.

## Reading and checking the result

A flat loss curve looks the same whether the vectors are useful or not, so watch what the vectors do.

~~~dg-fig 06.04-train.svg
Left: the loss per pair falls fast and flattens after three epochs. Right: average cosine similarity within a group (cat and dog) jumps to about 0.8 after one epoch, while similarity across groups (cat and bus) settles near 0.25.
~~~

Checks, from cheapest to most convincing:

- **Nearest neighbours** of 10 to 20 words you know, including rare ones. Neighbours like "the" and "of" usually mean no subsampling or too little training.
- **Word-similarity benchmarks** (WordSim-353, SimLex-999) correlate model cosines with human ratings.
- **Analogies**: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$ holds only if you exclude the three input words from the nearest-neighbour search; otherwise "king" itself usually wins. Treat it as a sanity check, not a quality measure.
- **Downstream accuracy** is the real test: plug the vectors into your classifier or tagger and compare with random initialisation.

## GloVe: fitting the counts directly

word2vec streams through text and never builds the count table. GloVe (Pennington et al., 2014) does the opposite: it builds the global co-occurrence matrix $X$, where $X_{ij}$ counts how often word $j$ appears in the context of word $i$, then fits vectors whose dot products match the log counts:

$$J = \sum_{i,j:\,X_{ij} > 0} f(X_{ij})\,\big(w_i^\top \tilde w_j + b_i + \tilde b_j - \log X_{ij}\big)^2.$$

Here $w_i$ is word $i$'s vector, $\tilde w_j$ a separate context vector, $b_i$ and $\tilde b_j$ scalar biases that absorb how common each word is, and $f$ a weight per pair. Only pairs that co-occur are summed, so zero counts (log undefined) are skipped. It is weighted least squares: a regression on the count matrix.

~~~dg-fig 06.04-glove.svg
The GloVe weight for a pair grows as count to the power 0.75 and is capped at 1 from 100 co-occurrences. A pair seen once gets weight 0.032; seen 10 times, 0.178.
~~~

The weight is $f(x) = \min\big(1, (x/x_{\max})^{0.75}\big)$ with $x_{\max} = 100$: noisy rare pairs count little, and very frequent pairs such as ("of", "the") cannot dominate. The final vector is usually $w + \tilde w$. In practice skip-gram and GloVe give similar quality; GloVe needs the count matrix, skip-gram streams over text.

## fastText: vectors from word pieces

Both methods treat each word as an atom: "running", "runner" and the typo "runnning" share nothing, and an unseen word has no vector. fastText (Bojanowski et al., 2017) represents a word as the sum of vectors for its **character n-grams**. With boundary markers, "where" gives `<wh`, `whe`, `her`, `ere`, `re>` for $n = 3$, plus the 4- to 6-grams and the whole word `<where>`. N-grams are hashed into about 2 million buckets. Training is otherwise skip-gram with negative sampling.

Any string, including a misspelling or an unseen inflection, now gets a vector, which matters for morphologically rich languages and noisy text.

~~~dg-fig 06.04-fasttext.svg
Share of character n-grams in common with a target word. A typo and a spelling variant overlap strongly, which is why fastText handles them. But look-alikes with different meanings (ampicillin, cunning) overlap just as much, and a true synonym (jogging) barely at all.
~~~

| Method | Trains on | Unit | Unseen words
| --- | --- | --- | --- |
| word2vec (SGNS) | streaming windows | word | no vector
| GloVe | global co-occurrence counts | word | no vector
| fastText | streaming windows | character n-grams | built from n-grams

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Stop words are everyone's neighbours | no subsampling, or raw dot products | subsample; compare with cosine
| Rare words have random neighbours | too few occurrences | raise `min_count`, use fastText, get more text
| "bank" is close to "river" and "loan" | one vector blends all senses | contextual embeddings (06.06)
| Pretrained vectors miss domain terms | trained on news or web text | train on in-domain text; fastText for unseen words
| Outputs show gender or ethnic stereotypes | associations absorbed from the corpus | audit; debias; avoid for decisions about people

Two need more words. **One vector per word type**: in our toy corpus "bank" has cosine 0.63 with the river words and 0.61 with the money words, while those groups have only 0.42 with each other; its vector is a compromise representing neither sense well. **Social bias**: Bolukbasi et al. (2016) found analogies like "man is to computer programmer as woman is to homemaker" in vectors trained on news, and such associations flow into every model built on top.

## Going deeper

**Skip-gram factorises a matrix.** Levy and Goldberg (2014) showed that, with enough capacity, the optimum of skip-gram with negative sampling satisfies $u_o^\top v_c = \text{PMI}(c, o) - \log k$, where the **pointwise mutual information** $\text{PMI}(c, o) = \log \frac{P(c, o)}{P(c)\,P(o)}$ measures how much more often a pair occurs than chance. So word2vec and GloVe are close relatives: both find low-rank vectors whose dot products approximate a transformed co-occurrence table, and a truncated SVD of a positive-PMI matrix is a competitive baseline.

**Why two vectors per word?** With one table, a word's score with itself, $\|v_w\|^2$, is always high, yet words rarely sit next to themselves. Separate tables remove that constraint.

**Negative sampling is not a softmax.** It is a simplified noise-contrastive objective, so its scores are not calibrated probabilities: fine for vectors, wrong for language modelling. The paper's alternative, hierarchical softmax, puts the vocabulary in a binary Huffman tree and costs about $\log_2 V$ decisions per pair.

**Static to contextual.** ELMo and then BERT (06.06) compute a fresh vector for every token from its sentence, which resolves "bank". Their input layer is still an embedding table, now over subword tokens (06.01).

## Key terms

| Term | Meaning
| --- | --- |
| Word embedding | A dense learned vector for one word; similar usage gives similar vectors
| Skip-gram / CBOW | Predict context words from the centre word / the centre word from its context
| Negative sampling | Replace the full softmax with a real-vs-noise logistic task using $k$ noise words
| Subsampling | Randomly dropping occurrences of very frequent words
| GloVe | Weighted least squares fit of dot products to log co-occurrence counts
| fastText | Word vector as a sum of character n-gram vectors, so unseen words get vectors
| Cosine similarity | Dot product of length-normalised vectors; the standard similarity measure
| Static embedding | One vector per word type, regardless of the sentence it appears in

<div class="take">
<div class="lab">Key takeaway</div>

word embeddings compress co-occurrence statistics into short dense vectors in which cosine similarity tracks usage. Skip-gram with negative sampling learns them as a cheap real-vs-noise classifier, GloVe fits log counts directly, and fastText builds them from character n-grams to cover unseen words; all three give one static vector per word, which is the limitation contextual models remove.

</div>

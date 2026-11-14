---
id: "02.01"
course: "ai-engineering-course"
module: "02"
slug: "from-data-to-vectors"
title: "From data to vectors"
keyIdeas: ["An embedding is a fixed-length list of numbers (e.g. 384, 768, 1536 dimensions) that represents meaning.", "Early methods: one-hot and TF-IDF vectors are sparse and know nothing about meaning. Word2vec learned dense word vectors by predicting neighbouring words.", "Modern embedding models are transformer encoders. Each token gets a contextual vector; pooling (mean or [CLS]) turns them into one vector.", "They are trained with contrastive learning: pull matching pairs (question, answer) together, push unrelated pairs apart.", "Images, audio and code can be embedded the same way. CLIP puts images and text into one shared space.", "Vectors from different models live in different spaces. Index and query with the same model and version, or re-embed everything."]
question: "Explain how a sentence becomes a single embedding vector in a modern embedding model, and why the same model must be used at index and query time."
answer: "The text is tokenized, a transformer encoder produces a contextual vector per token, and a pooling step (usually mean pooling or the [CLS] token) collapses them into one fixed-length vector, often L2-normalized. The model was trained contrastively so related texts are close. Each model defines its own vector space, so query vectors are only comparable with document vectors from the same model and version."
sources: [{"title": "Jay Alammar: The Illustrated Word2vec", "url": "https://jalammar.github.io/illustrated-word2vec/", "note": "How training on context turns words into vectors."}, {"title": "Sentence Transformers documentation", "url": "https://www.sbert.net/", "note": "The standard library for sentence and document embeddings."}, {"title": "Reimers & Gurevych: Sentence-BERT (2019)", "url": "https://arxiv.org/abs/1908.10084", "note": "Siamese BERT networks that made sentence embeddings practical."}]
---

## Why vectors

Computers compare numbers easily and meaning hardly at all. An embedding turns a piece of data (a word, a sentence, an image) into a fixed-length vector so that similar things end up close together. Once data is vectors, "find related documents" becomes "find nearby points".

## From sparse to dense

- **One-hot**: one dimension per vocabulary word. Huge, sparse, and every word is equally far from every other.
- **TF-IDF / BM25**: weights words by how distinctive they are. Good for keyword match, blind to synonyms ("car" vs "automobile").
- **Word2vec (2013)**: Jay Alammar's illustrated guide shows the trick. Train a small network to predict a word's neighbours; the learned weights become dense word vectors where king - man + woman lands near queen.
- **Contextual embeddings**: transformers give each token a vector that depends on the sentence, so "bank" in "river bank" and "bank account" differ.

## How a sentence embedding is built

~~~dg-flow A sentence embedding model in one line.
Text -(tokenize)-> Token ids -(encoder)-> Token vectors -(mean pool)-> *Sentence vector -(normalize)-> Unit vector
~~~

Sentence-BERT (Reimers and Gurevych, 2019) showed that plain BERT vectors are poor for similarity, and fixed it by fine-tuning BERT in a siamese setup so that pooled vectors of similar sentences have high cosine similarity. Modern embedding models keep this recipe and train contrastively on huge sets of pairs: a query and its relevant passage are pulled together while other passages in the batch are pushed apart.

## Choosing dimensions and models

| Model size | Typical dimensions | Notes |
| --- | --- | --- |
| Small (MiniLM class) | 384 | Fast, runs on CPU, fine for many apps |
| Base | 768 | Good default |
| Large / API models | 1024-3072 | Better quality, more storage and slower search |

Some newer models support Matryoshka training, where the first k dimensions are already a usable embedding, so you can truncate vectors to save space.

## Rules that save you later

- Use the **same model and version** for documents and queries. Vectors from two models are in unrelated spaces.
- Store which model produced each vector; a model upgrade means re-embedding the corpus.
- Embed chunks of a sensible size (a few hundred tokens). A whole book in one vector averages its meaning away.

## See meaning become arithmetic

Once text is a vector, relationships between meanings become directions in space. The classic demonstration from word2vec (Mikolov et al., 2013) is that the vector for "king" minus "man" plus "woman" lands closest to "queen". The figure below uses small hand-made vectors to show the idea: pick a word, subtract one concept, add another, and see which word is nearest to the result.

~~~dg-raw
%%widget analogy Vector arithmetic on a toy word space.
~~~

Modern sentence embedding models don't guarantee clean analogies like this, but the principle carries over: similar meanings end up near each other, and consistent differences (formal versus casual, question versus answer, English versus French) tend to line up as directions.

## What an embedding call looks like

```
POST /v1/embeddings
{"model": "text-embedding-3-small", "input": ["How do I reset my password?"]}

-> {"data": [{"embedding": [0.0123, -0.0442, 0.0871, ... 1536 numbers]}]}
```

That one call gives you a fixed-length vector no matter how long the input was (up to the model's limit, often 512 to 8,192 tokens). You store it next to the original text and an id.

## Where people go wrong

- **Mixing models.** Vectors from two different models live in different spaces. A query embedded with model B against documents embedded with model A gives garbage similarities, not an error. Store the model name and version with every vector.
- **Embedding too much at once.** A 20-page document squashed into one vector averages away the detail. Chunk first (lesson 03.05), then embed each chunk.
- **Forgetting the query side.** Some models (E5, BGE, Nomic and others) expect a prefix such as "query: " or "search_query: " on questions and a different one on documents. Skipping it quietly costs recall.

<div class="ex">
<div class="lab">Worked example</div>

**cost of embedding a knowledge base**

200,000 support articles, averaging 600 tokens each, is 120 million tokens.

At an illustrative \$0.02 per million tokens, the one-off embedding run costs about \$2.40. Storage at 1,536 float32 dimensions is 200,000 × 1,536 × 4 bytes ≈ 1.2 GB.

Embedding is cheap; the expensive parts come later, in storage, indexing and re-embedding when you change models.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

an embedding model is an encoder plus pooling, trained so that related inputs land near each other. Pick one model and keep it consistent for indexing and querying.

</div>

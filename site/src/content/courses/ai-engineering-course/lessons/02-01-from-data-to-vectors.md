---
id: "02.01"
course: "ai-engineering-course"
module: "02"
slug: "from-data-to-vectors"
title: "From data to vectors"
keyIdeas: ["An embedding is a fixed-length list of numbers (often 384 to 4,096 dimensions) placed so that similar inputs land close together.", "Modern embedding models are transformer encoders: each token gets a contextual vector, pooling collapses them into one, and contrastive training pulls matching pairs together.", "Current models (2025–26) differ in default dimensions, input length, languages and modalities; most support Matryoshka-style truncation through a dimensions parameter.", "A shorter embedding from the same model is cheaper to store and search for a small quality loss; truncated vectors must be re-normalized if you cut them yourself.", "Vectors from different models or versions live in unrelated spaces. Index and query with the same model, store which one produced each vector, and re-embed on upgrade."]
question: "Explain how a sentence becomes a single embedding vector, why the same model must be used at index and query time, and what changes if you ask for 512 dimensions instead of 3,072."
answer: "The text is tokenized, a transformer encoder produces a contextual vector for each token, and a pooling step (mean pooling, the first token, or the last token in decoder-based models) collapses them into one fixed-length vector, usually L2-normalized. Contrastive training put related texts close together in that model's own space, so query vectors are only comparable with document vectors from the same model and version; mixing models gives meaningless similarities without any error. Asking for 512 instead of 3,072 dimensions works on models trained Matryoshka-style, where the leading dimensions already carry most of the meaning. Each vector shrinks six-fold, so storage, memory and distance computations fall by about the same factor, while retrieval quality drops slightly. If you truncate vectors yourself instead of using the API parameter, re-normalize them, and never mix 512- and 3,072-dimension vectors in one index."
rubric: ["Describes tokenize, encode per token, pool to one vector (and normalization).", "Explains that each model defines its own space, so query and documents must use the same model and version.", "Explains Matryoshka-style truncation: leading dimensions carry most information, so fewer dimensions still work.", "States the trade-off: about six times less storage and compute for a small quality loss, and the need to re-normalize truncated vectors."]
sources: [{"title": "Reimers & Gurevych: Sentence-BERT (2019)", "url": "https://arxiv.org/abs/1908.10084", "note": "Siamese BERT networks that made sentence embeddings practical."}, {"title": "Kusupati et al.: Matryoshka Representation Learning (2022)", "url": "https://arxiv.org/abs/2205.13147", "note": "Training so that the first k dimensions are a usable embedding on their own."}, {"title": "OpenAI: Vector embeddings guide", "url": "https://platform.openai.com/docs/guides/embeddings", "note": "Default sizes of text-embedding-3 models and the dimensions parameter."}, {"title": "Jay Alammar: The Illustrated Word2vec", "url": "https://jalammar.github.io/illustrated-word2vec/", "note": "How training on context turns words into vectors."}]
practice: ["sentence-embeddings", "info-nce-loss", "l2-vector-norm", "embedding-similarity"]
tools: ["OpenAI embeddings", "Gemini embeddings", "Voyage", "Cohere Embed", "Qwen3-Embedding", "BGE-M3"]
---

## Why vectors

Computers compare numbers easily and meaning hardly at all. An embedding turns a piece of data (a word, a sentence, an image) into a fixed-length vector so that similar things end up close together. Once data is vectors, "find related documents" becomes "find nearby points", which the rest of this module makes fast.

## From sparse to dense

- **One-hot**: one dimension per vocabulary word. Huge, sparse, and every word is equally far from every other.
- **TF-IDF / BM25**: weights words by how distinctive they are. Good for keyword match, blind to synonyms ("car" vs "automobile"). It is still a strong baseline ([Lexical search and BM25](lesson:aieng:11.01)).
- **Word2vec (2013)**: train a small network to predict a word's neighbours; the learned weights become dense word vectors. Relationships between meanings become directions: "king" minus "man" plus "woman" lands near "queen".
- **Contextual embeddings**: transformers give each token a vector that depends on the sentence, so "bank" in "river bank" and "bank account" differ.

The figure uses small hand-made vectors to show the arithmetic: pick a word, subtract one concept, add another, and see which word is nearest.

~~~dg-raw
%%widget analogy Vector arithmetic on a toy word space.
~~~

Modern sentence models don't guarantee clean analogies like this, but the principle carries over: similar meanings sit near each other, and consistent differences (formal versus casual, question versus answer, English versus French) tend to line up as directions.

## How a sentence embedding is built

~~~dg-flow A sentence embedding model in one line.
Text -(tokenize)-> Token ids -(encoder)-> Token vectors -(pool)-> *Sentence vector -(normalize)-> Unit vector
~~~

Sentence-BERT (Reimers and Gurevych, 2019) showed that raw BERT vectors are poor for similarity, and fixed it by fine-tuning BERT in a siamese setup so that pooled vectors of similar sentences have high cosine similarity. Current models keep the recipe and scale it up: they train contrastively on very large sets of pairs, pulling a query and its relevant passage together while pushing the other passages in the batch apart. Many recent models are built from decoder LLMs (Qwen3-Embedding is one) and pool the last token's vector instead of averaging.

## Embedding models in 2025–26

The table lists widely used models as their documentation stood in late 2026. Model names, sizes and prices change quickly, so treat it as a snapshot and check the provider's page before you commit.

| Model | Access | Default dims | Can shorten to | Max input | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI text-embedding-3-small / -large | API | 1,536 / 3,072 | any size via `dimensions` | about 8k tokens | Cheap, common default |
| Google gemini-embedding-2 | API | 3,072 | 128 to 3,072 (768, 1,536 recommended) | 8,192 tokens | Text, images, audio and documents in one space |
| Voyage voyage-4 family | API | 1,024 | 256, 512, 2,048 | 32,000 tokens | All 4-series models share one space |
| Cohere embed-v5.0-pro / -fast | API | 2,048 | 256 to 1,536 | 128k tokens | Text and images, including PDFs |
| Qwen3-Embedding 0.6B / 4B / 8B | Open weights | 1,024 / 2,560 / 4,096 | 32 up to the default | 32k tokens | Multilingual, takes task instructions |
| BGE-M3 | Open weights | 1,024 | no | 8,192 tokens | Dense, sparse and multi-vector outputs from one model |
| EmbeddingGemma (300M) | Open weights | 768 | 512, 256, 128 | 2,048 tokens | Small enough to run on device |

The "can shorten to" column comes from **Matryoshka representation learning** (Kusupati et al., 2022): the model is trained so that the first k dimensions are already a good embedding on their own, like nested dolls. You request a shorter vector with a parameter (`dimensions` for OpenAI, `output_dimensionality` for Gemini, `output_dimension` for Voyage), or truncate a full vector yourself. Truncating by hand breaks unit length, so re-normalize afterwards; Google's docs note that gemini-embedding-001 needs this, while gemini-embedding-2 normalizes for you. Picking between models on your own data is covered in [Choosing and fine-tuning an embedding model](lesson:aieng:02.10).

## What an embedding call looks like

```json
POST /v1/embeddings
{"model": "text-embedding-3-large", "input": ["How do I reset my password?"], "dimensions": 1024}

-> {"data": [{"embedding": [0.0123, -0.0442, 0.0871, "... 1,024 numbers"]}]}
```

One call returns one fixed-length vector per input, however long the input was (up to the model's limit). You store it next to the original text, an id, and the model name and dimension count.

<div class="ex">
<div class="lab">Worked example</div>

**Cost and size of embedding a knowledge base**

200,000 support articles averaging 600 tokens each is 200,000 × 600 = 120 million tokens. At an illustrative \$0.02 per million tokens, the one-off embedding run costs \$2.40.

Storage at 1,536 float32 dimensions: 200,000 × 1,536 × 4 bytes = 1.23 GB. Requesting 512 dimensions instead: 200,000 × 512 × 4 = 0.41 GB, three times smaller, and every distance computation does a third of the work.

Embedding is cheap. The expensive parts come later: storage, indexing, and re-embedding all 120 million tokens again when you change models.

</div>

## Where people go wrong

- **Mixing models.** Vectors from two models, or two versions of one model, live in different spaces. Querying model-A documents with a model-B vector gives garbage similarities, not an error. The same goes for mixing dimension counts. Store the model name, version and dimension with every vector.
- **Embedding too much at once.** A 20-page document squashed into one vector averages away the detail, even when the model accepts 32k tokens. Chunk first ([Chunking and document parsing](lesson:aieng:11.04)), then embed each chunk.
- **Forgetting the query side.** Many models (E5, BGE, Nomic, Qwen3-Embedding and others) expect a prefix or instruction on queries, such as "query: " or a task description, and a different one or none on documents. API models often take an input type parameter instead. Skipping it quietly costs recall.
- **Truncating without re-normalizing.** Shortened vectors that are no longer unit length give wrong cosine and dot-product rankings.

<div class="take">
<div class="lab">Key takeaway</div>

An embedding model is an encoder plus pooling, trained so that related inputs land near each other. Pick one model, choose a dimension with the `dimensions` parameter rather than by habit, and keep model, version and size identical for indexing and querying.

</div>

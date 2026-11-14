---
id: "02.04"
course: "ai-engineering-course"
module: "02"
slug: "vector-compression"
title: "Vector compression"
keyIdeas: ["Vectors are big: a 768-dim float32 vector is 3 KB. At 100M vectors that is 300 GB, too much for RAM.", "Compression trades a little recall for much less memory and faster distance maths.", "Dimensionality reduction (PCA, Matryoshka truncation) keeps fewer dimensions.", "Product quantization (PQ) splits a vector into m subvectors and replaces each with the id of its nearest centroid. With 256 centroids, each id is one byte.", "Distances are computed from precomputed lookup tables (asymmetric distance), so search on codes is fast.", "Typical pattern: search compressed vectors to get candidates, then rescore the top few hundred with full-precision vectors."]
question: "A 768-dim float32 vector is compressed with PQ into 96 subvectors with 256 centroids each. What is the new size and compression ratio?"
answer: "Each of the 96 subvectors is stored as one byte (256 centroids = 8 bits), so the vector becomes 96 bytes. The original is 768 x 4 = 3,072 bytes, so the compression ratio is 32x (plus a small shared codebook)."
sources: [{"title": "Pinecone: Product Quantization", "url": "https://www.pinecone.io/learn/series/faiss/product-quantization/", "note": "Subvectors, codebooks and the memory math."}, {"title": "Jegou et al.: Product Quantization for Nearest Neighbor Search (2011)", "url": "https://ieeexplore.ieee.org/document/5432202", "note": "The original PQ paper."}, {"title": "Hugging Face: Embedding Quantization", "url": "https://huggingface.co/blog/embedding-quantization", "note": "Binary and int8 embeddings with rescoring."}]
---

## Why compress

Memory is the main cost of vector search. A 768-dimension float32 vector is 768 x 4 = 3,072 bytes. A billion of them is 3 TB. Even 100 million is 300 GB, more RAM than most single servers have. Compression shrinks each vector so the index fits in memory, and as a bonus, smaller vectors mean faster distance calculations because less data moves through the CPU cache.

## Ways to make vectors smaller

- **Fewer dimensions**: PCA, or Matryoshka-trained models where you can keep the first 256 of 1024 dimensions.
- **Scalar quantization**: store each number in fewer bits (int8 or even 1 bit). Covered in the next lesson.
- **Product quantization (PQ)**: replace chunks of the vector with codebook ids.

## How product quantization works

Pinecone's PQ guide walks through it:

1. Split each D-dim vector into m equal subvectors (768 dims into 96 pieces of 8).
2. For each subvector position, run k-means on the training data to learn k centroids (usually 256).
3. Encode each subvector as the id of its nearest centroid. With k = 256 the id fits in one byte.
4. The vector is now m bytes instead of D x 4 bytes.

~~~dg-flow Product quantization of one vector.
768 floats (3 KB) -(split)-> 96 subvectors of 8 -(nearest centroid)-> *96 one-byte codes -(store)-> 96 bytes
~~~

At query time the query is not compressed. For each subvector position the system precomputes the distance from the query piece to all 256 centroids, a small table. The distance to any stored vector is then the sum of m table lookups. This is called asymmetric distance computation, and it's very fast.

## The cost: recall

Each vector is now approximated by a mix of centroids, so distances are approximate and some true neighbours get ranked lower. Common fixes:

- **Rescoring**: fetch the top 100-1,000 candidates with PQ, then recompute exact distances with the original vectors (stored on disk or SSD).
- **More subvectors** (larger m): better accuracy, less compression.
- **Combine with an index**: Faiss's IVF-PQ clusters vectors first and stores PQ codes per cluster.

## Product quantization, interactively

The figure compresses one vector with product quantization (PQ). Change the number of sub-vectors and bits per code and watch the size and the reconstruction error.

~~~dg-raw
%%widget pq Splitting a vector into sub-vectors and replacing each with a codebook id.
~~~

Step by step, for a 768-dim vector with m = 96 sub-vectors and 8-bit codes:

1. Split the vector into 96 pieces of 8 dimensions each.
2. For each of the 96 positions, run k-means on a training sample to learn 256 centroids (8 bits gives 2^8 = 256). That is the codebook.
3. Replace each piece with the id of its nearest centroid: one byte.
4. The vector is now 96 bytes instead of 768 × 4 = 3,072 bytes, a 32× reduction.

## Why search still works on codes

At query time you don't decompress every vector. For each of the 96 positions you compute the distance from the query piece to all 256 centroids once, giving a 96 × 256 lookup table. The distance to any stored vector is then 96 table lookups and additions, with no multiplications. This is called asymmetric distance computation, because the query stays exact and only the database is compressed. It is why PQ indexes can scan millions of vectors per second on a CPU.

## Choosing parameters

| Setting | Higher means | Cost |
| --- | --- | --- |
| m (sub-vectors) | Finer pieces, better accuracy | More bytes per vector |
| Bits per code | More centroids per piece | Bigger lookup tables, slower training |
| Training sample | Better codebooks | Longer build |

A common starting point is 8 bits per code and m chosen so that each piece has 4 to 16 dimensions. Measure recall@10 against exact search on a few hundred real queries; if it drops below what your product needs, raise m or rescore the top candidates with full vectors.

<div class="take">
<div class="lab">Key takeaway</div>

compression trades a small amount of recall for large memory savings. PQ turns a vector into a handful of bytes, and rescoring the top candidates with full vectors wins most of the recall back.

</div>

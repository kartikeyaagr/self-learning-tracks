---
id: "02.04"
course: "ai-engineering-course"
module: "02"
slug: "vector-compression"
title: "Vector compression"
keyIdeas: ["Vectors are big: a 768-dim float32 vector is 3 KB, so 100M of them is 307 GB, more RAM than most servers have.", "Product quantization (PQ) splits a vector into m subvectors and replaces each with the id of its nearest centroid from a learned codebook; with 256 centroids each id is one byte.", "Search runs on the codes through lookup tables (asymmetric distance computation): the query stays exact and each distance is m table lookups.", "OPQ learns a rotation before splitting so information is spread evenly across subvectors, which lowers PQ error at the same size.", "IVF-PQ combines clustering with PQ codes of residuals; rescoring the top candidates with full vectors wins back most of the lost recall."]
question: "A 768-dim float32 vector is compressed with PQ into 96 subvectors with 256 centroids each. What is the new size and compression ratio, how is a distance computed without decompressing, and what would you add to recover recall?"
answer: "Each of the 96 subvectors is stored as the id of one of 256 centroids, which fits in 8 bits, so the vector becomes 96 bytes. The original is 768 × 4 = 3,072 bytes, a 32× reduction, plus a small shared codebook of 96 × 256 centroids of 8 floats (about 786 KB) for the whole collection. At query time the query is not compressed. For each of the 96 positions you compute the distance from the query's piece to all 256 centroids once, giving a 96 × 256 table; the distance to any stored vector is then the sum of 96 table lookups. To recover recall, fetch a few hundred candidates by PQ distance and rescore them with the original float vectors kept on disk, apply an OPQ rotation before quantizing, or use more subvectors at the cost of more bytes."
sources: [{"title": "Pinecone: Product Quantization", "url": "https://www.pinecone.io/learn/series/faiss/product-quantization/", "note": "Subvectors, codebooks and the memory math."}, {"title": "Jegou et al.: Product Quantization for Nearest Neighbor Search (2011)", "url": "https://ieeexplore.ieee.org/document/5432202", "note": "The original PQ paper, including IVFADC (IVF with PQ of residuals)."}, {"title": "Ge et al.: Optimized Product Quantization (CVPR 2013)", "url": "https://openaccess.thecvf.com/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html", "note": "Learning a rotation that lowers PQ error."}, {"title": "Faiss wiki: The index factory", "url": "https://github.com/facebookresearch/faiss/wiki/The-index-factory", "note": "How OPQ, IVF and PQ compose into one index string."}]
practice: ["ann-recall-at-k", "kmeans-step", "k-means-clustering", "embedding-similarity"]
tools: ["Faiss"]
---

## Why compress

Memory is the main cost of vector search. A 768-dimension float32 vector is 768 × 4 = 3,072 bytes. A billion of them is 3 TB; even 100 million is 307 GB. Compression shrinks each vector so the index fits in memory, and smaller vectors also mean faster distance calculations, because less data moves through the CPU cache.

There are three ways to make vectors smaller:

- **Fewer dimensions**: PCA, or Matryoshka-trained models where you keep the first 256 of 1,024 dimensions ([From data to vectors](lesson:aieng:02.01)).
- **Fewer bits per number**: int8 or 1-bit scalar quantization ([Compression and quantization](lesson:aieng:02.05)).
- **Product quantization (PQ)**: replace chunks of the vector with codebook ids. This lesson.

## How product quantization works

For a $D$-dimensional vector, PQ picks $m$ subvectors and $k$ centroids per subvector:

1. Split each vector into $m$ equal pieces: 768 dimensions into 96 pieces of 8.
2. For each of the 96 positions, run k-means on a training sample to learn $k = 256$ centroids. That set is the codebook.
3. Replace each piece with the id of its nearest centroid. With 256 centroids the id fits in one byte.
4. The vector is now $m \log_2 k / 8 = 96$ bytes instead of $4D = 3{,}072$, a 32× reduction.

~~~dg-flow Product quantization of one vector.
768 floats (3 KB) -(split)-> 96 subvectors of 8 -(nearest centroid)-> *96 one-byte codes -(store)-> 96 bytes
~~~

Change the number of subvectors and bits per code in the figure and watch the size and the reconstruction error move in opposite directions.

~~~dg-raw
%%widget pq Splitting a vector into sub-vectors and replacing each with a codebook id.
~~~

## Why search still works on codes

You never decompress the database. At query time the query stays exact. For each of the 96 positions you compute the distance from the query's piece to all 256 centroids once, giving a 96 × 256 lookup table. The distance to any stored vector is then 96 table lookups and additions, with no multiplications:

$$
d(q, x)^2 \approx \sum_{j=1}^{m} T_j\big[c_j(x)\big]
$$

where $c_j(x)$ is the stored code for piece $j$ and $T_j$ holds the squared distances from the query's piece $j$ to every centroid. This is asymmetric distance computation (ADC): only the database side is approximated. It is why PQ indexes scan millions of codes per second on a CPU.

## OPQ: rotate first

PQ assumes each piece carries a fair share of the information. Real embeddings often don't: some dimensions vary a lot, others hardly at all, and correlated dimensions may land in the same piece. Pieces with high variance get coarse codes and dominate the error.

Optimized product quantization (OPQ, Ge et al., 2013) learns an orthogonal rotation $R$ and applies it before splitting, choosing $R$ so that variance is spread evenly across the pieces and the quantization error falls. The rotation is one $D \times D$ matrix shared by the whole collection, applied once to each vector at insert and once to each query, so its cost is small. In Faiss it is one more stage in the index string, for example `OPQ96,IVF65536,PQ96`. On real embeddings it usually buys measurable recall at identical code size, so it is worth trying whenever you use PQ.

## IVF-PQ: clusters plus codes

PQ alone still scans every code. IVF-PQ first clusters the collection with k-means into `nlist` lists and only scans the few lists nearest the query ([Indexing techniques](lesson:aieng:02.07)). It also encodes the **residual**, the difference between each vector and its cluster centroid, instead of the raw vector. Residuals are smaller and more uniform than raw vectors, so the same 96 bytes describe them more accurately. This combination, from the original PQ paper (Jégou et al.), is how billion-scale indexes fit in RAM.

<div class="ex">
<div class="lab">Worked example</div>

**IVF-PQ memory for 100 million 768-dim vectors**

Raw float32: 100M × 3,072 bytes = 307 GB.

IVF-PQ with OPQ, `nlist` = 65,536 and m = 96 at 8 bits:

- Codes: 96 bytes per vector, plus an 8-byte id stored in the inverted list = 104 bytes. 100M × 104 = 10.4 GB.
- Coarse centroids: 65,536 × 768 × 4 bytes = 201 MB.
- PQ codebooks: 96 × 256 × 8 × 4 bytes = 0.79 MB.
- OPQ rotation: 768 × 768 × 4 bytes = 2.4 MB.

Total about 10.6 GB, roughly 29× less than raw: it fits one server's RAM. Keep the 307 GB of float32 vectors on SSD and rescore the top 200 candidates of each query; that reads 200 × 3 KB = 614 KB per query from disk.

</div>

## Choosing parameters

| Setting | Higher means | Cost |
| --- | --- | --- |
| m (subvectors) | Finer pieces, better accuracy | More bytes per vector |
| Bits per code | More centroids per piece | Bigger lookup tables, slower training |
| Training sample | Better codebooks | Longer build |

A common starting point is 8 bits per code, with m chosen so each piece has 4 to 16 dimensions, and $D$ must divide evenly by m. Measure recall@10 against exact search on a few hundred real queries. If it is below what your product needs, raise m, add OPQ, or rescore: fetch the top 100 to 1,000 by PQ distance and recompute exact distances with the original vectors. Rescoring usually wins back most of the loss for little latency.

## Where this breaks

- **Unrepresentative training data.** Codebooks learned on one slice of data (English only, one product line) quantize everything else badly. Train on a random sample of the full collection, typically tens to hundreds of thousands of vectors.
- **Drift.** When new content looks different from what the codebooks were trained on, recall falls without any code change. Track recall on a fixed query set and retrain codebooks when it drops.
- **Absolute distances are approximate.** A PQ distance is fine for ranking but poor for thresholds like "only return results closer than 0.3". Apply thresholds after rescoring.
- **Too few bytes.** Very aggressive settings (m = 8 for a 768-dimension vector, 8 bytes each) save memory but lose so much recall that no amount of rescoring of a short candidate list recovers it.

<div class="take">
<div class="lab">Key takeaway</div>

Product quantization turns a 3 KB vector into about 100 bytes and still ranks through table lookups. Add an OPQ rotation, encode IVF residuals, and rescore the top candidates with full vectors to keep recall close to exact search at a thirtieth of the memory.

</div>

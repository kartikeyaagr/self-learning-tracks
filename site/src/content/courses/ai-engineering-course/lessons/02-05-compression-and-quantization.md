---
id: "02.05"
course: "ai-engineering-course"
module: "02"
slug: "compression-and-quantization"
title: "Compression and quantization"
keyIdeas: ["Scalar quantization (int8) maps each float32 value onto 256 levels using a calibrated range: 4× smaller, very small recall loss.", "Binary quantization keeps only the sign of each value: 32× smaller, distance becomes Hamming distance (XOR plus popcount), but recall drops unless you rescore.", "The standard recipe is to search the small vectors, oversample (4× to 10× the final k), then rescore the candidates with int8 or float32 vectors.", "Matryoshka truncation and binary quantization stack: fewer dimensions times fewer bits, for 64× or more savings in the first-stage index.", "Quantizing stored vectors is a different job from quantizing model weights, but the trade-off is the same: bits for accuracy, measured on your own queries."]
question: "Compare int8 scalar quantization and binary quantization for 1,024-dim vectors on size, speed and accuracy, and explain how you would get binary-level memory with close to float32 recall."
answer: "int8 stores 1,024 dimensions in 1,024 bytes, 4× smaller than float32's 4,096. Distances use fast integer dot products, and recall usually stays very close to float32 because rounding each value to one of 256 levels is small next to the gaps between neighbours; the range must be calibrated on representative data. Binary keeps one bit per dimension, 128 bytes or 32× smaller, and compares vectors with XOR and popcount, the fastest option, but one bit per dimension loses much more information, so recall on its own drops noticeably. To get binary memory with near-float32 recall, keep only the binary vectors in RAM, fetch 4× to 10× more candidates than you need by Hamming distance, and rescore those candidates with int8 or float32 vectors kept on disk. Optionally truncate a Matryoshka model to fewer dimensions before binarizing to shrink the first stage further, and check recall@k on your own queries."
rubric: ["Gives correct sizes: 1,024 bytes for int8 and 128 bytes for binary, against 4,096 for float32.", "Notes integer dot products for int8 and Hamming distance via XOR and popcount for binary.", "States int8 loses very little recall while binary loses more.", "Describes oversampling with binary then rescoring with higher-precision vectors.", "Mentions measuring recall on your own data, or Matryoshka truncation as an extra step."]
sources: [{"title": "Hugging Face: Embedding Quantization", "url": "https://huggingface.co/blog/embedding-quantization", "note": "int8 and binary embeddings, speedups and retained accuracy."}, {"title": "Qdrant: Binary Quantization", "url": "https://qdrant.tech/articles/binary-quantization/", "note": "1-bit vectors with oversampling and rescoring."}, {"title": "Weaviate: Vector quantization", "url": "https://docs.weaviate.io/weaviate/concepts/vector-quantization", "note": "PQ, BQ and SQ in one vector DB."}, {"title": "Kusupati et al.: Matryoshka Representation Learning (2022)", "url": "https://arxiv.org/abs/2205.13147", "note": "Why truncated embeddings keep most of their quality."}]
practice: ["int8-symmetric-quantize", "affine-uint8-quantize", "ann-recall-at-k", "l2-vector-norm"]
tools: ["Qdrant", "Weaviate", "pgvector"]
---

## Fewer bits per number

Product quantization ([Vector compression](lesson:aieng:02.04)) compresses chunks of a vector with learned codebooks. Scalar and binary quantization compress each number on its own. They are simpler, need little or no training, and are built into most vector databases and pgvector.

| Method | Bytes for 1,024 dims | Distance | Typical accuracy |
| --- | --- | --- | --- |
| float32 | 4,096 | Dot product | Baseline |
| int8 (scalar) | 1,024 | Integer dot product | Very close to baseline |
| Binary | 128 | Hamming | Lower; good after rescoring |

## Scalar quantization: float32 to int8

For each dimension (or for the whole vector set), find the range of values in a representative sample, using a percentile to ignore outliers. Map that range linearly onto the 256 values of a byte. Each number now takes 1 byte instead of 4, integer SIMD maths is faster than float maths, and recall usually falls by well under a percent for good embedding models.

Suppose a dimension ranges from −0.21 to 0.19 in your sample, a width of 0.40. A value of 0.05 becomes

$$
\operatorname{round}\!\left(\frac{0.05 - (-0.21)}{0.40} \times 255\right) - 128 = \operatorname{round}(165.75) - 128 = 38
$$

Values outside the sampled range are clipped to −128 or 127, which is why you calibrate on representative data rather than a handful of vectors.

## Binary quantization: 1 bit per dimension

Keep only whether each value is positive or negative. A 1,024-dimension vector becomes 128 bytes, 32× smaller than float32. Distance becomes Hamming distance: XOR the two bit strings and count the ones. CPUs count bits with one popcount instruction per 64 bits, so comparing two 1,024-dimension binary vectors takes 16 of them.

The cost is accuracy. One bit per dimension keeps the direction of each axis and nothing else. Qdrant notes that binary works best on high-dimensional embeddings (around 1,000 dimensions and up), ideally from models trained with quantization in mind; on small or low-dimensional models the loss can be too large even with rescoring.

## Oversample, then rescore

~~~dg-flow The standard recipe for binary vectors.
Query -(binary search)-> *Top 4k candidates -(rescore with int8 / float32)-> Top k results
~~~

Fetch 4× to 10× more candidates than you need using cheap binary distances, then re-rank only those with higher-precision vectors, which can live on SSD because you read just a few hundred per query. The Hugging Face embedding quantization post reports binary retrieval keeping around 92 to 96% of float32 quality this way, with large speed and memory gains. Qdrant exposes the same idea as an oversampling factor plus a rescore flag; in pgvector you write it as an inner query on `binary_quantize(embedding)` and an outer `ORDER BY` on the full vector.

The figure runs a real experiment in your browser: 3,000 clustered 64-dimension vectors, 40 queries, and recall@10 measured against exact float32 search. Switch between int8 and binary, change the oversampling factor and toggle rescoring.

~~~dg-raw
%%widget binq Recall of int8 and binary quantization, with and without oversample-and-rescore.
~~~

Three things to see: int8 keeps recall close to 100% at a quarter of the memory; binary alone loses a lot; binary plus rescoring wins most of it back.

## Matryoshka plus binary: the combined recipe

Two independent savings multiply. A Matryoshka-trained model lets you keep the first $d' < d$ dimensions ([From data to vectors](lesson:aieng:02.01)); binary quantization then stores each of those in one bit. The first-stage index costs $d'/8$ bytes per vector instead of $4d$.

1. Embed with a Matryoshka model at full size, for example 1,024 dimensions.
2. Build the in-memory index on the first 512 dimensions, binarized: 64 bytes per vector, 64× smaller than 4,096.
3. Keep full 1,024-dimension int8 vectors on SSD: 1,024 bytes each.
4. Query: truncate and binarize the query the same way, take the top 10 × k by Hamming distance, rescore with int8 (or float32 for the very top), return k.

Each cut compounds the other's error, so the first stage gets worse faster than either alone. That is fine as long as the true neighbours survive into the oversampled candidate list. Raise the oversampling factor until recall@k plateaus, and stop truncating where it no longer does.

<div class="ex">
<div class="lab">Worked example</div>

**The memory plan for 100 M vectors at 1,024 dims**

float32 everywhere: 100M × 4,096 bytes = 410 GB of RAM.

Binary index on all 1,024 dims: 100M × 128 bytes = 12.8 GB of RAM, plus int8 vectors for rescoring on SSD: 100M × 1,024 bytes = 102 GB. Fetch the top 100 by Hamming distance, rescore with int8, return the top 10. RAM falls 32×.

Binary index on the first 512 Matryoshka dims: 100M × 64 bytes = 6.4 GB of RAM, a 64× reduction, with the same 102 GB of int8 on SSD. Each query reads 100 × 1 KB = 100 KB from disk for rescoring.

Recall@10 for plans like this typically lands in the mid to high 90s after rescoring. Measure it on your own queries before you rely on that figure.

</div>

## Picking a level

- Under a few million vectors: float32 is often fine; memory is cheap at that scale.
- Tens of millions: int8 is the easy default.
- Hundreds of millions or more, or cost-sensitive: binary (optionally on truncated dimensions) or PQ, with rescoring.
- Always measure recall@k on your own queries before and after.

## Where this breaks

- **Bad calibration.** An int8 range taken from a few vectors clips real values and hurts recall more than the format itself.
- **Models not built for it.** Low-dimensional models, or models never trained with quantization or Matryoshka objectives, can lose too much for rescoring to fix. Check the model card.
- **Rescoring without oversampling.** Rescoring only the top k fixes their order but cannot bring back neighbours the first stage missed.
- **Confusing the two kinds of quantization.** Compressing stored vectors changes search recall; quantizing model weights to FP8 or 4-bit changes the model's own outputs, with its own quality risks ([What your provider optimises](lesson:aieng:06.05), [BF16, FP8, INT8 and 4-bit inference](lesson:ieng:11.02)).

<div class="take">
<div class="lab">Key takeaway</div>

int8 gives 4× savings almost for free. Binary gives 32×, and Matryoshka truncation on top gives 64× or more, but only with oversampling and rescoring from higher-precision vectors on disk.

</div>

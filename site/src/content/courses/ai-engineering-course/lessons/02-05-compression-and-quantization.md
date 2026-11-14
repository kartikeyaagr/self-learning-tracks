---
id: "02.05"
course: "ai-engineering-course"
module: "02"
slug: "compression-and-quantization"
title: "Compression and quantization"
keyIdeas: ["Scalar quantization (SQ): map each float32 value to int8 using the value range per dimension. 4x smaller, small accuracy loss.", "Binary quantization (BQ): keep only the sign of each value, 1 bit per dimension. 32x smaller, distance becomes Hamming distance (XOR + popcount), very fast.", "Binary loses more accuracy; works best with high-dimensional models and is usually paired with oversampling and rescoring.", "Hugging Face reports binary retrieval keeping around 92-96% of float32 quality after rescoring, with large speedups.", "Pattern: search the small vectors, oversample (e.g. 4x top-k), rescore with int8 or float32 vectors.", "Quantize vectors (storage) is different from quantizing model weights (module 7), but the trade-off is the same: bits for accuracy."]
question: "Compare int8 scalar quantization and binary quantization on size, speed and accuracy for 1024-dim vectors."
answer: "int8 SQ stores 1024 dims in 1,024 bytes (4x smaller than float32), uses fast integer dot products, and loses very little accuracy. Binary stores 1 bit per dim, 128 bytes (32x smaller), uses Hamming distance via XOR and popcount (fastest), but loses more accuracy, so you oversample candidates and rescore them with higher-precision vectors."
sources: [{"title": "Hugging Face: Embedding Quantization", "url": "https://huggingface.co/blog/embedding-quantization", "note": "int8 and binary embeddings, speedups and retained accuracy."}, {"title": "Qdrant: Binary Quantization", "url": "https://qdrant.tech/articles/binary-quantization/", "note": "1-bit vectors with oversampling and rescoring."}, {"title": "Weaviate: Vector quantization", "url": "https://docs.weaviate.io/weaviate/concepts/vector-quantization", "note": "PQ, BQ and SQ in one vector DB."}]
---

## Fewer bits per number

Product quantization (previous lesson) compresses chunks of a vector. Scalar and binary quantization compress each number on its own. They are simpler, need little or no training, and are built into most vector databases.

## Scalar quantization: float32 to int8

For each dimension, find the range of values in your data (min and max, or a percentile to ignore outliers). Map that range linearly onto 256 integer buckets. Each number now takes 1 byte instead of 4.

- **Size**: 4x smaller.
- **Speed**: integer SIMD maths is faster than float maths.
- **Accuracy**: usually a small loss, often below one percent of recall for good embedding models.

## Binary quantization: 1 bit per dimension

Keep only whether each value is positive or negative. A 1024-dim vector becomes 128 bytes.

- **Size**: 32x smaller than float32.
- **Speed**: distance becomes Hamming distance: XOR the two bit strings and count the ones. CPUs do this in a couple of instructions.
- **Accuracy**: larger loss. Qdrant notes that it works best with high-dimensional embeddings (1,000+ dims), where enough information survives.

| Method | Bytes for 1024 dims | Distance | Typical accuracy |
| --- | --- | --- | --- |
| float32 | 4,096 | Dot product | Baseline |
| int8 (SQ) | 1,024 | Integer dot product | Very close to baseline |
| Binary (BQ) | 128 | Hamming | Lower; good after rescoring |

## Oversample, then rescore

~~~dg-flow The standard recipe for binary vectors.
Query -(binary search)-> *Top 4k candidates -(rescore with int8 / float32)-> Top k results
~~~

The Hugging Face embedding quantization post shows this: binary search to get many candidates fast, then rescore them with higher-precision vectors kept on disk. They report retrieval quality around 92-96% of float32 with big speed and memory gains. Qdrant exposes the same as an "oversampling" factor plus a rescore flag.

## Picking a level

- Under a few million vectors: float32 is often fine; memory is cheap at that scale.
- Tens of millions: int8 is the easy default.
- Hundreds of millions or more, or cost-sensitive: binary or PQ with rescoring.
- Always measure recall@k on your own queries before and after.

## Measure it live

The figure below runs a real experiment in your browser: 3,000 clustered 64-dim vectors, 40 queries, and recall@10 measured against exact float32 search. Switch between int8 and binary, change the oversampling factor and toggle rescoring.

~~~dg-raw
%%widget binq Recall of int8 and binary quantization, with and without oversample-and-rescore.
~~~

What to look for:

- **int8** keeps recall close to 100% at 4× less memory. The error from rounding each number to one of 256 levels is small compared with the gaps between neighbors.
- **Binary** alone loses a lot of recall at 32× less memory. One bit per dimension keeps only the sign.
- **Binary plus rescoring** wins most of it back: fetch 4× or 10× more candidates using cheap binary distances, then re-rank only those with full-precision vectors.

## How int8 scaling works

For each dimension (or for the whole vector set), find the range of values seen in a sample, for example -0.21 to 0.19. Map that range linearly to -128 to 127. A value of 0.05 becomes round((0.05 - (-0.21)) / 0.40 × 255) - 128 = 38. Values outside the sampled range get clipped, which is why you calibrate on representative data, not on a handful of vectors.

## Binary in practice

Binary distance is Hamming distance: XOR the two bit strings and count the ones. CPUs do this with a single popcount instruction per 64 bits, so a 1,024-dim binary vector is compared in 16 instructions. Binary works best on high-dimensional embeddings (1,024 dims and up) from models trained with quantization in mind; on small or low-dimensional models the recall loss can be too large even with rescoring.

<div class="ex">
<div class="lab">Worked example</div>

**the memory plan for 100 M vectors at 1,024 dims**

float32 everywhere: 410 GB in RAM.

Binary index in RAM (12.8 GB) plus int8 vectors for rescoring on fast SSD (102 GB): fetch the top 100 by Hamming distance, rescore with int8, return the top 10.

RAM need drops by about 30×, and recall@10 typically stays in the mid to high 90s. Check on your own data before relying on that figure.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

int8 gives 4x savings almost for free; binary gives 32x but needs oversampling and rescoring to keep quality.

</div>

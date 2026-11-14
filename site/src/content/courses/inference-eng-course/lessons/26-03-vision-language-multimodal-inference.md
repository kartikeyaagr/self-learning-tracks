---
id: "26.03"
course: "inference-eng-course"
module: "26"
slug: "vision-language-multimodal-inference"
title: "Vision-Language & Multimodal Inference"
keyIdeas: ["VLMs: a vision encoder turns images into embeddings, projected into the LLM's token space.", "Images become many tokens (hundreds to thousands each), inflating prefill.", "Encoder compute runs on GPU too; schedule it alongside LLM steps.", "Cache encoder outputs and image-token KV by image hash.", "Preprocessing (decode, resize) on CPU can bottleneck; parallelize it."]
question: "Why do images often dominate TTFT in vision-language models, and what can you cache?"
answer: "Each image is encoded by a vision encoder into hundreds or thousands of tokens, and both encoding and the prefill over those image tokens take time, often more than the text prompt. You can cache the encoder output per image (by hash) and, through the prefix cache keyed with image hashes, the KV for image tokens, so repeated images skip both steps."
sources: [{"title": "LLaVA: Visual Instruction Tuning", "url": "https://arxiv.org/abs/2304.08485", "note": "A common VLM architecture."}, {"title": "vLLM: Multimodal inputs", "url": "https://docs.vllm.ai/en/latest/features/multimodal_inputs.html", "note": "Serving multimodal models."}]
tags: ["multimodal"]
---

## Pipeline

~~~dg-flow A vision-language request.
Image -(CPU decode, resize)-> Vision encoder (GPU) -> Projector -> *Image tokens plus text tokens -> LLM prefill -> Decode
~~~

1. The image is decoded and resized on the CPU.
2. A vision encoder, typically a ViT, turns it into patch embeddings.
3. A projector maps those embeddings into the language model's token space.
4. The image tokens join the text tokens, and the language model prefills them like any other prompt.

## Cost

| Item | Typical |
| --- | --- |
| Tokens per image | 256 to several thousand, depending on resolution |
| Encoder time | milliseconds to tens of ms per image |
| Extra prefill | the same as that many text tokens |

<div class="ex">
<div class="lab">Worked example</div>

**tiles add up**

A model splits a 1344 × 1344 image into 9 tiles of 448 × 448 plus one thumbnail. Each tile becomes 256 tokens.

That is 10 × 256 = 2,560 image tokens, before any text.

A request with 4 such images carries over 10,000 prompt tokens. Its TTFT is closer to a long-document RAG request than to a chat message.

</div>

## Optimizations

- **Encoder cache:** hash each image and reuse its embeddings when the same image appears again.
- **Prefix cache with image hashes:** include image hashes in the prefix key so KV for repeated images and prompts is reused.
- **Resolution control:** fewer tiles mean fewer tokens. Many tasks do not need maximum resolution.
- **Separate encoder workers:** at scale, run encoders on their own GPUs (encoder disaggregation).
- **CPU preprocessing in thread pools,** so image decoding never stalls the engine loop.

## Deep dive: image tokens grow with resolution

A ViT encoder with patch size ⟦m1108⟧ turns an ⟦m1109⟧ image into ⟦m1110⟧ tokens. Tile-based encoders split a large image into fixed-size tiles plus a thumbnail:

⟦M1107⟧

~~~dg-raw
%%chart Image tokens against image size, for 336-pixel tiles with 14-pixel patches (576 tokens per tile, plus a thumbnail tile once there is more than one tile).
x = 336, 672, 1008, 1344
xlab = Image side (pixels, square)
ylab = Tokens
Tokens: 576, 2880, 5760, 9792
%%
~~~

A 1344-pixel screenshot costs as much prefill as a 9,800-token prompt, and holds as much KV. Downscaling images that do not need fine detail is the cheapest optimization in many multimodal services.

<div class="take">
<div class="lab">Key takeaway</div>

in vision-language models, images are expensive prompts. Control resolution, cache embeddings and KV by image hash, and keep preprocessing off the critical path.

</div>

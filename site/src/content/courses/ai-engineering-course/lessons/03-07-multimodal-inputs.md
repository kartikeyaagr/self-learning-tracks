---
id: "03.07"
course: "ai-engineering-course"
module: "03"
slug: "multimodal-inputs"
title: "Multimodal inputs"
keyIdeas: ["Images, audio and video become input tokens: an image is cut into patches (28 or 32 px squares, depending on the provider), audio and video are charged per second. They use context and are billed at the input price.", "Each provider caps and resizes images differently, so the same screenshot costs anywhere from under 200 to over 2,600 tokens. Resize deliberately.", "Images in an agent's history are resent on every step; keeping only the last few cuts image tokens by several times.", "For documents, an OCR or parsing pipeline is cheap, deterministic and citable; a vision model reads layout, charts and handwriting. Many systems use both.", "ColPali-style retrieval embeds page images as many small vectors and matches them with late interaction, at the cost of tens of times more storage per page."]
question: "You are building an assistant that answers questions over 50,000 scanned PDF pages of equipment manuals, full of diagrams and tables. Compare an OCR-and-text-RAG pipeline with a vision-based approach, and estimate what the vision approach costs per question."
answer: "The OCR pipeline parses each page to text and tables, chunks, embeds and retrieves text; it is cheap per question, deterministic and easy to cite, but it loses diagrams and garbles complex tables. The vision approach retrieves whole pages (with a ColPali-style multi-vector index or text retrieval over captions) and sends the page images to a vision-language model, which reads diagrams and layout directly. Cost: a page image at about 1,500 to 2,700 tokens on a high-resolution tier, 5 retrieved pages is 7,500 to 13,500 input tokens per question, a few cents at 4 dollars per million tokens, versus perhaps 3,000 tokens of text chunks. Storage for a ColPali index is about 257 KB per page, so 50,000 pages is about 13 GB. I would build a hybrid: OCR text for retrieval and citations, page images sent to the model only for the top pages, and an eval on questions that need diagrams versus text to check the vision path earns its cost."
rubric: ["Contrasts OCR/parsing and VLM reading on cost, determinism, layout, charts and citations", "Estimates image tokens per page and per question with a provider formula", "Mentions page-image retrieval (ColPali or equivalent) and its storage cost", "Proposes a hybrid and an eval split by question type"]
sources: [{"title": "Anthropic: Vision", "url": "https://platform.claude.com/docs/en/build-with-claude/vision", "note": "28-pixel patches, resolution tiers, resizing and token cost per image."}, {"title": "OpenAI: Images and vision", "url": "https://developers.openai.com/api/docs/guides/images-vision", "note": "Patch-based and tile-based image token rules, detail levels and multipliers."}, {"title": "Google: Gemini media resolution", "url": "https://ai.google.dev/gemini-api/docs/media-resolution", "note": "Token budgets per image, video frame, PDF page and second of audio."}, {"title": "Faysse et al.: ColPali (2024)", "url": "https://arxiv.org/abs/2407.01449", "note": "Retrieving document pages by embedding their images with late interaction."}]
practice: ["vit-patchify", "clip-cosine-retrieval", "log-mel-spectrogram", "clip-zero-shot-classifier"]
tools: ["Claude API", "OpenAI API", "Gemini API", "ColPali"]
---

## Everything becomes tokens

A vision-language model does not read an image the way it reads text, but the bill treats it the same way. An image encoder (usually a Vision Transformer, see [Vision Transformers](lesson:dlcore:04.05)) cuts the picture into square patches, embeds each patch, and a projector maps those vectors into the language model's embedding space. From there they are ordinary input positions: they take context, they are prefilled, they are billed at the input price, and they sit in the KV cache. Audio and video go through their own encoders and are charged per second.

~~~dg-flow How an image reaches the language model.
Image -(resize to provider limits)-> Pixels -(cut into 28 or 32 px patches)-> Patch vectors -(projector)-> *Image tokens -(prepend or interleave)-> Text tokens -> LLM
~~~

The resize step is where cost and quality are decided, and every provider does it differently. [Vision-language and multimodal inference](lesson:ieng:26.03) covers the serving side; this lesson covers what you control from the API.

## Image token rules, provider by provider

| Provider (late 2026) | Rule | Cap |
| --- | --- | --- |
| Anthropic Claude | $\lceil w/28 \rceil \times \lceil h/28 \rceil$ tokens | High-resolution tier (Claude 4.7 and later): 2,576 px long edge, 4,784 tokens. Older models: 1,568 px, 1,568 tokens |
| OpenAI patch-based models (GPT-5.2 and later) | $\lceil w/32 \rceil \times \lceil h/32 \rceil$ patches, times a model multiplier (1.2 for GPT-5.5) | `detail: high` up to 2,500 patches; `low` fits the image in 512 × 512 |
| OpenAI tile-based models (GPT-4o, GPT-4.1, GPT-5.1) | Base tokens plus a fixed cost per 512 px tile after resizing | `detail: low` is base tokens only |
| Google Gemini 3 and later | A budget per image set by `media_resolution` | 280 low, 560 medium, 1,120 high and default, 2,240 ultra-high |

Each provider shrinks oversized images to fit its cap while keeping the aspect ratio. That protects your bill and quietly destroys small text, so the cap matters as much as the rate.

<div class="ex">
<div class="lab">Worked example</div>

**a full-HD screenshot and a half-size one**

1920 × 1080 on a Claude high-resolution-tier model: $\lceil 1920/28 \rceil \times \lceil 1080/28 \rceil = 69 \times 39 = 2{,}691$ tokens (no resize; inside both caps). At 960 × 540: $35 \times 20 = 700$ tokens. At Claude Opus 5.5's \$4 per million input tokens, that is \$10.76 versus \$2.80 per thousand screenshots. On an older standard-tier Claude model the full-HD image is first shrunk to 1456 × 819 and costs 1,560 tokens.

The same two images on GPT-5.5 with `detail: high`: $60 \times 34 = 2{,}040$ patches, times 1.2, is 2,448 tokens; the half-size one is $30 \times 17 = 510$ patches, 612 tokens. With `detail: low` both are fitted into 512 × 288, $16 \times 9 = 144$ patches, 173 tokens. On Gemini 3 both cost up to 1,120 tokens at the default setting and 280 at `low`.

Halving each side cuts tokens by about 3.8 times on Claude and 4 times on GPT-5.5. Whether that is safe depends on the smallest text the task needs to read, which only an eval on your images can tell you.

</div>

Image tokens compound in agents. A computer-use agent that keeps every 1920 × 1080 screenshot in its history for 30 steps resends 1 + 2 + … + 30 = 465 images in total, 1.25 million image tokens, about \$5.00 of input on Opus 5.5 before any text. Keeping only the last three screenshots sends 87 images, 234,000 tokens, about \$0.94. Prompt caching cuts this further, and [Context engineering](lesson:aieng:09.01) covers clearing old tool results generally.

## Audio and video

Audio is billed by duration. Gemini's documentation gives 25 tokens per second for Gemini 3 and later (32 per second on earlier models), so an hour of audio is 90,000 tokens. Video at the default setting is sampled at one frame per second, about 70 tokens per frame plus the audio track, close to 100 tokens per second: a one-hour video is around 350,000 tokens, a third of a million-token window. For long recordings it is often cheaper to transcribe once with a speech-to-text model, keep timestamps, and send only the relevant segments.

Real-time voice is a latency problem more than a cost problem. There are two architectures:

- **Cascade:** speech-to-text, then the LLM, then text-to-speech. Each stage is swappable and the transcript is easy to log and evaluate, but the hops add up.
- **Native speech-to-speech:** one realtime model takes audio in and streams audio out, keeping tone and handling interruptions more naturally, with fewer hops.

An illustrative cascade budget: end-of-turn detection 300 ms, final transcript 150 ms, LLM time to first token 350 ms, first sentence of text 150 ms, first audio from speech synthesis 150 ms, network 100 ms, for 1.2 seconds before the user hears anything. Everything must stream, and end-of-turn detection is often the largest and hardest part: too eager and the assistant interrupts people, too patient and it feels slow. [Latency budgeting](lesson:aieng:12.05) covers per-stage budgets.

## Documents: OCR pipeline or vision model

| | OCR and layout parsing, then text LLM | Vision model reads page images |
| --- | --- | --- |
| Cost per page | Hundreds of text tokens; parsing is a one-off | 1,000 to 2,700 image tokens on every call |
| Layout, multi-column, forms | Depends on the parser; errors are silent | Usually good |
| Charts and diagrams | Lost, unless captioned separately | Read, but values are approximate |
| Complex tables | Often garbled cell order | Better, but check row and column counts |
| Determinism and citations | Same text every run; exact character spans | Varies with sampling; cite by page |
| Search and retrieval | Standard chunk embeddings and BM25 | Needs captions or page-image embeddings |

Hosted APIs blur the line. Claude converts each PDF page into an image and also extracts its text, so a page costs 1,500 to 3,000 text tokens (per Anthropic's documentation) plus the image tokens. Gemini 3 budgets 560 tokens per PDF page at its default setting.

Practical rules: parse once with a layout-aware parser for retrieval and citations; send page images to the model only for the few retrieved pages where layout or figures matter; for charts, ask for the extracted data as a table and verify any number used downstream against the source. [Chunking and document parsing](lesson:aieng:11.04) goes deeper into parsers.

## Retrieving pages as images

ColPali skips text extraction for retrieval. A vision-language model (PaliGemma-3B in the paper) embeds each page image as one 128-dimensional vector per patch, 1,030 vectors per page. A query is embedded as one vector per token, and the score is late interaction (MaxSim), as in ColBERT:

$$
s(q, d) = \sum_{i} \max_{j} \; \mathbf{q}_i^\top \mathbf{d}_j
$$

Each query token finds its best-matching patch, and the matches are summed. On the paper's ViDoRe benchmark of visually rich documents it beat text-extraction pipelines while being simpler to build.

Storage is the price. At 2 bytes per value, $1{,}030 \times 128 \times 2 = 263{,}680$ bytes, the 257.5 KB per page the paper reports. A 100,000-page corpus needs about 26 GB of vectors, against under 1 GB for two 1,024-dimension float32 vectors per page. Pooling similar patches cut the vector count by two-thirds while keeping 97.8% of retrieval quality in the paper's tests. Late-interaction scoring and its serving costs are covered in [Reranking](lesson:aieng:11.03).

## Where this breaks

- **Silent downscaling.** A dense invoice photographed at 4,000 px wide is shrunk to the cap and its small print becomes unreadable. Crop to the region of interest instead of sending the whole page.
- **Coordinates after resizing.** Boxes or click positions refer to the resized image unless the provider maps them back. Resize yourself so you know the scale.
- **Confident chart readings.** Models read bar heights and line values approximately and rarely say so.
- **Instructions inside images.** Text in a screenshot or scanned letter can carry a prompt injection, exactly like text in a web page ([Prompt injection](lesson:aieng:12.03)).
- **Personal data.** Screenshots and scans capture names, addresses and card numbers you never meant to send. Redact or crop before upload.
- **Cost estimates from text-only habits.** Forgetting that images are resent with history, or that a 60-page PDF is 60 images, produces bills several times the estimate. Count with the provider's token counter.

## What interviewers probe

- How an image becomes tokens, and the cost of one image under a given provider's rule.
- OCR pipeline versus VLM for document question answering, and when to combine them.
- How ColPali-style retrieval works, what MaxSim computes, and what it costs to store.
- Cost control for screenshot-heavy agents.
- The latency budget of a voice assistant and where you would cut it.

<div class="take">
<div class="lab">Key takeaway</div>

images, audio and video are input tokens with provider-specific rates and caps, so resize on purpose, trim them from agent history, and count before you ship. For documents, keep cheap, citable text extraction for retrieval and send page images only where layout, tables or figures carry the answer.

</div>

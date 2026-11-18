---
id: "01.05"
course: "ai-engineering-course"
module: "01"
slug: "choosing-a-model"
title: "Choosing a model"
keyIdeas: ["Choose on five axes: quality on your task, cost, latency, context and modality, and licence and data terms. Quality on your own eval set is the gate; the rest decide among models that pass.", "Public benchmarks and arenas give a shortlist, not an answer: they are contaminated, gamed and measured on someone else's distribution.", "Cost per request is input tokens times input price plus output (and thinking) tokens times output price; multiply by traffic before comparing.", "Self-hosting open weights is a fixed cost that only beats an API at high, steady volume, and only against API models of similar quality.", "Models are retired on a schedule. Pin versions, keep the eval that qualified the model, and put a gateway in front so you can switch."]
question: "A support assistant handles 100,000 requests a day, each about 2,000 input and 300 output tokens. Your team proposes self-hosting an open-weights model instead of a hosted API to cut cost. How do you decide?"
answer: "First check quality: run the open model and the API candidates on our eval set; only models that pass the bar are in the comparison. Then cost per month: at 2,000 in and 300 out, a mid-tier API at 2 and 10 dollars per million tokens is 0.7 cents a request, about 21,000 dollars a month at 3 million requests; a small-tier API at 0.10 and 0.50 is about 1,050. Self-hosting needs at least two GPUs for availability plus engineer time, roughly 7,000 to 8,000 dollars a month in my illustrative numbers, so it beats the mid-tier API but loses to the small tier. Then latency at p95 from our region, data terms (retention, residency), licence, and the operational cost of patching, scaling and on-call. At this volume I would pick the cheapest API model that passes the eval, add prompt caching, and revisit self-hosting if volume grows 10 to 30 times or data rules require it."
rubric: ["Makes quality on the team's own eval the first gate", "Computes cost per request and per month from token counts and prices", "Treats self-hosting as fixed cost with a minimum footprint and engineer time", "Compares against the cheapest API model that passes, not only the frontier one", "Mentions latency, data and licence terms, and caching or routing as alternatives"]
sources: [{"title": "Chiang et al.: Chatbot Arena (2024)", "url": "https://arxiv.org/abs/2403.04132", "note": "How the crowd-sourced pairwise leaderboard works and what it measures."}, {"title": "Singh et al.: The Leaderboard Illusion (2025)", "url": "https://arxiv.org/abs/2504.20879", "note": "Private testing and data asymmetries that distort Arena rankings."}, {"title": "Anthropic: Pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing", "note": "Per-token input, output, cache and batch prices used in the worked example."}, {"title": "Anthropic: Model deprecations", "url": "https://platform.claude.com/docs/en/about-claude/model-deprecations", "note": "Retirement schedule and what happens to requests to a retired model."}]
practice: ["fit-elo-ratings-from-pairwise-votes", "model-cascade-routing", "request-cost-with-thinking-tokens", "optimize-throughput-and-cost-under-latency-slo", "bootstrap-confidence-interval"]
tools: ["LMArena", "Artificial Analysis", "vLLM", "Claude API"]
---

## Five axes, one gate

Model choice is the first big decision in an LLM feature and the one teams most often make by reputation. A useful frame is five axes.

| Axis | What to measure | Typical deal-breaker |
| --- | --- | --- |
| Quality | Pass rate on your eval set, per slice | Fails a slice you care about (a language, a document type) |
| Cost | Price per request at your token profile, times traffic | A frontier model on a high-volume, easy task |
| Latency | Time to first token and output speed, p50 and p95, from your region | Voice or autocomplete that needs sub-second responses |
| Context and modality | Window size, image/audio/PDF input, structured output, tool calling | Needs 300k tokens of documents or native audio |
| Licence and data terms | Retention, training on your data, region, open-weight licence clauses | Regulated data that may not leave a region |

Quality is a gate, not a weight: a cheap model that fails your eval is not cheap, because every wrong answer has a cost downstream. Among the models that pass, cost, latency and terms decide.

~~~dg-flow A model selection loop that survives the next model release.
Public benchmarks -(shortlist 3 to 5)-> Candidates -(run)-> *Your eval set -(passing models)-> Cost + p95 latency at your traffic -(check)-> Data + licence terms -(pin version)-> Decision
~~~

## Public benchmarks and your eval

Leaderboards are good at one thing: telling you which five models are worth testing. They are poor at telling you which one wins on your task.

- **Contamination.** Benchmark questions leak into training data, so scores rise without capability rising. Newer benchmarks rotate questions to resist this, which also makes scores hard to compare over time.
- **Distribution.** A coding benchmark measures someone else's repositories; a maths benchmark says little about extracting fields from invoices.
- **Arena effects.** Chatbot Arena ranks models from crowd votes on pairwise battles with a Bradley-Terry model (the practice problem fits one). Votes reward style, length and formatting as well as correctness. "The Leaderboard Illusion" documented private testing of many variants before release (27 for one Llama 4 launch) with only the best published, and data asymmetries: Google and OpenAI each received about a fifth of all Arena battles.

Your own eval set settles it: 100 to 300 real inputs with graded expected outputs, split into the slices you care about. [Building an eval set](lesson:aieng:10.02) covers how to build it and how many cases you need; with 200 cases a 5-point gap between two models is inside the noise unless you compare them on the same cases.

## Cost per request, then per month

For one request,

$$
c = t_{\text{in}} \, p_{\text{in}} + (t_{\text{out}} + t_{\text{think}}) \, p_{\text{out}}
$$

where prices are per token. Thinking tokens from reasoning models are billed as output even when you never see them; [Using reasoning models in apps](lesson:aieng:08.06) works through that. Count tokens with the model's own tokenizer ([Tokenization in practice](lesson:aieng:01.04)).

<div class="ex">
<div class="lab">Worked example</div>

**API tiers versus self-hosting at three traffic levels**

A support assistant sends 2,000 input tokens and gets 300 output tokens per request. API prices are Anthropic's list prices in November 2026 (pricing page in sources): a large model at \$4 / \$20 per million input / output tokens, a mid-tier model at \$2 / \$10 and a small model at \$0.10 / \$0.50.

Per request: large $2{,}000 \times 4 + 300 \times 20 = 14{,}000$ millionths of a dollar, so 1.40 cents; mid-tier 0.70 cents; small 0.035 cents.

Self-hosting an open-weights mixture-of-experts model that fits on one 80 GB GPU (illustrative assumptions, measure your own): \$2.50 per GPU-hour, so \$1,825 per GPU-month; each GPU sustains 5 requests per second at the latency target; provision for a peak of twice the average plus one spare, never fewer than 2 GPUs; engineer time for serving, upgrades and on-call at \$4,000 a month, rising to \$8,000 at the largest size.

| Monthly | 1k req/day | 100k req/day | 3M req/day |
| --- | --- | --- | --- |
| Large API | \$420 | \$42,000 | \$1,260,000 |
| Mid-tier API | \$210 | \$21,000 | \$630,000 |
| Small API | \$10.50 | \$1,050 | \$31,500 |
| Self-hosted (GPUs) | 2 GPUs | 2 GPUs | 15 GPUs |
| Self-hosted total | \$7,650 | \$7,650 | \$35,375 |

At 3 million a day the average load is 34.7 requests per second, peak 69.4, so 14 GPUs plus a spare. Read the table by quality tier. If only a mid-tier API model passes your eval and the open model also passes, self-hosting wins by 18 times at the top volume and loses below about 1 million requests a month. If the small API model passes, it beats self-hosting at every volume here.

The API column also moves with features you control. If 1,500 of the 2,000 input tokens are a fixed system prompt and the mid-tier model's cache reads cost \$0.10 per million, the request drops to $500 \times 2 + 1{,}500 \times 0.10 + 300 \times 10 = 4{,}150$ millionths, 41% less (ignoring cache writes). Offline jobs on a batch API pay half price.

</div>

Self-hosting is a fixed cost with a floor: two GPUs and someone on call, whether you serve 10 requests or 10 million. It also carries costs the table hides: GPU availability, upgrading serving software, capacity planning ([Capacity planning from benchmark data](lesson:ieng:25.04)) and slower access to new models. Teams self-host for steady high volume, strict data rules or a fine-tuned open model, and rarely for a prototype.

## Latency is a separate axis

Two models at the same price can differ by several times in time to first token and in output tokens per second, and hosted latency varies by region and time of day. Measure p50 and p95 with your real prompt lengths from where your servers run; independent trackers such as Artificial Analysis publish speed and price comparisons that help with the shortlist. Long outputs dominate total time, so a model that answers in 150 tokens beats a faster-per-token model that writes 600. [Latency budgeting](lesson:aieng:12.05) splits the budget by stage.

## Routing and cascades

You rarely need one model for all traffic. A **router** sends each request to a model chosen by rules or a classifier (short FAQ questions to the small model, long complaints to the mid-tier). A **cascade** tries the small model first and escalates when a check fails or confidence is low. With the prices above, if 70% of requests stop at the small model and 30% escalate (paying for both calls), the mean cost is $0.035 + 0.3 \times 0.70 = 0.245$ cents, 65% below sending everything to the mid-tier model. The catch is that the check itself must be reliable; a router is only as good as the eval of each slice it routes. [Cost budgeting](lesson:aieng:12.04) covers the threshold maths and [Reliability: gateways, retries and fallbacks](lesson:aieng:12.06) the gateway that makes switching cheap.

## Licence, data and deprecation

- **Licences differ for open weights.** gpt-oss and Qwen3 are Apache 2.0. Llama models use Meta's community licence, with an acceptable-use policy and extra terms above 700 million monthly active users. Read the licence before you build on a model.
- **Data terms differ for APIs.** Check retention, whether inputs train future models, zero-data-retention eligibility and regional processing. Region pinning can cost more: Anthropic charges 1.1 times for US-only inference on recent models.
- **Hosted models are retired.** Providers publish deprecation schedules, and requests to a retired model fail. Pin an exact model version in config rather than a moving alias, keep the eval that qualified it, and budget a re-qualification when a successor ships. A successor can also bring a new tokenizer, changing token counts for the same prompt.
- **Open weights never retire, but they age.** You own the upgrade path and the security fixes in the serving stack.

## Where this breaks

- **Choosing by leaderboard rank.** The top model on a public board can lose on your data, especially on non-English slices or long documents.
- **Comparing list prices instead of request cost.** A model with cheaper tokens that writes three times as much, or thinks for 2,000 hidden tokens, costs more.
- **Pricing self-hosting at 100% utilisation.** Real traffic has peaks, nights and failover; the GPUs you pay for sit idle much of the time.
- **Evaluating once.** Providers update models behind aliases and your traffic drifts. Re-run the eval on a schedule and on every model change.
- **Single-provider lock-in.** One outage or rate limit takes the feature down. A gateway with a tested fallback model is cheap insurance.

## What interviewers probe

- Walk through choosing a model for a given product: axes, gate, and the order you check them.
- Build the cost model on a whiteboard: tokens per request, price, traffic, and the crossover for self-hosting.
- Why Arena and benchmark scores mislead, and what you would run instead.
- How routing or a cascade changes cost, and what has to be true for it to be safe.
- How you protect a product against a model being deprecated or changed under an alias.

<div class="take">
<div class="lab">Key takeaway</div>

shortlist from benchmarks, decide on your own eval, then compare cost per month at your real token profile and traffic. Self-hosting is a fixed cost that wins only at high steady volume against API models of the same quality, and every choice needs a pinned version and a path to switch.

</div>

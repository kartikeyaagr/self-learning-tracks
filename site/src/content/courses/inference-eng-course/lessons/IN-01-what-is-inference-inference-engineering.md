---
id: "IN.01"
course: "inference-eng-course"
module: "IN"
slug: "what-is-inference-inference-engineering"
title: "What Is Inference & Inference Engineering?"
keyIdeas: ["Inference is running a trained model on new inputs. For LLMs it means turning a prompt into generated tokens.", "Training is a one-off, throughput-only job. Inference runs forever, on live traffic, with latency targets.", "An inference engineer makes a given model faster, cheaper and more reliable without changing what it outputs (or while measuring how much it changes).", "The work spans math (memory, FLOPs), systems (scheduling, batching, caching), hardware (GPUs, interconnects) and operations (autoscaling, observability).", "The core trade-off is latency for one user vs throughput across all users. Most techniques move you along that curve."]
question: "In two sentences, explain the difference between training a model and serving it, and name one metric an inference engineer owns."
answer: "Training adjusts a model's weights using huge batches of data, usually offline, and is judged by final model quality. Inference runs the frozen weights on live requests, one token at a time, and is judged by latency, throughput and cost. An inference engineer owns metrics such as time to first token, tokens per second per GPU, or cost per million tokens."
sources: [{"title": "Lilian Weng: Large Transformer Model Inference Optimization", "url": "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/", "note": "Survey of the main optimization families."}, {"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "The back-of-envelope maths behind inference costs."}]
tags: ["basics"]
---

## What inference means

A model is trained once and then used many times. Using it is called **inference**: you feed in an input, the weights stay frozen, and you get an output. For a large language model (LLM), the input is a prompt and the output is a stream of tokens, produced one at a time.

That last detail matters more than anything else in this course. A chat model does not produce an answer in one shot. It produces one token, appends it to the input, runs the whole network again, and repeats until it emits a stop token or hits a limit. A 500-token answer is 500 sequential passes through a network with billions of parameters. Almost every technique you will learn exists because of that loop.

## Training versus serving

Training and inference use the same network but are opposite problems.

| Concern | Training | Inference |
| --- | --- | --- |
| Goal | Find the best weights | Produce outputs from fixed weights |
| Who waits | Nobody waits on one step | A user waits on every token |
| Batch | Fixed, huge (millions of tokens) | Whatever traffic arrives, changing each step |
| Work per step | Forward + backward + optimizer | Forward only |
| Usual bottleneck | Compute (FLOPs) | Memory bandwidth during decode |
| Lifetime | Weeks, then it stops | Runs for as long as the product exists |
| Cost driver | Total GPU-hours for one run | GPU-hours per million tokens, forever |

Training is a batch job you can schedule. Inference is a live service with traffic spikes, latency targets and a bill that keeps running whether the GPUs are busy or idle. For a popular model, total inference spend overtakes training spend quickly, because training is paid once and inference is paid per token.

<div class="ex">
<div class="lab">Worked example</div>

**when does inference cost more than training?**

Suppose training a model cost \$5M of GPU time. You serve it at an all-in cost of \$0.50 per million output tokens and your users generate 20 billion tokens per day.

Daily serving cost = 20,000 × \$0.50 = \$10,000. After 500 days, serving has cost as much as training.

At 200 billion tokens per day the crossover is 50 days. Popular models hit this quickly, which is why companies hire people to make serving cheaper.

</div>

## What an inference engineer does

An inference engineer takes a model as given and makes serving it better. "Better" has four parts:

- **Latency**: how long one user waits for the first token (time to first token, TTFT) and how fast the rest stream in (time per output token, TPOT).
- **Throughput**: how many tokens per second a GPU or cluster produces across all users.
- **Cost**: dollars per million tokens at a stated latency target and quality.
- **Reliability**: the service stays up, degrades gracefully under overload and recovers from GPU failures.

These pull against each other. Packing more users onto a GPU raises throughput and lowers cost per token, but each user's tokens arrive more slowly. Most of the job is choosing where to sit on that curve and then moving the whole curve outward with better engineering.

~~~dg-flow Where inference sits relative to training.
Data -(train)-> Weights -(load)-> *Inference server -(tokens)-> Users
~~~

## The latency and throughput trade-off, with numbers

Try the figure below. Each decode step reads all the model weights from GPU memory once. If one user is decoding, that read produces one token. If 64 users decode together, the same read produces 64 tokens. Throughput climbs almost for free until the GPU runs out of memory bandwidth or compute. Each user's speed falls slowly at first, then sharply.

~~~dg-raw
%%widget batch Move the batch size and watch total throughput rise while per-user speed falls.
model = Llama 3 8B
b = 4
%%
~~~

At batch 1 an H100 serving an 8B model produces about 150 to 200 tokens per second for one person and leaves most of its compute idle. At batch 64 it can produce several thousand tokens per second in total while each user still sees a comfortable reading speed. That gap is the main economic lever in inference, and you will spend several modules learning to exploit it safely.

## Why it is its own discipline

Serving an LLM is not like serving a web app:

- **Requests differ in size by 1000×.** One prompt is 20 tokens, another is 100,000. One answer is 5 tokens, another 4,000.
- **Work is sequential.** You cannot compute token 50 before token 49.
- **Memory grows during a request.** Every generated token adds to a per-request cache (the KV cache), so a request's footprint is not known when it starts.
- **The hardware is expensive and fussy.** An H100 rents for \$2 to \$4 per hour. Leaving it 70% idle is the default if you do nothing clever.

A naive server wastes most of the GPU. Techniques like KV caching, continuous batching, paged memory, quantization and speculative decoding each recover a large share of that waste. They interact in ways you only learn by measuring, which is why the course leans heavily on benchmarking.

## Deep dive: the cost model behind "inference is the bigger bill"

Training a dense model with ⟦m6⟧ parameters on ⟦m7⟧ tokens costs about

⟦M0⟧

The factor 6 is 2 FLOPs per parameter for the forward pass (one multiply and one add) and about 4 more for the backward pass. Generating one token costs only the forward pass:

⟦M1⟧

So the number of tokens you must serve before inference has used as many FLOPs as training is

⟦M2⟧

That number does not depend on model size. A model trained on 15 trillion tokens matches its training FLOPs after serving about 45 trillion tokens.

FLOPs are not dollars, though. Training runs at a high model FLOPs utilization (MFU), often 35 to 50%. Decode at small batch is memory bound and may use only 1 to 5% of peak FLOPs. In GPU-hours, the break-even point therefore arrives 10 to 40 times sooner:

⟦M3⟧

<div class="ex">
<div class="lab">Worked example</div>

**Llama 3 8B**

Training: ⟦m8⟧ FLOPs. At 40% MFU on H100s (989 TFLOP/s BF16 peak), that is about ⟦m9⟧ GPU-seconds, about 510,000 GPU-hours.

Serving at batch 64, an H100 produces about 3,000 tokens per second. That is about 1.1×10⁷ tokens per GPU-hour.

510,000 GPU-hours of serving produce about 5.5×10¹² tokens. A service generating 1 billion tokens a day passes that in about 15 years. One generating 100 billion a day, about two months.

</div>

~~~dg-raw
%%chart Cumulative GPU-hours: one training run against serving at two traffic levels.
x = 0, 30, 60, 90, 120, 150, 180
xlab = Days in production
ylab = GPU-hours (thousands)
Training (one-off): 510, 510, 510, 510, 510, 510, 510
Serving at 100B tokens/day: 0, 273, 545, 818, 1091, 1364, 1636
Serving at 10B tokens/day: 0, 27, 55, 82, 109, 136, 164
%%
~~~

## Deep dive: why batching trades latency for throughput

One decode step must read every weight once, whatever the batch size ⟦m10⟧. It also does ⟦m11⟧ FLOPs per sequence. With bandwidth ⟦m12⟧ and peak compute ⟦m13⟧, a simple model of step time is

⟦M4⟧

The first term is the time to read BF16 weights (2 bytes per parameter). The second is the time to do the math. They are equal at

⟦M5⟧

For an H100, ⟦m14⟧. Below that batch, adding sequences is almost free, so total throughput ⟦m15⟧ grows nearly linearly while each user's speed ⟦m16⟧ barely falls. Above it, step time grows with ⟦m17⟧, total throughput flattens, and each user slows down. In practice KV reads and imperfect kernels move the knee well below 295, often to between 64 and 160.

~~~dg-raw
%%chart Model: total and per-user throughput for an 8B model on an H100 (weights only, 60% efficiency).
x = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
logx = 1
logy = 1
xlab = Batch size
ylab = Tokens per second
Total tokens/s: 126, 251, 503, 1005, 2010, 4020, 8040, 16080, 32160, 37100
Per-user tokens/s: 126, 126, 126, 126, 126, 126, 126, 126, 126, 72
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

inference engineering is serving a fixed model with the best latency, throughput, cost and reliability you can prove with measurements. The token-by-token loop is the root cause of almost every problem you will solve.

</div>

---
id: "10.10"
course: "system-design-101"
module: "10"
slug: "reservoir-sampling"
title: "Reservoir sampling"
keyIdeas: ["Reservoir sampling picks a uniform random sample of k items from a stream of unknown length in one pass.", "Algorithm R: keep the first k items. For item number i (i > k), with probability k/i replace a random slot in the reservoir with it.", "Every item ends up in the sample with equal probability k/n, no matter when it arrived.", "Uses: sample logs or requests for debugging, keep a representative set of latency samples, approximate statistics on huge data.", "Weighted variants exist for streams where items should have different chances."]
question: "Sample 1,000 items uniformly from a stream of unknown length. Explain the replacement probability."
answer: "Fill the reservoir with the first 1,000 items. For the i-th item after that, draw a random integer from 1 to i; if it is at most 1,000, the item replaces that slot, so item i is kept with probability 1,000/i and evicts a uniformly random current member. By induction: an existing item is evicted at step i only if the new item is kept and picks its slot, probability (1,000/i) × (1/1,000) = 1/i, so if it was in the sample with probability 1,000/(i−1) it remains with probability 1,000/(i−1) × (i−1)/i = 1,000/i. The new item also has 1,000/i, so every item seen so far has the same probability 1,000/i of being in the sample at every point, using only 1,000 slots of memory."
sources: [{"title": "Wikipedia: Reservoir sampling", "url": "https://en.wikipedia.org/wiki/Reservoir_sampling", "note": "Algorithm R."}]
---

## The problem

You want a **uniform random sample** of k items from a stream whose length n you don't know in advance, in one pass, without storing the stream. "Uniform" means every item that has passed has the same chance of being in the sample, and every set of k items is equally likely.

This comes up constantly in systems:

- Keep 1,000 sample log lines from a day of logs, so an engineer can see what typical errors look like without storing billions of lines.
- Sample requests for distributed tracing, where tracing everything would cost more than the service itself.
- Show a random sample of 100 rows from a query result that is still streaming in.
- Pick a survey panel from users as they log in during the day.
- Estimate statistics (median, distribution of sizes) over a stream by computing them on a sample.

The difficulty is that you don't know n. If you knew a day had exactly 10 million log lines, you could keep each one with probability 1,000/10,000,000 and get about 1,000. But traffic varies, the day isn't over yet, and you want a valid sample at any moment.

### Why the obvious approaches fail

- **Keep the first k.** Biased: the sample only contains the start of the day.
- **Keep everything, then pick.** Correct, but needs memory proportional to n, which is the thing you're trying to avoid.
- **Keep each item with a fixed probability p.** Needs n to choose p, and the sample size varies: too small on a quiet day, too large on a busy one.
- **Keep the last k.** Biased the other way: only the end of the day.

~~~dg-compare Ways to keep a sample of k from a stream of unknown length. Only reservoir sampling is uniform with fixed memory.
[Fixed probability p]
Keep each item with chance p
- Must know n to choose p
- Sample size varies with traffic
+ Trivial, stateless
[First k or last k]
Keep a window at one end
- Biased towards one end
+ Trivial
[*Reservoir sampling]
Keep k, replace with chance k/i
+ Uniform at every moment
+ Fixed memory: k items
+ One pass, n unknown
~~~

**Reservoir sampling** solves it, and the simplest version is a few lines of code.

## Algorithm R

1. Put the first k items into the **reservoir**, an array of k slots.
2. For the i-th item (i > k), draw a random integer j between 1 and i inclusive. If j ≤ k, **replace reservoir slot j** with item i. Otherwise, discard item i.

So item i enters the reservoir with probability **k / i**, and if it does, it evicts a uniformly random current member (because j, conditioned on being ≤ k, is uniform over 1 to k).

```
reservoir = []
for i, item in enumerate(stream, start=1):
    if i <= k:
        reservoir.append(item)
    else:
        j = random.randint(1, i)
        if j <= k:
            reservoir[j - 1] = item
```

~~~dg-flow Algorithm R with k = 1,000: the first thousand fill the reservoir, then item i replaces a random slot with probability 1,000 / i.
Items 1..1,000 -(fill)-> *Reservoir of 1,000 -(item i: keep with prob 1,000 / i)-> Replace random slot -(else)-> Discard
~~~

Notice how the acceptance probability falls as the stream grows. Item 2,000 has a 50% chance; item 100,000 has 1%; item 10 million has 0.01%. Late items are rarely accepted, but there are many more of them, which is exactly what keeps the sample balanced.

## Why it is uniform

The proof is by induction on the number of items seen. The claim: after i items (i ≥ k), each of them is in the reservoir with probability k/i.

**Base case.** After k items, all k are in the reservoir: probability k/k = 1. True.

**Step.** Suppose after i − 1 items, each of them is in the reservoir with probability k/(i − 1). Now item i arrives:

- Item i is included with probability **k/i**. That matches the claim for item i.
- An earlier item that is currently in the reservoir is evicted at step i only if item i is included **and** item i picks its particular slot: probability (k/i) × (1/k) = 1/i. So it survives step i with probability 1 − 1/i = (i − 1)/i.
- Its probability of being in the reservoir after step i is therefore k/(i − 1) × (i − 1)/i = **k/i**.

Every item seen so far has the same probability k/i of being in the sample, at every point in the stream. That's what lets you stop at any time and have a valid sample. (A stronger version of the argument shows every subset of size k is equally likely, not just every individual item.)

<div class="ex">
<div class="lab">Worked example</div>

**a tiny trace with k = 2**

Stream A, B, C, D. After B: reservoir {A, B}, each with probability 1.

C arrives (i = 3): kept with probability 2/3. If kept, it replaces A or B with equal chance. So A survives with probability 1 − (2/3)(1/2) = 2/3, and the same for B. All three items now have probability 2/3, which is k/i = 2/3.

D arrives (i = 4): kept with probability 2/4 = 1/2. Each current member is evicted with probability (1/2)(1/2) = 1/4, so survives with 3/4. A's probability: 2/3 × 3/4 = 1/2. C's: 2/3 × 3/4 = 1/2. D's: 1/2. All four have probability 1/2 = k/i.

</div>

~~~dg-bars Probability that each item is in a reservoir of 1,000 after 5,000 items. The first item and the last are equally likely: that is what uniform means.
Item 1 | 20 | 20%
Item 1,000 | 20 | 20%
Item 2,500 | 20 | 20%
*Item 5,000 | 20 | 20%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**k = 1,000, item 5,000**

Item 5,000 is kept with probability 1,000/5,000 = 20%. If kept, it replaces a random one of the 1,000 slots.

Each of items 1 to 4,999 was in the reservoir with probability 1,000/4,999 and survives this step with probability 1 − 1/5,000, giving 1,000/4,999 × 4,999/5,000 = 1,000/5,000 = 20%.

All 5,000 items now have a 20% chance of being in the sample. By item 1,000,000, each has 0.1%. Memory never changed: 1,000 slots.

</div>

## How much work does it do?

Algorithm R is O(n): one random number per item. How many replacements happen? The expected number of times item i is accepted is k/i, so the total expected replacements after the first k is the sum of k/i for i from k + 1 to n, which is about k × ln(n/k).

<div class="ex">
<div class="lab">Worked example</div>

**replacements over a day**

k = 1,000 and n = 1 billion log lines. Expected replacements: 1,000 × ln(1,000,000) ≈ 1,000 × 13.8 ≈ 13,800. So of a billion items, only about 14,000 ever enter the reservoir after it fills. Algorithm R still generates a billion random numbers to decide that, which is where the next algorithm saves time.

</div>

## Faster: Algorithm L

Since later items are almost always discarded, **Algorithm L** (Li, 1994) computes how many items to **skip** before the next replacement, instead of flipping a coin for each one. The skip length follows a geometric distribution whose parameter it updates after each replacement, and the expected running time becomes **O(k(1 + log(n/k)))**, which is optimal: proportional to the number of replacements, not the number of items. Skipped items aren't even examined, which matters when reading items is cheap (seeking forward in a file, skipping records in a buffer) but random number generation or per-item work isn't. For the billion-line example, that's about 14,000 random draws instead of a billion.

## Weighted and distributed variants

### Weighted sampling

Sometimes items should be sampled in proportion to a weight: sample requests weighted by their latency, so slow ones are over-represented, or sample users weighted by spend. **A-Res** (Efraimidis and Spirakis) gives each item a key u^(1/w), where u is uniform random in (0, 1) and w is the item's weight, and keeps the k items with the largest keys in a min-heap. Heavier items get keys closer to 1 and are more likely to be kept.

### Distributed sampling

Algorithm R can't be run across ten servers and simply merged: each server's reservoir represents a different number of items, so combining them naively over-represents the quiet servers. The trick is to make the sample defined by a property of each item:

- Give every item a random **tag** in [0, 1).
- Keep the k items with the **smallest** tags (a max-heap of size k, so you can evict the largest).

The k smallest tags among n items form a uniform random sample, since tags are independent and identically distributed. Each server keeps its local k smallest. Merging means taking the k smallest of the union, which is exactly the k smallest tags of the combined stream: equivalent to sampling the combined stream directly. That makes it mergeable like the sketches in lesson 10.05.

~~~dg-flow Distributed reservoir sampling by random tags: each server keeps its k smallest tags, and the merge keeps the k smallest overall.
Server 1: k smallest tags, Server 2: k smallest tags -(union)-> *Keep k smallest -> Uniform sample of all streams
~~~

If the tag is a hash of the item instead of a random number, the same item gets the same tag everywhere, which turns this into a sample of **distinct** items. That is exactly the idea behind the Theta sketch for distinct counting.

## Sampling in real systems

- **Tracing.** Head-based sampling decides at the start of a request (keep 1%). Tail-based sampling buffers spans and decides at the end, so it can keep all errors and slow requests plus a reservoir of normal ones. Reservoir sampling per endpoint gives each endpoint a guaranteed number of examples, even rare ones.
- **Logging.** A per-error-type reservoir keeps 100 examples of each distinct error message per hour, however noisy the noisiest one is.
- **Databases.** `TABLESAMPLE` and query planners' statistics collectors use sampling to estimate distributions without full scans.

## Limits and pitfalls

- **The reservoir must fit in memory.** For k in the millions, use the distributed tag approach with spill to disk.
- **When k is a large fraction of n**, simpler methods (shuffle, or Bernoulli sampling with a known n) are fine.
- **Bad randomness biases the sample.** Using `random() % i` with a small random range, or a poor generator, skews acceptance. Use a proper uniform integer generator.
- **Time-sensitive samples.** A day's reservoir is uniform over the day. If you want "recent" examples, use a sliding window of reservoirs (one per hour) or a decaying weighted scheme.
- **Small samples hide rare events.** A uniform sample of 1,000 from 1 billion contains, on average, one example of something that happens once per million. Stratify (one reservoir per category) when rare categories matter.

## In an interview

If the design needs examples from a stream ("keep sample logs", "trace a sample of requests"), say "reservoir sampling: keep k, then accept item i with probability k/i into a random slot", and give the one-line proof: the newcomer gets k/i; an existing item survives with (i − 1)/i, so k/(i − 1) × (i − 1)/i = k/i. Mention random tags for a mergeable distributed version.

<div class="take">
<div class="lab">Key takeaway</div>

reservoir sampling keeps a uniform sample of k items from a stream of unknown length: keep the first k, then keep item i with probability k/i, replacing a random slot. By induction every item seen is in the sample with probability k/i at every moment. Algorithm L skips ahead for speed, weighted keys handle weights, and keeping the k smallest random tags makes it mergeable across servers.

</div>

---
id: "10.04"
course: "system-design-101"
module: "10"
slug: "hyperloglog-cardinality-estimation"
title: "HyperLogLog cardinality estimation"
keyIdeas: ["HyperLogLog estimates the number of distinct items using tiny, fixed memory.", "Idea: hash every item. In uniformly random hashes, a run of k leading zero bits shows up about once per 2^k items. So the longest run you've seen hints at how many distinct items there are.", "One estimate is noisy, so HLL splits hashes into m buckets (by the first bits), tracks the max run per bucket, and combines them with a harmonic mean.", "Redis HLL uses 12 KB for about 0.81% standard error, for counts into the billions.", "Duplicates hash the same, so they never change the estimate."]
question: "Why does the longest run of leading zeros in hashes estimate the number of unique items?"
answer: "With a uniform hash, each bit is an independent fair coin, so a hash starting with k zeros then a 1 has probability 1 in 2ᵏ⁺¹; a long run only becomes likely after many different hashes have been seen, so a maximum run of r suggests about 2ʳ items. The same item always produces the same hash, so duplicates can never raise the maximum, which is why it counts distinct items rather than events. One maximum is very noisy (a single lucky hash can overstate the count many times), so HLL uses the first bits of the hash to split items over many registers, keeps the maximum per register, and combines them with a harmonic mean, giving a standard error of 1.04/√m."
sources: [{"title": "antirez: Redis new data structure, the HyperLogLog", "url": "http://antirez.com/news/75", "note": "Intuition and implementation."}, {"title": "Google: HyperLogLog in Practice (paper)", "url": "https://research.google/pubs/hyperloglog-in-practice-algorithmic-engineering-of-a-state-of-the-art-cardinality-estimation-algorithm/", "note": "HLL++."}, {"title": "Hello Interview: Data structures for big data", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/data-structures-for-big-data", "note": "Bloom, count-min, HLL, quantiles."}]
---

## The problem: count distinct without remembering

"How many distinct users visited today?" An exact answer needs a set of every user seen so far, because each new event must be checked against it: is this user new? The set grows with the answer. A billion users need gigabytes, per day, per site, per metric. If you want unique users per country per device per hour, you multiply that by thousands.

**HyperLogLog** (HLL) answers within about 1% using a fixed 12 KB, however large the count. It sees each item once, never stores the items, and can be merged across machines (lesson 10.05). It is one of the most widely deployed probabilistic data structures: Redis, PostgreSQL extensions, BigQuery's `APPROX_COUNT_DISTINCT`, Druid, Presto and Spark all ship it. To trust it, you should understand why it works.

## The coin-flip intuition

Start with coins. Flip a coin repeatedly and note your longest run of heads in a row. If someone tells you their longest run was 3, they probably didn't flip many times. If it was 13, they must have flipped a lot, because a particular run of 13 heads has probability 1 in 8,192. The longest run is a rough clock of how many flips happened, even though nobody wrote the flips down.

HLL applies this to hashes. A good hash function (such as MurmurHash or xxHash, 64 bits) turns any item into a bit string that looks uniformly random: each bit is 0 or 1 with probability one half, independently. Look at the number of **leading zeros**, the zeros before the first 1:

- Half of all hashes start with `1` (0 leading zeros).
- A quarter start with `01` (1 leading zero).
- One in 8 start with `001`.
- In general, one in 2ᵏ⁺¹ start with exactly k zeros followed by a 1.

~~~dg-bars How likely a random hash is to start with at least k zeros. Each extra zero halves the chance, so a long run is evidence of many distinct hashes.
At least 1 zero | 50 | 1 in 2
At least 3 zeros | 12.5 | 1 in 8
At least 6 zeros | 1.5625 | 1 in 64
At least 10 zeros | 0.0977 | 1 in 1,024
*At least 20 zeros | 0.0000954 | 1 in 1,048,576
~~~

So if the longest run of leading zeros you have seen is r, you have probably seen around 2ʳ **distinct** items. Why distinct? Because the same item always hashes to the same bit string. Seeing user:812 a million times produces one hash a million times, and a repeat can never extend the maximum. The structure is blind to duplicates by construction, which is exactly what count-distinct needs.

<div class="ex">
<div class="lab">Worked example</div>

**how likely is a long run?**

Suppose you hash 1,000 distinct items. The chance that a given hash has at least 10 leading zeros is 1/1,024. The chance that none of the 1,000 do is (1 − 1/1,024)^1,000 ≈ e^(−0.98) ≈ 0.38. So with 62% probability you will have seen a run of 10 or more, and 2¹⁰ = 1,024 is a decent guess of 1,000.

The same calculation shows the weakness. The chance of at least one run of 13 or more among 1,000 hashes is about 1 − e^(−1000/8192) ≈ 11%. One time in nine, a single estimator would say "8,000" when the truth is 1,000. One maximum is far too noisy to use alone.

</div>

## Registers: many small experiments

The fix is to run many experiments and average them, like flipping many coins in parallel. Hashing every item with many different hash functions would be slow, so HLL uses one hash and splits it into two parts:

- The first p bits choose one of m = 2ᵖ **registers** (buckets).
- The remaining bits are scanned for leading zeros. The register keeps the **maximum** value seen, where the value stored is the position of the first 1 bit (leading zeros plus one).

~~~dg-layers How Redis splits one 64-bit hash: 14 bits pick a register, and the run of zeros in the other 50 bits gives the value (Redis actually takes the low 14 bits and counts zeros from the other end, which works the same way).
Bits 1 to 14 | register index, 0 to 16,383
*Bits 15 to 64 | count leading zeros, store position of first 1
Register | 6 bits, keeps the maximum ever seen
~~~

Each register sees about 1/m of the distinct items and estimates its own share. Because each item is routed by its hash, an item always goes to the same register, so duplicates still can't change anything.

~~~dg-flow Each item updates exactly one register: the first 14 bits pick register 9,301, and the rest of the hash has 5 leading zeros, so the register becomes at least 6.
Item user:812 -(64-bit hash)-> First 14 bits: register 9,301 -(rest: 5 leading zeros)-> *register 9,301 = max(old, 6)
~~~

### Combining registers: the harmonic mean

Each register j holds a value M[j], which suggests that register saw about 2^M[j] items. You could average those, but an arithmetic mean is dominated by the one register that got lucky. HLL uses the **harmonic mean** instead, which is the reciprocal of the average of reciprocals, and is pulled towards the small values. The estimate is:

`E = α · m² / Σ 2^(−M[j])`

where the sum runs over all m registers and α is a constant (about 0.7213 for large m) that corrects a known bias. Intuitively, m / Σ 2^(−M[j]) is the harmonic mean of the per-register estimates, and multiplying by m again scales one register's share up to the whole.

<div class="ex">
<div class="lab">Worked example</div>

**why the harmonic mean, with four registers**

Registers hold 3, 3, 3 and 13 (the last one got a lucky hash). Per-register estimates: 8, 8, 8 and 8,192.

Arithmetic mean: (8 + 8 + 8 + 8,192) / 4 = 2,054. One outlier has inflated the answer 250 times.

Harmonic mean: 4 / (1/8 + 1/8 + 1/8 + 1/8,192) = 4 / 0.3751 ≈ 10.7. The outlier barely moves it.

The full estimate multiplies by m and α: about 0.7 × 4 × 10.7 ≈ 30 distinct items in total. (Four registers are far too few for real use; this only shows how the mean behaves.)

</div>

The standard error of the estimate is **1.04 / √m**. Doubling accuracy costs four times the registers. With m = 16,384 the error is 0.81%; with m = 1,024 it is 3.25%; with m = 65,536 it is 0.41%.

~~~dg-matrix Register count against memory and error, using 6-bit registers. Redis uses the 16,384 row.
Registers (m) | Index bits (p) | Memory | Standard error
1,024 | 10 | 768 bytes | 3.25%
4,096 | 12 | 3 KB | 1.63%
16,384 | 14 | 12 KB | 0.81%
65,536 | 16 | 48 KB | 0.41%
~~~

~~~dg-widget hll A real HyperLogLog running in your browser: add items and watch the registers fill; more registers cost more memory and give a smaller error.
~~~

## Corrections at the edges

The raw formula is biased when the count is small compared with m. With 100 distinct items and 16,384 registers, almost every register is still 0, and the harmonic mean of mostly-empty registers is meaningless. For small counts, below about 2.5 × m (about 40,000 for Redis), the original algorithm switches to **linear counting**: if V registers are still zero, estimate the count as `m · ln(m / V)`. This is the same maths as "how many balls were thrown into m bins if V bins are still empty", and it is very accurate when most bins are empty.

<div class="ex">
<div class="lab">Worked example</div>

**linear counting**

16,384 registers, and after a morning of traffic 15,900 are still zero. Estimate = 16,384 × ln(16,384 / 15,900) = 16,384 × ln(1.0304) ≈ 16,384 × 0.0300 ≈ 491 distinct items. Linear counting is used here because 491 is far below 2.5 × 16,384.

</div>

In the middle range there is still some bias, which early implementations (including early Redis versions) corrected with curves fitted to measurements. Since version 4.0, Redis uses an improved estimator, published in 2017, that works from the histogram of register values and is accurate across small, middle and large counts in one formula, with no switch between methods. At the top end, the original 32-bit algorithm had a correction for hash collisions near 2³² items; with 64-bit hashes that problem effectively disappears.

## Redis's implementation

The numbers for Redis (since 2.8.9):

- **16,384 registers** (14 bits of a 64-bit hash pick the register; 50 bits remain).
- Each register is **6 bits**, enough to store values up to 63, more than the 51 possible here.
- 16,384 × 6 bits = 12,288 bytes, so **12 KB** per dense HLL (12,304 bytes including a small header and an 8-byte cached result).
- Standard error **0.81%** (1.04 / √16,384).
- **Sparse encoding** for small sets: runs of zero registers are compressed, so an HLL with a few hundred items takes a few hundred bytes. It is converted to the dense form once it exceeds `hll-sparse-max-bytes` (3,000 bytes by default).
- Commands: `PFADD key item...` (returns 1 if any register changed), `PFCOUNT key` (estimate), and `PFMERGE dest src...`, which takes the **per-register maximum** across HLLs. `PFCOUNT` on several keys merges them on the fly.
- `PFCOUNT` caches its last result in the header and recomputes only if a register changed, so repeated reads are almost free.

The "PF" prefix honours Philippe Flajolet, who created the algorithm with Éric Fusy, Olivier Gandouet and Frédéric Meunier in 2007.

## HyperLogLog in practice

Google's improved version, **HyperLogLog++**, reduces memory and significantly improves accuracy for an important range of cardinalities, while still running in a single pass and parallelising perfectly. The main changes are a 64-bit hash, empirical bias correction tables for the middle range, and a sparse representation for small sets. Redis's design follows the same ideas.

<div class="ex">
<div class="lab">Worked example</div>

**daily unique visitors**

`PFADD uv:2026-11-09 user:1 user:2 user:1`: registers are updated for user:1 and user:2; the repeat of user:1 changes nothing. After 3 million visits by 1.2 million distinct users, `PFCOUNT uv:2026-11-09` returns about 1,190,000 to 1,210,000 (within one standard error, 0.81%, about two times in three; within 2.4% almost always). The key is 12 KB. A set of 1.2 million 8-byte IDs would be about 10 MB before overhead. For a weekly figure, `PFCOUNT uv:2026-11-03 ... uv:2026-11-09` merges seven days without double-counting anyone.

</div>

## Pitfalls

- **Bad hashes break the maths.** The whole argument assumes uniform random bits. A weak hash on structured IDs (sequential integers) can cluster in some registers. Use a well-mixed 64-bit hash; libraries do this for you.
- **Error is relative, not absolute.** 0.81% of 100 million is about 800,000. Fine for a dashboard, not for billing.
- **No deletes.** A register only remembers a maximum, so you can't remove an item. Use time-bucketed HLLs and stop including old buckets instead.
- **No membership.** HLL can't answer "has user 7 visited?". That's a Bloom filter's job.
- **Intersections are weak.** |A ∩ B| = |A| + |B| − |A ∪ B| works, but the errors of three estimates add up, so a small overlap can come out wildly wrong or even negative.

## In an interview

When you need distinct counts at scale (unique visitors, unique searchers, distinct IPs), say "HyperLogLog: 12 KB per counter, about 1% error, mergeable". If asked how it works, give the three-step version: hash each item; use the first bits to pick a register and record the longest run of leading zeros; combine registers with a harmonic mean, because a run of r zeros is a 1-in-2ʳ event and only shows up after about 2ʳ distinct items.

<div class="take">
<div class="lab">Key takeaway</div>

HyperLogLog hashes each item and records, in each of m registers, the longest run of leading zeros; seeing a run of r zeros suggests about 2ʳ distinct items, and duplicates can never change the sketch. Combining many registers with a harmonic mean gives an error of 1.04/√m: in Redis, 0.81% with 16,384 registers in 12 KB, with linear counting and bias correction for small counts.

</div>

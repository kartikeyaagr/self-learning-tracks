---
id: "12.07"
course: "system-design-101"
module: "12"
slug: "seen-filtering-bloom-vs-exact-sets"
title: "Seen filtering: Bloom vs exact sets"
keyIdeas: ["Seen filtering stops showing users items they've already seen (profiles in a dating app, posts in a feed).", "Exact sets (per-user set of seen IDs in Redis or a DB) are precise but grow large for heavy users.", "A per-user Bloom filter uses far less memory. A false positive means the system thinks you've seen an item you haven't, so that item is skipped. You never see repeats.", "That's usually fine when there are plenty of items. Size the filter so the false-positive rate stays low, and rotate or rebuild it as it fills.", "Combine: exact set for recent items, Bloom filter for long history."]
question: "A dating app must never show a profile twice. Bloom filter or exact set, and what happens on a false positive?"
answer: "A Bloom filter fits, because its errors go only one way: once a profile is added its bits are set and never cleared, so the filter never says \"not seen\" for a swiped profile and it is never shown again. A false positive means an unseen profile is judged \"seen\" and silently skipped, a small loss of one potential match at the designed rate (for example 1% with about 24 KB and 7 hashes for 20,000 swipes), not a violation of the rule, and nobody notices in a large pool. Size the filter for the expected number of swipes, keep the exact swipe log as the durable source of truth, grow or rebuild the filter before it overfills (its error rate climbs quickly past the design size), and use an exact set for users with very small candidate pools or for rules like blocks that must be exact."
sources: [{"title": "Bloom filters by example", "url": "https://llimllib.github.io/bloomfilter-tutorial/", "note": "False positive tradeoffs."}]
---

## The requirement

A dating app shows profiles one at a time and the user swipes left or right. The product rule is strict: a profile the user has already swiped on must **never** appear again. Showing it twice looks broken, and showing someone you rejected is worse.

The scale makes this hard. An active user may swipe through **20,000 profiles** over a few months. The app has **50 million users**. Every time the recommendation service builds a batch of profiles to show (lesson 12.08), it starts with a few hundred **candidates** from nearby, then must remove every candidate this user has already seen, within a latency budget of a few milliseconds for this step.

So the question is a **set membership** question, asked hundreds of times per request: "is profile X in user U's seen set?" There are two ways to answer it: an exact set, or a probabilistic one.

## Option 1: an exact set

Store every seen profile ID per user and check membership directly. Choices:

- A **Redis set** per user (`SADD seen:u9 4471`, `SMISMEMBER seen:u9 ...` for a batch).
- A database table keyed by `(user_id, seen_id)`, which is also the durable swipe log.
- A **sorted set** scored by swipe time, if you want to expire very old swipes.

Exact sets give exact answers, with no surprises. The cost is **memory or I/O**.

- Raw size: 20,000 IDs × 8 bytes = **160 KB** for a heavy user. Redis's set encoding adds overhead: once a set exceeds a small size it becomes a hash table, at roughly 50 to 70 bytes per member in practice, so 20,000 members is over 1 MB.
- Across users: if the average user has seen 2,000 profiles, that is 50 million × 2,000 = **100 billion** seen pairs. At 8 bytes each that is 800 GB of raw IDs; with set overhead, several terabytes of RAM.
- If the sets live in a database instead, each recommendation request needs a batch lookup of 500 IDs against a large index: workable, but tens of milliseconds and heavy load at peak.

## Option 2: a Bloom filter

A **Bloom filter** (lesson 04.13 introduces it) is a compact probabilistic set. It can tell you an element is **definitely not** in the set, or **probably** in it.

### The structure

- A **bit array of m bits**, all initially 0.
- **k independent hash functions**, each mapping an element to one of the m positions.

### Operations

- **Add** an element: compute its k hashes and set those k bits to 1.
- **Test** an element: compute its k hashes and look at those bits. If **any** is 0, the element was **definitely never added**, because adding it would have set all of them. If **all** are 1, it **may** have been added, or those bits may all have been set by other elements.

~~~dg-flow A Bloom filter test: any zero bit proves "never added"; all ones only means "probably added".
Profile 4471 -(k = 7 hashes)-> Bits 12, 88, 301 ... -(any bit is 0)-> *Definitely not seen: show it
Profile 9020 -(k = 7 hashes)-> Bits 5, 140, 777 ... -(all bits are 1)-> Probably seen: skip
~~~

In practice you do not need k truly independent hash functions. The standard trick computes two hashes h1 and h2 with a fast non-cryptographic function (MurmurHash, xxHash) and derives the rest as `h1 + i × h2` for i from 0 to k − 1. Standard Bloom filters **cannot delete** (clearing a bit might remove other elements too; **counting Bloom filters** store small counters instead of bits to allow it) and cannot **list** their contents.

### The maths

For n elements in m bits with k hashes, the false positive rate is approximately:

`p ≈ (1 − e^(−kn/m))^k`

The intuition: after n insertions of k bits each, the chance a given bit is still 0 is about e^(−kn/m). A false positive needs all k probed bits to be 1. The **optimal number of hashes** for a given m and n is `k = (m/n) ln 2`, which makes about half the bits 1. Turning it around, to reach a target rate p you need:

`m = −n ln p / (ln 2)²` bits, which is about **9.6 bits per element for 1%** and 14.4 bits per element for 0.1%.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the filter**

Target: n = 20,000 seen profiles, false positive rate 1%.

m = −20,000 × ln(0.01) / (ln 2)² = 20,000 × 4.605 / 0.4805 ≈ **191,700 bits ≈ 24 KB**.

k = (191,700 / 20,000) × 0.693 ≈ 6.6, so **7 hash functions**.

At 0.1%: about 287,500 bits (36 KB) and 10 hashes. At 0.01%: about 383,400 bits (48 KB) and 13 hashes. Each tenfold improvement in accuracy costs only about 12 KB more.

Compare the exact set: 160 KB of raw IDs, over 1 MB as a Redis set. The Bloom filter is about 7 times smaller than the raw IDs and 40 times smaller than the Redis set.

</div>

~~~dg-bars Memory per heavy user (20,000 seen profiles). The Bloom filter is a fixed-size bit array; exact sets grow with overhead per member.
Redis set | 1200 | about 1.2 MB
Raw 8-byte IDs | 160 | 160 KB
Bloom, 0.01% | 48 | 48 KB
Bloom, 0.1% | 36 | 36 KB
*Bloom, 1% | 24 | 24 KB
~~~

## The asymmetry decides the answer

The important property is not the size, it is the **direction of the errors**. A Bloom filter's errors are **one-sided**.

- **False negatives never happen.** If a profile was added, its k bits are 1 and stay 1 (nothing ever clears them). Every test of it returns "probably seen". So a profile the user has swiped is **never shown again**. The hard product rule is guaranteed.
- **False positives happen at the designed rate.** An **unseen** profile can test as "seen" because other profiles happened to set all of its bits. It is then **skipped**. The user never sees that one profile, which is a small loss (a missed potential match), but the rule "never show twice" is not broken.

Map that onto the product requirement: the requirement is about never showing seen profiles, which is exactly the guarantee a Bloom filter gives. The cost of its errors falls on the side that does not matter much: a few fresh profiles quietly disappear from the candidate pool. **So a Bloom filter fits**, provided the false positive rate is kept low and under control.

When would the false positives matter?

- **Small pools.** In a small town, or with narrow filters (age 34 to 36, within 5 km), a user may have only 300 eligible profiles in total. Losing 1% of them is 3 people, and if the user has seen 280 already, those 3 might be half of what is left. For such users, use an exact set; it is small anyway because the pool is small.
- **A filter that has overfilled.** The rate is only 1% at the designed n. Beyond that it climbs fast.

~~~dg-bars False positive rate of a filter sized for 20,000 items at 1% (24 KB, 7 hashes) as the user keeps swiping. Past the design size it degrades quickly.
10,000 swipes | 0.025 | 0.03%
*20,000 swipes | 1 | 1% as designed
30,000 swipes | 5.8 | 5.8%
40,000 swipes | 15.7 | 16%
60,000 swipes | 43.6 | 44%
~~~

At 60,000 swipes, nearly half of all fresh candidates would be wrongly hidden. A user who swipes a lot would see their recommendations dry up for no visible reason. So you must handle growth.

## Practical design

### Where the filter lives

- **Per-user filter in Redis**, using the RedisBloom module (`BF.ADD`, `BF.MEXISTS` to test a batch of 500 in one call), or as a plain Redis string with `SETBIT` and `GETBIT` if you implement the hashing yourself.
- Or **loaded into the recommendation service's memory** when the user's session starts, and written back periodically. At 24 KB per active user, 2 million concurrently active users is about 48 GB, spread across the fleet.

### Handling growth

- **Scalable Bloom filters** chain several filters of growing size: when the current one reaches its design capacity, a new, larger one (with a tighter target rate) is added. Tests check all of them; adds go to the newest. The overall false positive rate stays bounded. RedisBloom's filters do this automatically when you set an expansion factor.
- **Size per user.** New users start with a small filter (say for 1,000 items, about 1.2 KB); heavy users get bigger ones. There is no reason to give 50 million users a 24 KB filter each when most will never swipe 20,000 times.
- **Rebuild periodically** from the durable swipe log, for example when a user crosses a threshold, so the filter is resized exactly.

### Source of truth

The Bloom filter is a **derived, disposable index**. The exact swipe log, a table of `(user, profile, direction, time)`, is the source of truth. It is needed anyway for matching (a match is two right swipes), analytics and moderation. If a filter is lost or corrupted, rebuild it from the log: for 20,000 rows, a few milliseconds of hashing.

### Write path ordering

When a user swipes, write the swipe log **first**, then add to the filter. If the second step fails, a background job can repair the filter from the log. The opposite order risks a filter entry with no log row, which is harmless here (a profile hidden that was not really seen), but the log-first order keeps the source of truth complete.

~~~dg-seq Building one batch of profiles with a per-user Bloom filter. Seen profiles are always removed; about 1% of unseen ones are removed too.
App > Recommender : next batch for u9
Recommender > Candidate index : 500 nearby profiles matching filters
Candidate index -> Recommender : 500 candidate IDs
Recommender > Redis : BF.MEXISTS seen:u9 (500 IDs)
Redis -> Recommender : 176 probably seen, 324 not seen
Recommender > Ranker : score the 324 fresh profiles
Ranker -> Recommender : top 20 in order
Recommender -> App : 20 profiles to show
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one request walked through**

User u9 has swiped 12,400 profiles. Their filter was sized for 20,000 at 1% (24 KB), so at 12,400 items its actual false positive rate is about 0.1%.

Candidate generation returns 500 profiles near them. Of these, 175 really are profiles u9 has seen (heavy users exhaust their local pool quickly). The filter flags all 175, because there are no false negatives. Of the 325 unseen ones, at about 0.1% it is most likely to flag none or one by mistake; say 1. The remaining 324 go to the ranker. If the filter had been sized right at its limit (1%), it would have flagged about 3.

The wrongly flagged profile is simply never shown to u9. Nobody can tell. An exact set would have avoided that tiny loss at about 7 to 40 times the memory.

</div>

### Belt and braces

If the cost of a mistake were higher (say, showing a profile the user **blocked**), you would not rely on any probabilistic structure. Blocks are few per user, so keep them in an exact set and check them separately. You can also do a final exact check of the 20 profiles actually being sent against the swipe log: 20 point lookups instead of 500.

~~~dg-compare Choosing the seen filter. The Bloom filter's errors fall on the harmless side of the requirement.
[Exact set]
Candidates -> SMISMEMBER -> Exact answer
+ No errors in either direction
+ Supports delete and listing
- 160 KB to over 1 MB per heavy user
- Terabytes of RAM across all users
[*Bloom filter]
Candidates -> BF.MEXISTS -> Seen or maybe seen
+ About 24 KB per heavy user at 1%
+ Never shows a seen profile
- Hides about 1% of fresh profiles
- Needs resizing as users swipe more
~~~

## Common mistakes

- **Getting the direction backwards.** Some designs store "profiles not yet seen" in a Bloom filter. Then a false positive means "probably not seen" for a seen profile, and it gets shown again: the exact failure the requirement forbids. Always store the set whose false positives are harmless.
- **Fixed-size filters forever.** A filter sized at sign-up silently degrades as the user swipes. Monitor fill ratio (fraction of bits set) and resize.
- **Using it for things that must be exact.** Blocks, reports and legal removals need exact sets.
- **No source of truth.** If the only record of what a user has seen is the Bloom filter, you cannot rebuild, resize or debug it.
- **Cryptographic hashes.** SHA-256 is ten times slower than needed. Use MurmurHash or xxHash.

## In an interview

Name the trade-off clearly: exact sets cost memory proportional to swipes with overhead per entry; a Bloom filter costs about 10 bits per item at 1% and has one-sided errors. Then say the key sentence: "there are no false negatives, so a seen profile is never shown; a false positive only hides an unseen profile, which is a small, tunable loss". Size it with the formula, mention growth (scalable filters, rebuild from the swipe log), and give the small-pool exception. That shows you understand both the data structure and the product.

<div class="take">
<div class="lab">Key takeaway</div>

a Bloom filter answers "definitely not seen" or "maybe seen" with no false negatives, so seen profiles are never shown again; false positives only hide some unseen profiles, a small, tunable loss. Size it with m = −n ln p / (ln 2)² and k = (m/n) ln 2 (about 24 KB and 7 hashes for 20,000 items at 1%), keep the exact swipe log as the source of truth, grow or rebuild the filter before it overfills, and use exact sets for small pools and for anything that must be exact, like blocks.

</div>

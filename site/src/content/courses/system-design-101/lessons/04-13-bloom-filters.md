---
id: "04.13"
course: "system-design-101"
module: "04"
slug: "bloom-filters"
title: "Bloom filters"
keyIdeas: ["A Bloom filter is a bit array plus k hash functions. To add an item, set the k bits it hashes to. To check, see if all k bits are set.", "If any bit is 0, the item is definitely not present. If all are 1, it's probably present: those bits might have been set by other items (false positive).", "No false negatives, tunable false-positive rate, very small memory (about 10 bits per item for 1% false positives).", "You can't delete items from a basic Bloom filter (counting Bloom filters allow it).", "Uses: skip disk reads for missing keys (LSM trees), avoid recommending seen items, check weak passwords, cache admission."]
question: "A Bloom filter says \"maybe present\" or \"definitely not\". Why can't it say \"definitely present\"?"
answer: "Because the bits are shared between all items and never cleared. When you check an item, its k bits may all be 1 because other items set them, not because this item was added; the filter stores no record of which item set which bit, so it cannot distinguish the two cases and can only say \"maybe\" (a false positive, at a rate set by bits per item and the number of hash functions). The reverse is certain: if any of the k bits is 0, the item was definitely never added, because adding it would have set that bit to 1 and bits are never reset. The same sharing is why a standard Bloom filter cannot delete items."
sources: [{"title": "Bloom filters by example", "url": "https://llimllib.github.io/bloomfilter-tutorial/", "note": "Interactive explanation."}, {"title": "Jason Davies: Bloom filter demo", "url": "https://www.jasondavies.com/bloomfilter/", "note": "Visual demo."}]
tags: ["Code"]
---

## The problem a Bloom filter solves

Many systems repeatedly ask "is this item in the set?" where the honest answer is usually **no**, and finding out is expensive. A database checks whether a key might be in a data file before reading the file from disk. A web crawler checks whether it has already visited a URL among billions. A URL shortener checks whether a requested code exists before querying the database. Storing the whole set in memory would answer exactly, but a billion URLs at about 100 bytes each is 100 GB.

A **Bloom filter**, invented by Burton Bloom in 1970, answers set membership in a tiny fraction of that memory, at the cost of sometimes being wrong in **one direction only**. It can say "**definitely not in the set**" or "**maybe in the set**". It never says "definitely in the set".

## The structure

A Bloom filter has two parts:

- An array of **m bits**, all starting at 0.
- **k hash functions**, each mapping an item to one bit position between 0 and m − 1.

**Adding** an item: hash it with all k functions and set those k bits to 1. Bits that are already 1 stay 1.

**Checking** an item: hash it with the same k functions and look at those k bits.

- If **any** bit is 0, the item was **definitely never added**. Adding it would have set that bit to 1, and bits are never cleared.
- If **all** k bits are 1, the item is **probably** in the set.

Nothing about the items themselves is stored: no keys, no pointers, only bits. That is where the memory saving comes from.

~~~dg-seq Adding two items to a 16-bit filter with three hash functions, then checking two items that were never added.
App > Filter : add "cat"
Filter > Filter : h1, h2, h3 give 2, 7, 11: set to 1
App > Filter : add "owl"
Filter > Filter : 3, 9, 14: set to 1
App > Filter : contains "dog"? bits 3, 7, 10
Filter -> App : bit 10 is 0: definitely not
App > Filter : contains "fox"? bits 2, 9, 14
Filter -> App : all 1: maybe, a false positive
~~~

## Why it cannot say "definitely present"

Bits are **shared**. Every item sets bits in the same array, and those bits are never cleared. An item you never added may hash to k positions that **other** items happened to set. In the trace above, "fox" was never added, but bit 2 was set by "cat" and bits 9 and 14 by "owl". The filter has no record of which item set which bit, so it cannot tell whether those 1s came from "fox" or from others. All it can honestly say is "maybe". That wrong "maybe" is a **false positive**.

**False negatives are impossible**: if an item was added, its k bits were set to 1 at that moment, and nothing ever sets them back to 0. So every check for an added item finds all 1s.

The same fact explains why a standard Bloom filter **cannot delete** items. To remove "cat" you would clear bits 2, 7 and 11, but bit 7 might also belong to another item, which would then start failing its checks: a false negative, breaking the one guarantee the filter offers. Variants fix this at a cost:

- A **counting Bloom filter** stores a small counter (typically 4 bits) per position instead of one bit. Adding increments, deleting decrements. It uses about 4 times the memory.
- A **cuckoo filter** stores short fingerprints of items in a hash table, supports deletion, and is often smaller than a Bloom filter at low error rates. Redis's probabilistic module offers both Bloom and cuckoo filters.

~~~dg-matrix Exact and approximate ways to answer "is X in the set?", for one billion items.
Structure | Memory for 1 billion items | False positives | False negatives | Delete
Hash set of the items | about 100 GB | no | no | yes
Bloom filter at 1% | about 1.2 GB | yes, 1% | no | no
Counting Bloom filter | about 4.8 GB | yes | no | yes
Cuckoo filter at 1% | about 1.1 GB | yes | no | yes
~~~

## Sizing it

Three numbers describe a Bloom filter: n items, m bits and k hash functions. The false positive rate after inserting n items is approximately:

- p ≈ (1 − e^(−kn/m))^k

The intuition: after n insertions of k bits each, the chance that a particular bit is still 0 is about e^(−kn/m). A false positive needs all k checked bits to be 1. For a given m/n, too few hash functions leave too few bits to check; too many fill the array with 1s. The **best number of hash functions** is:

- k = (m/n) × ln 2, which is about 0.69 × bits per item.

At that optimum, the bits needed per item for a target error rate p is about 1.44 × log2(1/p). Useful rules of thumb:

- About **10 bits per item with 7 hash functions** gives roughly a **1%** false positive rate.
- Each extra **4.8 bits per item** divides the error rate by 10.
- **Halving** the error rate costs only about **1.44 extra bits** per item.

~~~dg-bars Bits needed per item for a target false positive rate, with the best number of hash functions.
10% false positives | 4.8 | 4.8 bits, k = 3
1% | 9.6 | 9.6 bits, k = 7
*0.1% | 14.4 | 14.4 bits, k = 10
0.01% | 19.2 | 19.2 bits, k = 13
~~~

So a billion items at 1% needs about 9.6 billion bits, **1.2 GB**, compared with around 100 GB to store the items themselves. Try changing the numbers below and watch how the false positive rate responds.

~~~dg-widget bloom How the false positive rate depends on bits per item and the number of hash functions, plus a tiny filter to add items to and test.
~~~

### The sizing process

1. Estimate **n**, the number of items, with headroom for growth. A Bloom filter cannot grow: if you add many more items than planned, the error rate climbs steeply. At double the planned n, a 1% filter's error rate rises above 10%.
2. Choose the target false positive rate **p** from what a false positive costs (an extra disk read? an extra database query?).
3. Compute **m** = −n × ln(p) / (ln 2)^2 and **k** = (m/n) × ln 2, rounded.
4. If the data keeps growing, plan to rebuild the filter periodically, or use a **scalable Bloom filter** that adds new, larger filters as it fills.

### Hashing

Use fast non-cryptographic hashes such as MurmurHash, xxHash or FNV, not SHA-1 or MD5, which are several times slower and add nothing here. You do not need k independent hash functions either: compute two hashes h1 and h2 and derive the rest as h1 + i × h2 (the **double hashing** trick), with almost no loss in accuracy. Insert and lookup are both O(k), independent of how many items are stored.

## Where they are used

The pattern is always the same: **a cheap, in-memory check in front of an expensive operation**, where a "no" saves the expensive work, and an occasional false "maybe" only costs one wasted lookup that gives the real answer.

~~~dg-flow A point read in an LSM-tree database: a Bloom filter per data file lets the read skip files that cannot contain the key.
[Read key 42] Memtable -(miss)-> Filter SSTable 1 -(definitely not)-> Filter SSTable 2 -(maybe)-> *Read SSTable 2 from disk
~~~

- **LSM-tree databases** (RocksDB, Cassandra, ScyllaDB, HBase) keep a Bloom filter for each immutable data file. A read for a key checks each file's filter first and skips files that definitely lack the key, so most point reads touch at most one file on disk (lesson 04.03 and Module 7).
- **Web crawlers**: "have I already crawled this URL?" across billions of URLs. A false positive means one page is not crawled, which is acceptable.
- **Caches and CDNs**: avoid caching **one-hit wonders**, objects requested only once. Only cache an object the second time it is requested, using a Bloom filter to remember first requests. Akamai found that a large share of objects were requested only once, and this saved a great deal of cache space.
- **Avoiding lookups for missing keys**: reject requests for IDs that do not exist without touching the database (a defence against cache penetration attacks that request random keys).
- **Safe browsing and availability checks**: the Chrome browser once used a local Bloom filter of malicious URLs, asking the server only on a "maybe". A sign-up form can rule out most usernames as "definitely available" locally before confirming with the server.
- **Distributed joins**: send a Bloom filter of one side's join keys to the other machines so they only send rows that might match.

<div class="ex">
<div class="lab">Worked example</div>

**avoiding database lookups for missing short codes**

A URL shortener has 500 million valid codes and receives 20,000 requests per second, of which 30% are for codes that do not exist (typos, scanners, bots), about 6,000 per second.

A Bloom filter of all valid codes at 10 bits each needs 5 billion bits, about 600 MB of RAM per app server, with k = 7 and a false positive rate of about 0.8%.

Missing codes: 99.2% are rejected in RAM with a 404 in microseconds. The other 0.8%, about 50 per second, fall through to a database lookup that returns 404. Valid codes always pass the filter (no false negatives) and are looked up as normal.

Net effect: the database sees about 14,050 lookups per second instead of 20,000. New codes are added to the filter as they are created; the filter is rebuilt nightly from the database to stay sized correctly.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**computing a filter by hand**

You expect n = 10 million items and accept p = 0.1%.

m = −n × ln(0.001) / (ln 2)^2 = 10,000,000 × 6.91 / 0.48 ≈ 143.8 million bits, about 18 MB. That is 14.4 bits per item.

k = 14.4 × 0.693 ≈ 10 hash functions.

If the set unexpectedly grows to 20 million items in the same 18 MB, bits per item fall to 7.2 and the false positive rate rises to about 6% with k still 10 (about 3% even with the best k for that load, which is 5). This is why the estimate of n needs headroom.

</div>

## Common mistakes

- **Treating "maybe" as "yes"**, skipping the authoritative check after a positive.
- **Undersizing n**, so the filter fills up and the error rate explodes.
- **Trying to delete from a standard Bloom filter.**
- **Using slow cryptographic hashes.**
- **Using a Bloom filter where false positives are unacceptable without a fallback**, such as deciding whether a payment was already processed.

## In an interview

Reach for a Bloom filter when you see "most lookups are for things that do not exist" or "check against a huge set cheaply". State the guarantee precisely (no false negatives, tunable false positives), give the sizing rule (about 10 bits per item for 1%), and say what happens on a "maybe": fall through to the real lookup.

<div class="take">
<div class="lab">Key takeaway</div>

a Bloom filter sets k bits per item in one shared bit array. A 0 bit proves the item was never added; all 1s only suggests it was, because other items may have set those bits. About 10 bits per item gives a 1% false positive rate, making it ideal for skipping expensive lookups that would mostly find nothing.

</div>

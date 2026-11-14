---
id: "09.03"
course: "system-design-101"
module: "09"
slug: "boolean-and-tiered-search"
title: "Boolean and tiered search"
keyIdeas: ["Boolean retrieval answers AND/OR/NOT queries by merging postings lists. \"cat AND sat\" = intersection of [1,2] and [1,3] = [1].", "Intersect the shortest list first: the result can't be longer than the shortest list, so you do less work. Skip pointers jump ahead in long lists.", "Boolean retrieval doesn't rank: documents match or they don't.", "**Tiered indexes**: put high-quality or high-scoring documents in tier 1 and search it first; only go to lower tiers if tier 1 doesn't return enough results."]
question: "Why intersect the shortest posting list first?"
answer: "Because the intersection of several lists can never be longer than the shortest one. Starting with the rarest term (lowest document frequency, known from the dictionary before reading postings) makes the intermediate result small from the first step, so every later intersection only checks a few candidates against the longer lists, which can be jumped through with skip pointers or galloping search instead of read in full. Starting with a common term would walk millions of postings only to throw almost all of them away, for exactly the same answer."
sources: [{"title": "IR book ch.1 and 7: Boolean queries, tiered indexes", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "Intersecting postings; tiered indexes in 7.2.1."}]
---

## Boolean retrieval

The simplest way to use an inverted index is **Boolean retrieval**: a query is a logical expression over terms, and a document either matches or it does not. There is no score, no "better" or "worse" match. A classic example over Shakespeare's plays is `Brutus AND Caesar AND NOT Calpurnia`.

Each Boolean operator becomes an operation on postings lists (the sorted lists of document IDs for each term, lesson 09.02):

- **AND** is the **intersection** of postings lists: documents in both.
- **OR** is the **union**: documents in either.
- **NOT** is the **difference**: documents in one list but not the other.

Boolean search is still everywhere, even in engines that rank. In Elasticsearch, a `bool` query has `must` (AND, and contributes to score), `filter` (AND, no scoring, cacheable), `should` (OR, adds to score) and `must_not` (NOT). Filters such as `in_stock = true AND category = shoes` are pure Boolean, evaluated before or alongside ranking.

## Intersecting two lists

Because postings are sorted by docID, two lists can be intersected in one pass, like the merge step of merge sort. Keep a pointer into each list:

1. If the two current IDs are equal, output the ID and advance both pointers.
2. Otherwise, advance the pointer that holds the smaller ID (that ID can't be in the other list, since everything after the larger ID's position is even larger).
3. Stop when either list runs out.

This takes **O(x + y)** time for lists of length x and y, and needs no extra memory beyond the output.

```
def intersect(p1, p2):
    i = j = 0; out = []
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]: out.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]: i += 1
        else: j += 1
    return out
```

~~~dg-timeline Tracing the merge of cat [1, 2, 7, 9] and sat [1, 3, 9]: each step compares the two pointers and advances the smaller.
Step 1 | cat 1 vs sat 1: equal, *output 1
Step 2 | cat 2 vs sat 3: advance cat
Step 3 | cat 7 vs sat 3: advance sat
Step 4 | cat 7 vs sat 9: advance cat
Step 5 | cat 9 vs sat 9: equal, *output 9
Step 6 | cat list ends: result [1, 9]
~~~

Union works the same way, outputting every ID once. `A AND NOT B` walks both lists and outputs IDs from A that aren't matched in B. Note that a bare `NOT B` would need the list of all documents, which is huge, so engines require NOT to be combined with a positive clause.

## Skip pointers and galloping

When one list is much shorter than the other, the linear merge wastes time stepping through the long list one posting at a time. Two tricks fix this:

- **Skip pointers**: every √n entries or so, the long list stores a pointer that jumps ahead. When the short list's current ID is 50,000 and the long list is at 1,200, the merge can follow skip pointers past whole runs of IDs instead of reading each one. Lucene stores skip data in its postings so it can jump within a block-compressed list.
- **Galloping (exponential) search**: from the current position, probe 1, 2, 4, 8, ... entries ahead until you pass the target, then binary search back. This costs about O(x log(y / x)) for a short list of x and long list of y, close to optimal.

With either, intersecting a list of 300 with a list of 10 million costs a few thousand operations, not 10 million.

## Why the shortest list goes first

For a query with several ANDed terms, **process terms in order of increasing document frequency**, starting with the rarest. The reason is simple: the result of an intersection can never be longer than the shorter of its two inputs. So if you start with the rarest term, the intermediate result is small from the first step, and every later step only checks a few candidates against the longer lists, which skip pointers let you jump through.

The engine knows each term's document frequency (df) from the dictionary for free, before reading any postings, so it can plan the order up front. That's a tiny **query planner**, the same idea as a database choosing the most selective index first.

~~~dg-bars Postings scanned for "the AND quick AND zebra" in two evaluation orders. Starting with the rarest term avoids reading the huge list.
the first, linear | 10050300 | ~10 million postings
zebra first, linear | 10050300 | still ~10 million if no skipping
*zebra first, with skips | 6000 | ~6,000 postings checked
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why order matters**

Say "the" appears in 10,000,000 documents, "quick" in 50,000 and "zebra" in 300.

Bad order: start with "the AND quick". The merge walks 10,050,000 postings to produce up to 50,000 results, then intersects those with "zebra" to keep at most 300. Most of the work was thrown away.

Good order: start with "zebra AND quick". With galloping through "quick", that takes a few thousand steps and yields at most 300 IDs (say 40). Then check those 40 against "the": with skip pointers, each is a short jump, so maybe 40 × 20 = 800 steps. Total: a few thousand operations instead of ten million, a speed-up of over a thousand times for the same answer.

</div>

For OR queries the order doesn't matter for correctness, but the size does: a union is at least as long as the longest list. That's why common words in OR queries are expensive, and why ranked engines use tricks such as **WAND** and **block-max WAND**: they keep track of the score threshold needed to enter the current top 10 and skip over documents that can't possibly beat it, using the maximum possible score contribution of each term.

## The limit of Boolean search

Boolean retrieval has no notion of "better". Every matching document is equally good, and results usually come back in docID order, which means nothing to the user. In practice queries either return far too many results to read ("feast") or none at all ("famine"): `shoes AND red` returns 40,000 products; `shoes AND red AND waterproof AND size-9 AND leather` returns zero.

That is fine for expert users such as patent examiners, librarians and lawyers who write precise queries, want every match and can refine iteratively. Legal research systems still offer Boolean search for exactly this reason. Most users, though, type a few words and want the best results first. Ranking (lessons 09.04 and 09.05) fixes that. But ranking has a cost: scoring every matching document is expensive when there are millions of matches. Tiered indexes are one way to keep it fast.

## Tiered indexes

A **tiered index** avoids scoring every matching document by splitting each term's postings into tiers by importance. For example:

- Tier 1 holds documents where the term appears at least 20 times (or, more usefully, documents with high quality scores).
- Tier 2 holds those where it appears at least 10 times.
- Tier 3 holds the rest.

A query runs on **tier 1 only** first. If tier 1 yields at least K results (K is the number you want to show, say 10), stop and rank those. If not, fall back to tier 2 and merge, and so on. Most popular queries are satisfied by the top tier, which is a small fraction of the index.

~~~dg-flow A tiered index: search the high-quality tier first and only fall back when it can't fill the page.
Query -> *Tier 1: best docs -(fewer than K results)-> Tier 2: good docs -(still fewer)-> Tier 3: the rest
~~~

Real web engines apply the same idea with **document quality** rather than term frequency. A small tier of high-quality, popular pages (perhaps a few percent of the index, often kept in memory) is searched first. Only rare queries that can't be answered from it go to the large, slower tiers of the long tail. Splitting by quality rather than per term also means one tier is a whole separate index that can live on faster hardware.

~~~dg-compare Searching everything versus searching in tiers. Tiers are faster for common queries but can miss a great document that landed in a low tier.
[Single index]
Query -> All postings -> Score all matches
+ Never misses a match
- Scores millions for common queries
[*Tiered index]
Query -> Tier 1 -(enough?)-> Stop or tier 2
+ Most queries touch a small tier
+ Hot tier fits in memory
- A relevant doc in tier 3 can be missed
~~~

### Related tricks

All of these trade a little accuracy for speed, and all rely on ordering or pruning postings:

- **Index elimination**: only consider query terms whose idf is above a threshold (rare, informative terms), or only documents that contain most of the query terms. "the best pizza in leeds" is effectively evaluated on "pizza" and "leeds".
- **Champion lists**: for each term, precompute the top r documents by weight and search only those first (lesson 09.06).
- **Impact-ordered postings**: sort each postings list by the term's weight in the document instead of by docID, so the best postings come first and scoring can stop early. You lose cheap linear merges, so this is used with score-at-a-time algorithms.

<div class="ex">
<div class="lab">Worked example</div>

**a tiered product index**

An e-commerce site has 50 million products, but 2 million account for 95% of sales. It builds tier 1 from those 2 million (in memory, on fast nodes) and tier 2 from everything else. The query "iphone case" finds 3,000 tier-1 matches: plenty to fill a page of 48, so tier 2 is never touched, and the query takes 15 ms. The query "replacement hinge for 2009 dishwasher model x" finds only 2 tier-1 matches, so the engine falls back to tier 2, finds 40 more, and answers in 90 ms. Average latency drops sharply because most traffic is head queries.

</div>

## Common mistakes

- Putting common terms first in AND evaluation, or letting user order decide.
- Implementing `NOT` without a positive clause, forcing a scan of all documents.
- Forgetting that filters (Boolean, unscored) can be cached as bitsets and reused across queries, which is one of the cheapest speed-ups available.
- Using tiers without monitoring how often queries fall through to lower tiers. If most do, the tiers are wrong.

## In an interview

When explaining query execution, say: "postings are sorted by docID, so AND is a linear merge; evaluate terms rarest first using df from the dictionary, and use skip pointers so long lists are jumped through, not read." If the interviewer pushes on latency, mention tiers or early termination (WAND) for ranked queries.

<div class="take">
<div class="lab">Key takeaway</div>

Boolean queries are evaluated by merging sorted postings lists in linear time, starting with the rarest term so intermediate results stay small, and skip pointers let the merge jump through long lists. Boolean search doesn't rank, so tiered indexes put the best documents in a first tier and only fall back to lower tiers when the first can't supply enough results.

</div>

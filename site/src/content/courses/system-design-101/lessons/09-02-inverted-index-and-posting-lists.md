---
id: "09.02"
course: "system-design-101"
module: "09"
slug: "inverted-index-and-posting-lists"
title: "Inverted index and posting lists"
keyIdeas: ["An inverted index maps each term to a postings list: the IDs of documents containing it, often with positions and frequencies.", "Example: docs 1 \"cat sat\", 2 \"cat ran\", 3 \"dog sat\". Index: cat → [1,2], sat → [1,3], ran → [2], dog → [3].", "Postings lists are sorted by doc ID so they can be intersected or merged quickly and compressed (delta encoding + variable byte).", "Building: tokenize each doc, emit (term, docID) pairs, sort, group. At scale, build segments in memory and merge them, much like an LSM tree."]
question: "Build the inverted index by hand for: \"cat sat\", \"cat ran\", \"dog sat\"."
answer: "Tokenise the documents: doc 1 gives cat, sat; doc 2 gives cat, ran; doc 3 gives dog, sat. Emit the pairs (cat,1) (sat,1) (cat,2) (ran,2) (dog,3) (sat,3), sort them by term then docID to get (cat,1) (cat,2) (dog,3) (ran,2) (sat,1) (sat,3), and group by term. The index is cat → [1, 2], dog → [3], ran → [2], sat → [1, 3], with document frequencies 2, 1, 1 and 2. Each postings list is sorted by docID, so a query like cat AND sat is answered by merging [1, 2] and [1, 3] to get [1]."
sources: [{"title": "IR book ch.1-2: Boolean retrieval, postings", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "Inverted index construction."}]
tags: ["Code"]
---

## The idea: invert the document

A document naturally lists its words in order: "the cat sat on the mat". That's the **forward** direction: document to words. To answer "which documents contain *cat*?" from forward data, you'd read every document. An **inverted index** flips the relationship: for every term, it lists the documents that contain it. Now the question is one lookup.

Three words to learn now, because the rest of the module uses them:

- The **dictionary** (or vocabulary, or lexicon) is the set of all distinct terms in the collection.
- Each term's list of document IDs is its **postings list**.
- Each entry in a postings list is a **posting**: at minimum a document ID, often with extra data such as how many times the term appears.

A **document ID** (docID) is a small integer the engine assigns to each document, usually in the order documents are added. Integers are cheaper to store, sort and compare than URLs or product SKUs, and the engine keeps a separate mapping from docID back to the real document.

## Building it by hand

Take three tiny documents:

| Doc ID | Text |
| --- | --- |
| 1 | cat sat |
| 2 | cat ran |
| 3 | dog sat |

Step 1: **tokenise** each document into tokens. Doc 1 gives `cat`, `sat`. Doc 2 gives `cat`, `ran`. Doc 3 gives `dog`, `sat`.

Step 2: **normalise** tokens into terms (lower-case, maybe stem). These are already lower-case single words, so nothing changes.

Step 3: **emit a pair** `(term, docID)` for every token, in document order:
`(cat,1) (sat,1) (cat,2) (ran,2) (dog,3) (sat,3)`

Step 4: **sort** the pairs by term, then by docID:
`(cat,1) (cat,2) (dog,3) (ran,2) (sat,1) (sat,3)`

Step 5: **group** by term, merging duplicates within a document, and record each term's **document frequency** (df), the number of documents in its list:

~~~dg-tree The finished inverted index for the three documents: a sorted dictionary of terms, each pointing at a sorted postings list.
Dictionary
  cat | df 2
    *postings: 1, 2
  dog | df 1
    *postings: 3
  ran | df 1
    *postings: 2
  sat | df 2
    *postings: 1, 3
~~~

Now "which documents contain sat?" is one dictionary lookup that returns `[1, 3]`. "Which contain cat AND sat?" is the overlap of `[1, 2]` and `[1, 3]`, which is `[1]` (lesson 09.03 covers how to compute that efficiently).

The same steps in code:

```
docs = {1: "cat sat", 2: "cat ran", 3: "dog sat"}
pairs = []
for doc_id, text in docs.items():
    for term in text.lower().split():
        pairs.append((term, doc_id))
pairs.sort()                      # by term, then doc id
index = {}
for term, doc_id in pairs:
    plist = index.setdefault(term, [])
    if not plist or plist[-1] != doc_id:
        plist.append(doc_id)      # sorted, no duplicates
print(index)  # {'cat': [1, 2], 'dog': [3], 'ran': [2], 'sat': [1, 3]}
```

Why sort rather than just append to a hash map as you go? For a few documents either works. For a collection that doesn't fit in memory, sorting is the operation we know how to do efficiently on disk (external merge sort), and it produces postings that are already in docID order, which every later operation depends on.

## What a posting can hold

A bare docID answers "does this document contain the term?". Ranking and phrase search need more:

- **Term frequency** (tf): how many times the term appears in that document. Needed for scoring (lessons 09.04 and 09.05).
- **Positions**: the word offsets of each occurrence. Needed for **phrase queries** ("new york" must have "york" at position p + 1 when "new" is at p) and **proximity** scoring (terms close together count more).
- **Offsets** (character start and end) to highlight matches in snippets without re-analysing the text.

A richer index for "the cat sat on the cat mat" (doc 7) would store `cat → (7, tf 2, positions [1, 5])`. Positions often make the index two to four times bigger, so engines let you turn them off per field.

~~~dg-compare How much each posting stores, and what queries that enables.
[DocID only]
cat -> 1, 2, 7
+ Smallest index
- Only yes or no matching
[With tf]
cat -> 1:1, 2:1, 7:2
+ Enables ranking
- A little larger
[*With positions]
cat -> 7:2 at 1, 5
+ Phrases and proximity
- Two to four times larger
~~~

## Where the pieces live

An inverted index has two parts of very different size.

- The **dictionary** is small: maybe a few million distinct terms, each with its df and a pointer to its postings. It's kept in memory, or mostly in memory, in a structure that supports fast lookup and prefix search. Lucene uses a **finite state transducer** (FST), a compressed automaton that shares prefixes and suffixes between terms, so tens of millions of terms fit in a few hundred megabytes.
- The **postings** are large, typically most of the index, and live on disk (and in the operating system's page cache when hot).

### Why postings are sorted

Postings are kept **sorted by docID**. That one decision makes:

- Intersecting two lists a single linear merge (lesson 09.03).
- Merging partial indexes cheap.
- Compression very effective, as follows.

### Gap encoding and compression

Because lists are sorted, store the **gaps** between successive IDs instead of the IDs themselves. The list `[1000, 1002, 1005, 1100]` becomes `[1000, 2, 3, 95]`. Common terms have many postings close together, so their gaps are tiny. Small numbers need few bits, so encode each gap with a variable-length code:

- **Variable-byte** (varint): 7 bits of each byte carry data and 1 bit says "more bytes follow". Gaps under 128 take one byte instead of four.
- **Bit packing** in blocks (as Lucene does with blocks of 128): find the largest gap in the block, and store all 128 gaps with exactly that many bits each. Very fast to decode with SIMD instructions.

<div class="ex">
<div class="lab">Worked example</div>

**how much compression buys**

A term appears in 10 million of a collection's 100 million documents. Stored as 4-byte integers, its postings take 40 MB. On average, the gap between postings is 100 million / 10 million = 10, which fits in 4 bits. With bit packing at about 4 to 5 bits per posting, the list takes about 5 to 6 MB, an eight-fold saving. Smaller lists also mean fewer disk reads and more of the index in the page cache, so compression makes queries faster, not slower.

</div>

## Building at real scale

The hand-built method sorts all pairs in memory. A billion-document index produces hundreds of billions of pairs; that doesn't fit. Two approaches:

- **Blocked sort-based indexing**: process documents in blocks that fit in memory, build a sorted partial index for each block, write it to disk, then merge all the partial indexes in one pass (like the merge phase of an external sort).
- **MapReduce-style** distributed indexing: mappers emit `(term, docID)` pairs, the shuffle groups pairs by term across machines, and reducers write postings lists. This is how web-scale indexes were built in batch.

### Lucene's segments

Engines that index continuously, like Lucene (and therefore Elasticsearch, OpenSearch and Solr), use a different trick:

1. New documents go into an **in-memory buffer** and the transaction log.
2. Periodically (Elasticsearch's default **refresh** is every 1 second), the buffer is written out as a new **segment**: a small, complete, immutable inverted index. Once a segment exists, its documents are searchable.
3. A query searches every segment and combines the results.
4. In the background, a **merge policy** combines small segments into bigger ones, so the number of segments stays manageable.
5. **Deletes are soft**: a deleted document is marked in a per-segment bitmap (the "live docs") and filtered out of results. It's physically removed only when its segment is merged. An **update** is a delete plus a re-insert of the whole document.

~~~dg-timeline One shard's segments over a few seconds: small segments appear on each refresh and are merged in the background.
[Writes] 0.0 s | docs 1-500 buffered in memory
[Segments] 1.0 s | refresh: segment A (500 docs) searchable
[Writes] 1.4 s | doc 42 deleted, marked in A's bitmap
[Segments] 2.0 s | refresh: segment B (480 docs)
[Segments] 3.0 s | refresh: segment C (510 docs)
[Merge] 3.5 s | *A, B, C merged into D; doc 42 dropped
~~~

This is the same idea as the **LSM tree** from Module 7: buffer writes in memory, flush immutable sorted files, compact in the background. Immutability has big benefits: segments never need locks for reading, they cache perfectly in the page cache, and they compress well. The cost is that updates are expensive (rewrite the whole document) and merges use disk I/O.

<div class="ex">
<div class="lab">Worked example</div>

**an update to a product**

Product 9001's price changes. Elasticsearch doesn't edit the old posting lists; it marks the old version of doc 9001 deleted in segment A's bitmap, adds the new version to the in-memory buffer, and within a second the next refresh writes it into a new segment. Until that refresh, queries still see the old version; after it, they see only the new one, never both, because the delete and the new segment become visible together and the live-docs bitmap filters the old copy. Later, when segment A is merged, the old version's postings disappear for good. If a product is updated 100 times a day, 99 dead copies pile up until merges clean them, which is why frequently updated indexes need tuning of merge settings.

</div>

## Common mistakes

- Indexing with positions on every field "just in case", doubling index size for fields nobody phrase-searches.
- Treating updates as cheap. In an inverted index an update rewrites the whole document; high-churn fields (view counts) are better kept outside the text index or in a separate store.
- Forgetting that df and other statistics include deleted-but-not-yet-merged documents until merges happen, which can skew scores slightly.

## In an interview

If asked "how does full-text search work?", build the index for two or three tiny documents on the board in under a minute: tokens, pairs, sort, group, postings with df. Then mention positions for phrases, gap compression, and immutable segments with background merges and soft deletes. That covers what most interviewers want to hear.

<div class="take">
<div class="lab">Key takeaway</div>

an inverted index maps each term to a sorted postings list of document IDs (plus term frequencies and positions), built by emitting (term, docID) pairs, sorting and grouping. The dictionary stays in memory, compressed postings live on disk, and at scale the index is built as immutable segments that are merged later, with deletes marked in a bitmap.

</div>

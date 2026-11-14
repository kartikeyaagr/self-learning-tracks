---
id: "11.07"
course: "system-design-101"
module: "11"
slug: "hashtag-extraction-and-tag-store"
title: "Hashtag extraction and tag store"
keyIdeas: ["Extract hashtags when a post is created (a regex on `#word`, normalized to lowercase), not at query time.", "Store either as a tags array on the post with a GIN index, or in a separate post_tags(tag, post_id, created_at) table.", "A separate table indexed by (tag, created_at) makes \"latest posts for #x\" a single range scan and shards nicely by tag.", "Very popular tags become hot keys; split by time bucket.", "Trending tags come from counting tag usage in recent time windows (see top-k)."]
question: "Tags in an array column vs a separate tag table: tradeoffs for \"all posts with #x\"?"
answer: "An array column with a GIN index keeps one row per post, is simple, and handles multi-tag queries with @> or &&, but GIN's posting lists are ordered by row, not time, so getting the newest 20 posts for a popular tag means fetching and sorting millions of matches; GIN writes are also relatively costly, and the index can't be sharded by tag because tags live with their posts. A separate tag table keyed by (tag, created_at, post_id) costs an extra row per tag per post, but \"all posts with #x, newest first\" becomes a 20-row range scan with keyset pagination at any depth, per-tag counts are easy to maintain, and the table can be partitioned by tag (plus a time bucket for very hot tags). Use the array for small datasets and multi-tag filters, and the tag table at scale."
sources: [{"title": "PostgreSQL docs: GIN indexes", "url": "https://www.postgresql.org/docs/current/gin.html", "note": "Indexing arrays such as tags."}]
---

## The feature

Posts contain hashtags: "Great run today #marathon #NYC". The system must extract them when a post is saved and then answer:

- "All posts with #x", newest first, paginated (the tag page).
- Posts with several tags at once ("#marathon and #nyc"), sometimes.
- How many posts use #x, and which tags are trending right now.

The question asks you to choose between two storage designs: keep the tags as an **array column** on the posts table, or keep them in a **separate tag table** with one row per (tag, post). Both work at small scale. They behave very differently when a tag is popular, which is exactly when the feature matters most.

## Extraction: a small parser at write time

**Extraction** happens once, when a post is created or edited, never at query time. The steps:

1. **Find candidates** with a regex such as `#(\w+)`, using Unicode word characters so `#café`, `#東京` and `#ราชการ` work. Exclude matches inside URLs (`example.com/#section`) and code.
2. **Normalise**: lower-case (`#NYC` and `#nyc` are the same tag), apply Unicode normalisation (NFKC) so visually identical strings compare equal, and optionally strip accents or fold known variants.
3. **Deduplicate** within one post: "#nyc #NYC #nyc" is one tag.
4. **Cap** the number of tags (ignore the 31st) and their length, to block spam that stuffs posts with hundreds of trending tags.
5. **Store** the normalised tag for lookup and keep the original text, with its capitalisation, for display.

~~~dg-flow Tag extraction runs once on write: find candidates, normalise, deduplicate and cap, then store normalised tags for lookup.
Post text -(regex, Unicode words)-> Candidates #NYC, #nyc, #Marathon -(lower-case, NFKC)-> nyc, nyc, marathon -(dedupe, cap at 30)-> *Tags: nyc, marathon
~~~

<div class="ex">
<div class="lab">Worked example</div>

**normalisation edge cases**

"#NYC #nyc #Nyc!!" yields one tag, `nyc`. "#Café" typed with a combining accent and "#Café" typed with a precomposed é look identical but are different byte sequences; NFKC normalisation makes them the same `café`. "Visit example.com/#pricing" yields no tag, because the match is inside a URL. "#2024" is often rejected (pure numbers are not tags on most platforms). A post with 200 hashtags is stored with its first 30.

</div>

## Option 1: an array column with a GIN index

In PostgreSQL, store the tags in each post row: `tags text[]`. A normal B-tree index can't answer "rows whose array contains 'marathon'", because a B-tree indexes the whole value, and the whole value is `{marathon,nyc}`. **GIN** (Generalized Inverted Index) is built for exactly this: it is designed for composite values (arrays, jsonb, full-text documents) and indexes the **elements** inside them. Its built-in `array_ops` operator class supports `&&` (overlap), `@>` (contains), `<@` (is contained by) and `=`.

```
CREATE INDEX posts_tags_gin ON posts USING GIN (tags);
SELECT id FROM posts WHERE tags @> ARRAY['marathon'];
SELECT id FROM posts WHERE tags @> ARRAY['marathon','nyc'];  -- both tags
```

Internally, a GIN index is a B-tree over **keys** (the distinct tag values), and each key points to a **posting list** of the row IDs that contain it, or to a **posting tree** (a B-tree of row IDs) when the list is large. It is the inverted index from Module 9 (lesson 09.02), inside your database.

~~~dg-tree Inside a GIN index on the tags column: a B-tree of distinct tags, each pointing to the rows that contain it, with large lists stored as their own trees.
GIN entry tree | B-tree of distinct tags
  "marathon" | posting list
    rows 7, 19, 42 | small list stored inline
  "nyc"
    *posting tree | 40 million row IDs, sorted by row ID
  "running"
    rows 7, 88
~~~

The trade-off: GIN favours **search speed over write speed**, since one row can generate many index entries (one per tag), each landing on a different page. To soften this, the **fastupdate** option buffers new entries in an unsorted **pending list** that is merged into the main structure later (on VACUUM, or when it exceeds `gin_pending_list_limit`, 4 MB by default). Searches must also scan the pending list, so a big backlog slows reads and makes latency spiky.

The deeper problem is **ordering**. The posting list is sorted by row ID (physical location), not by creation time. GIN can tell you **which** rows have #nyc, but not which are newest. To return the newest 20, the database must fetch every matching row's `created_at` and sort, or walk a time index and check each row's tags.

## Option 2: a separate tag table

```
CREATE TABLE post_tags (
  tag        TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  post_id    BIGINT NOT NULL,
  PRIMARY KEY (tag, created_at, post_id)
);
```

One row per (post, tag). The primary key is a B-tree sorted by tag, then time, then post ID. "All posts with #marathon, newest first" is a backwards range scan on the key starting at `tag = 'marathon'`, **already in time order**, so it reads exactly the rows it returns. Pagination with a cursor `(created_at, post_id) < (last_seen_time, last_seen_id)` is efficient no matter how deep you go (keyset pagination, lesson 04.12). Including post_id in the key makes it unique and gives a stable tie-break for posts created in the same millisecond.

This is a hand-built inverted index whose posting lists are sorted **by time** instead of by row ID, which is exactly the order the tag page needs.

## Trade-offs

~~~dg-matrix Array column with GIN against a separate tag table, for the operations a hashtag feature needs.
Operation | Array column + GIN | Separate tag table
Write a post with 3 tags | yes, one row, 3 GIN entries | partial, 1 post row + 3 tag rows
Newest 20 posts with #x, popular tag | no, sort all matches | yes, reads 20 index entries
Deep pagination | partial, re-sort each page | yes, keyset cursor
Posts with #a AND #b | yes, tags @> both | partial, intersect or self-join
Count posts per tag, trending | no, count all matches | yes, counter table per tag
Shard by tag | no, tags live with posts | yes, partition by tag
Simplicity at small scale | yes | partial, more moving parts
~~~

The decisive question is "all posts with #x" for a popular tag. With GIN, `#nyc` might match 40 million rows that must be sorted to get the newest 20 (unless combined with other selective filters). The tag table, keyed by (tag, created_at), reads exactly 20 rows. At scale, the tag table usually wins; the array column is fine for small datasets, for multi-tag filters, and for tags used as attributes ("labels on an issue") rather than as feeds.

~~~dg-bars Index entries read to serve the newest 20 posts for a tag with 40 million posts. The tag table reads only what it returns.
GIN, then sort by time | 40000000 | 40 million rows
Tag table, first page | 20 | 20 entries
Tag table, page 500 | 20 | 20 entries, cursor
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the two queries side by side**

GIN: `SELECT id FROM posts WHERE tags @> '{nyc}' ORDER BY created_at DESC LIMIT 20`. The planner either reads all 40 million matching row IDs from the posting tree, fetches each row's created_at and does a top-20 sort (seconds), or walks a created_at index backwards checking each row for the tag, which is fast for #nyc but terrible for a rare tag whose newest use was a year ago.

Tag table: `SELECT post_id FROM post_tags WHERE tag = 'nyc' AND (created_at, post_id) < ($t, $id) ORDER BY created_at DESC, post_id DESC LIMIT 20`. It descends the B-tree once and reads 20 consecutive index entries, about a millisecond for any tag, any page.

</div>

## Counts and trending

"How many posts use #x?" with GIN is a count over a posting tree, expensive for big tags. With the tag table you still shouldn't run `COUNT(*)` on each tag page view; keep a `tag_counts(tag, count)` table incremented when tag rows are inserted (batched for hot tags, lesson 10.01).

Trending is a different question: which tags are growing fastest right now. That belongs in the streaming pipeline (lesson 10.09): emit a "tag used" event for each extracted tag, count per tag in time buckets, compare with a baseline, and keep the top K in a cache. Don't compute trending from the tag table with `GROUP BY` queries.

## Scaling the tag table

Once the tag table is sharded, the natural partition key is the tag: all of #marathon's rows on one shard, so a tag page is one shard query. That creates a new problem: **hot tags**. During a World Cup final, #worldcup might get 50,000 posts per second, all inserted into one partition, at the "newest" end of one B-tree.

The fix is to add a **time bucket** to the partition key: partition by (tag, day) or (tag, hour) for big tags. Inserts for #worldcup today go to the (worldcup, 2026-11-12) partition; the tag page reads today's partition first and moves to yesterday's when it runs out. For extreme tags, add a small random suffix as well and merge a few partitions on read. Wide-column stores such as Cassandra are a natural fit: partition key (tag, bucket), clustering key (created_at DESC, post_id).

~~~dg-compare Partitioning the tag index only by tag, versus by tag and time bucket, when one tag gets very hot.
[Partition by tag]
#worldcup -> One partition, forever
- Every insert hits one shard
- Partition grows without limit
+ Tag page is always one partition
[*Partition by tag and day]
#worldcup, 2026-11-12 -> Today's partition
+ Load moves to a fresh partition daily
+ Old buckets are cold and compact
- Deep pages read several buckets
~~~

Very large systems often move tag search into the search engine (Module 9), which also handles combining tags with text queries, language filtering and ranking ("top" rather than "latest" posts for a tag). The tag table or wide-column index still serves the simple "latest" tab.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the tag table**

500 million posts a day, an average of 0.8 tags each: 400 million tag rows a day. At about 50 bytes per row including index overhead, that's 20 GB a day, about 7 TB a year.

Partitioned by (tag, day), most partitions are tiny (a rare tag used three times today) and a few are large (#worldcup with 20 million rows on final day, 1 GB). Spread over 128 shards by hash of (tag, day), the 400 million rows a day are about 4,600 inserts per second in total, under 40 per shard on average, with peaks several times higher; the shard holding a hot tag's bucket takes far more, which is why the hottest tags also get a random suffix.

</div>

## Common mistakes

- **Extracting at query time** with `LIKE '%#nyc%'`. It can't use an index and matches #nycmarathon.
- **No normalisation**, so #NYC and #nyc are different pages.
- **Assuming GIN gives you order.** It finds matches; it doesn't sort them by time.
- **COUNT(*) per tag page view.** Keep counters.
- **Ignoring edits and deletes.** When a post is edited, diff its old and new tag sets and delete or insert tag rows; when it is deleted, remove its tag rows or filter at read time.

## In an interview

Say you'll extract and normalise tags on write, then compare the two options in one breath: "an array column with a GIN index is simple and good for multi-tag filters, but can't give a popular tag's newest posts without sorting millions of matches; a (tag, created_at, post_id) table is a time-sorted inverted index, so the tag page is a 20-row range scan with keyset pagination, counts are easy, and it shards by tag with time buckets for hot tags". Put trending in the streaming pipeline.

<div class="take">
<div class="lab">Key takeaway</div>

extract and normalise tags at write time. An array column with a GIN index (an inverted index on array elements) is simple and handles multi-tag filters, but returns matches in row order, so a popular tag's newest posts need a huge sort. A separate (tag, created_at, post_id) table gives time-ordered, cursor-paginated reads, cheap counts, and sharding by tag (with time buckets for hot tags), at the cost of an extra row per tag.

</div>

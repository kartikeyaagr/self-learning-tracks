---
id: "16.08"
course: "system-design-101"
module: "16"
slug: "nearby-search"
title: "Nearby search"
keyIdeas: ["Nearby search for places (Yelp-style) combines a geo filter with text search and ranking.", "Index places with location and text in one engine (Elasticsearch geo_point, or PostGIS + full-text) so a query like \"pizza within 2 km\" runs as one filtered query.", "Usually apply the geo filter first (cheap, very selective in dense cities), then text match and rank by relevance, distance, rating and open hours.", "Places change rarely, so the index is mostly read; cache popular area queries.", "Reviews, photos and ratings live in separate services and are joined at display time."]
question: "Search \"pizza within 2 km\". Combine text search and geo filter: which goes first?"
answer: "It depends on which filter is more selective, and the best answer is a single engine that does both, such as Elasticsearch with a text match plus a geo_distance filter, or PostGIS with full-text search, where the engine intersects the two sets and leads with the cheaper one. For \"pizza\", a common term, in a dense area, the 2 km geo filter shrinks the set most (thousands of nearby businesses against tens of thousands of pizza places worldwide), so filter by location first using geohash cells or a spatial index, then match \"pizza\". For a rare term such as an exact restaurant name or \"ethiopian\", matching text first gives a tiny set to distance-check. Either way, finish with exact distances because cells overshoot the circle, and rank by relevance, distance decay and rating."
sources: [{"title": "Hello Interview: Yelp breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp", "note": "Nearby search."}]
tags: ["Design"]
---

## What we are building

Local search in the style of Yelp or Google Maps: a user types "pizza" and the app shows pizza places within 2 km, best first. The query combines two very different filters:

- a **text** match: the business is a pizza place, because its name, category or menu matches "pizza";
- a **geographic** filter: it is within 2 km of the user.

Then the results are sorted, usually by a mix of text relevance, distance and rating. The interesting question is how the two filters work together, and in particular which should run first. Lessons 12.01 to 12.03 cover the spatial indexes (geohash, quadtrees, S2 and H3) this lesson builds on.

### Requirements

- Search businesses by keyword and location (a point plus a radius, or the visible map area), with filters: open now, price level, category.
- See business details and reviews; write a review (one per user per business) and a star rating.
- 100 million daily users, 10 million businesses worldwide.
- Search p99 under 300 ms. Businesses change rarely, so a new or edited business appearing in search a minute later is fine.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing**

100 million daily users run about 5 searches a day: 500 million searches a day, about 6,000 per second on average and perhaps 20,000 at peak.

10 million businesses at about 2 KB each in the search index (name, categories, description, coordinates, hours, rating) is 20 GB. Even with replicas, the whole index fits comfortably in the memory of a small cluster.

Reviews: if each business has 50 reviews on average, 500 million reviews of about 1 KB is 500 GB. Stored in the database, not in the search index; only the aggregate rating goes into the index.

So the search problem is read-heavy and small in data but high in query rate and latency-sensitive. That favours a replicated, in-memory-friendly search index.

</div>

## Why plain database indexes are not enough

A B-tree index on latitude and another on longitude do not answer "within 2 km". A range on latitude gives a horizontal strip around the world; a range on longitude gives a vertical strip. The database can use one of them, then scan the strip. In a busy latitude band (most of Europe's cities sit in a few degrees), that strip holds millions of rows.

Spatial indexes fix this by indexing two dimensions together:

- **Geohash** (lesson 12.02) turns a point into a string where a shared prefix means nearby. A query covers the circle with a handful of cells (the cell containing the user plus its neighbours) and fetches businesses with those prefixes.
- **Quadtrees** split space recursively into four squares until each square holds at most N businesses, so dense cities get small squares and deserts get huge ones.
- **R-trees and BKD trees** group nearby points into nested bounding boxes. PostGIS uses R-tree style GiST indexes; Elasticsearch and Lucene use BKD trees for geo points.

All of them find the businesses in an area quickly. None of them knows anything about "pizza". That is the job of the **inverted index** (lesson 09.02), which maps each word to the list of businesses containing it, and knows nothing about location.

~~~dg-tree The two indexes a local search query needs. One is organised by place, the other by word.
Local search query
  Spatial index | where is it?
    Geohash cells
    Quadtree squares
    BKD tree boxes
  Inverted index | what is it?
    pizza | 40,000 businesses
    ethiopian | 900 businesses
~~~

## Which filter goes first?

This is the lesson's question. Ideally neither: use one index structure that handles both and let the engine choose based on **selectivity**, which is the fraction of rows a filter keeps. Run the filter that cuts the most first, then apply the other to the survivors. A filter that keeps 0.01% of rows is highly selective; one that keeps 10% is not.

- **"Pizza" in Manhattan.** Worldwide, about 40,000 businesses match "pizza" (0.4% of 10 million). Within 2 km of a Manhattan street corner, perhaps 4,000 businesses of any kind exist (0.04%). The geo filter is ten times more selective: find businesses in the area first, then check which of those match "pizza".
- **"Ethiopian restaurant" in a big metro area.** Maybe 900 match the text worldwide, and only a few dozen near the user. The text filter is the more selective one: fetch the 900 and check distance.
- **A rare word**, such as a restaurant's exact name. The text filter shrinks results to a handful; distance-checking them is trivial.

If you must build it from parts, the rule of thumb is **geo first for common terms in dense areas, text first for rare terms**. Both orders then compute the exact distance on the candidates, because cells and boxes are approximations: a geohash cell is a rectangle, and its corners are farther away than 2 km.

~~~dg-flow Two orders for the same query; the more selective filter should run first.
[Geo first] All businesses -(cells covering 2 km)-> *4,000 nearby -(text match pizza)-> 120 results
[Text first] All businesses -(inverted index pizza)-> 40,000 pizza places -(distance under 2 km)-> 120 results
~~~

<div class="ex">
<div class="lab">Worked example</div>

**counting the work in each order**

Geo first: the geohash cells covering 2 km around the user contain about 6,000 businesses (more than 4,000 because cells overshoot the circle). For each, check whether "pizza" appears in its text. If each check is a lookup in a per-document term set, that is 6,000 cheap operations, then an exact distance on the 150 matches, leaving 120.

Text first: read the "pizza" posting list of 40,000 IDs, fetch each one's coordinates (from a column store or doc values), compute 40,000 distances, keep 120.

Geo first does about one seventh of the work here. Now change the query to "ethiopian": geo first still examines 6,000 nearby businesses; text first examines 900. The better order flipped because selectivity flipped. This is exactly the decision a query planner makes using statistics.

</div>

## Doing both in one engine

Search engines such as **Elasticsearch** (also OpenSearch and Solr) handle this natively, which is why they are the standard choice for Yelp-like search. A business document has a text field (name, categories, description), a `geo_point` field, and structured fields (price, rating, opening hours). A query is a `bool` query:

```
{ "query": { "bool": {
    "must":   { "match": { "text": "pizza" } },
    "filter": [ { "geo_distance": { "distance": "2km",
                  "location": { "lat": 40.741, "lon": -73.989 } } },
                { "term": { "open_now": true } } ] } } }
```

- The **must** clause is a full-text match on "pizza", which produces a relevance score (BM25, lesson 09.05).
- The **filter** clause holds `geo_distance` and the structured filters. Filters do not affect scoring, and their results can be cached as bitsets and reused across queries.

Inside each shard, the engine walks the posting list for "pizza" and the set of documents passing the geo filter **together**, leapfrogging: it advances whichever iterator is behind to the next ID at least as large as the other's. It never materialises all 40,000 pizza places or all 4,000 nearby businesses in full. The engine's cost estimates also let it lead with the cheaper iterator, which is the selectivity choice made automatically.

~~~dg-flow One search engine query: text and geo filters intersected inside each shard.
[Query] Query pizza -(must: match)-> Posting list
[Filter] User location -(geo_distance 2 km, BKD tree)-> Nearby set
[Combine] Posting list -(intersect with nearby set)-> *Scored top 20
~~~

**PostgreSQL** can do the same with PostGIS (`ST_DWithin` on a GiST index) plus full-text search (a `tsvector` column with a GIN index). The planner picks which index to start from using its statistics about how many rows each condition will match, and may combine both with a bitmap AND.

## Ranking the results

Matching is only half the job. Users expect the best pizza nearby, not every pizza nearby in random order. A typical scoring function combines:

- **Text relevance**: "Joe's Pizza" (name match) beats a café that mentions pizza once on its menu.
- **Distance decay**: closer is better, but not linearly. A Gaussian or exponential decay function keeps most of the score within a few hundred metres and drops it steeply towards 2 km. In Elasticsearch this is a `function_score` with a `gauss` decay on the location.
- **Quality**: average rating, adjusted for review count, so a 5.0 from 2 reviews does not beat a 4.7 from 2,000. A Bayesian average (blend each business's mean with the global mean, weighted by review count) is a simple and robust choice.
- **Context**: open now, price filter, personal signals if in scope.

~~~dg-bars How a Gaussian distance decay with a 1 km scale weights results at different distances (score multiplier, out of 100).
100 m | 99 | 0.99
500 m | 84 | 0.84
1 km | 50 | 0.50
*1.5 km | 21 | 0.21
2 km | 6 | 0.06
~~~

## The rest of a Yelp-like design

~~~dg-flow Businesses live in the database; search reads a derived index kept in sync by change data capture.
[Write] Owner edits -(API)-> Business DB -(CDC events)-> *Search index
[Review] User review -(one transaction)-> Reviews + rating totals -(CDC)-> Search index
[Read] Search API -(bool query)-> Search index -(IDs)-> Details cache
~~~

- **Data.** Businesses live in a relational database, the source of truth. The search index is a derived copy, synced by change data capture (lesson 09.10). Businesses change rarely, so index lag of seconds to minutes is fine, and you can rebuild the index from the database at any time.
- **Reviews and ratings.** Store reviews in the database. Keep `review_count` and `rating_sum` on the business row and update them **in the same transaction** as each new review, so the average is cheap to read rather than averaging all reviews on every query. A unique constraint on `(user_id, business_id)` enforces one review per user per business at the database level, which is the only place it can be enforced reliably under concurrency.
- **Caching.** Popular searches ("pizza" around a city centre) can be cached for a minute by query, **rounded location** (a geohash cell of about 1 km, so nearby users share entries) and filters. Business details pages are cached by ID.
- **Sharding.** 20 GB fits on one node, so for this size you **replicate** the index rather than shard it: every node holds everything and serves a share of the 20,000 queries per second. If the data were much larger, shard by region so a local query touches one or two shards, and handle queries near a shard boundary by querying both sides.

<div class="ex">
<div class="lab">Worked example</div>

**"pizza within 2 km" end to end**

The user is in Manhattan. The engine's geo filter finds about 4,000 businesses within 2 km; "pizza" matches 120 of them; "open now" leaves 95.

Each is scored: relevance (name contains "pizza" scores higher than a menu mention), distance decay (a place 300 m away keeps about 94% of its score with the 1 km decay scale above; one 1.8 km away keeps about 11%), and Bayesian-adjusted rating. The top 20 come back in about 50 ms.

In a small town the same query finds 30 businesses within 2 km and only 3 pizza places. Many apps then widen the radius automatically (to 5 km, then 10 km) until they have enough results, and say so in the interface.

</div>

## Edge cases

- **The user is at a cell boundary.** Always search the neighbouring cells too, or use a circle-covering function that returns every cell that touches the circle.
- **Map view instead of radius.** Use a bounding-box filter and re-run as the user pans; debounce requests so dragging does not fire 50 queries.
- **Dense hotspots.** A 2 km radius in central Tokyo contains far more businesses than one in a suburb. A quadtree or a k-nearest-neighbours query ("closest 50 matches") adapts better than a fixed radius.
- **Moving entities.** This design assumes businesses do not move. Drivers and couriers do; lesson 17.01 handles fast-moving points with an in-memory geo index updated every few seconds.

## Common mistakes

- Separate latitude and longitude B-tree indexes.
- Running text and geo in two systems and intersecting huge ID lists in application code.
- Forgetting the exact distance check after cell-based filtering.
- Sorting by raw average rating.
- Computing the average rating from all reviews on every query.
- Sharding a 20 GB index that would fit on one machine.

## In an interview

Name the two indexes (spatial and inverted), then answer the "which first" question with selectivity and a concrete count, then say you would use one engine that intersects both and let it choose. Spend remaining time on ranking (distance decay, Bayesian rating) and on keeping the index in sync. Ask: radius or map viewport, how fresh must edits be, and whether personalisation is in scope.

<div class="take">
<div class="lab">Key takeaway</div>

for text plus location, use one engine that indexes both (Elasticsearch with a geo_point and a geo_distance filter, or PostGIS plus full-text search in PostgreSQL), let the more selective filter drive the intersection, finish with an exact distance check, and rank with relevance, distance decay and an adjusted rating. If you hand-build it, go geo first for common terms in dense areas and text first for rare terms.

</div>

---
id: "15.04"
course: "system-design-101"
module: "15"
slug: "a-product-listing"
title: "A product listing"
keyIdeas: ["Product listing pages are read far more than written, with search, filters, sorting and pagination.", "Source of truth in a relational DB (products, variants, prices, inventory). Search and filtering from a search index (Elasticsearch) updated by CDC.", "Cache product pages and listing results (CDN for anonymous users, Redis for fragments).", "Prices and stock change often: keep them out of long-lived caches, or invalidate on change via events. Show stock as approximate on listings and check exact stock at checkout.", "Paginate with cursors; precompute facet counts in the search engine."]
question: "A product page is read 1M times per update. How do you keep price changes visible within seconds?"
answer: "Precompute a denormalised product view and serve it from layered caches (CDN, then Redis, then read replicas), since reads dwarf writes. For freshness, do not rely on TTLs: when a price changes, the write to the primary emits an event (via an outbox or change data capture), and a consumer overwrites the Redis entry with the new value and purges the CDN path, with a short TTL as a safety net. Serve price and stock as a small separate fragment with a very short cache time while the heavy page content stays cached for long. Overwrite rather than delete (or refill from the primary) so a lagging replica cannot re-cache the old price, use version numbers so out-of-order updates are ignored, coalesce requests to avoid a stampede, and have checkout always read the price from the source of truth."
sources: [{"title": "Hello Interview: Scaling reads", "url": "https://www.hellointerview.com/learn/system-design/patterns/scaling-reads", "note": "Replicas, caches, CDNs for read-heavy catalogs."}, {"title": "System Design Primer: Sales rank solution", "url": "https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/sales_rank/README.md", "note": "Catalog-adjacent design."}]
tags: ["Design"]
tools: ["PostgreSQL"]
---

## The shape of the problem

A product page on a large shop is read about a million times for every update. At that ratio the answer is clearly caching: if every view hit the database, a popular product during a sale could demand tens of thousands of queries per second for the same row. But caching works against the other requirement. A price change must show up **within seconds**, and a stale price costs money (selling below cost after a supplier price rise, or honouring a mistaken price) or trust (charging more at checkout than the page showed, which in some countries is also a legal problem).

So the design is a negotiation between two forces: serve almost everything from cache, yet make the few things that change fast visible fast. The order in which to reach for tools is usually: first make each read cheap inside the database (indexes, denormalisation, better hardware), then scale out (read replicas, sharding), then add external caches (application cache, CDN and edge). Read-to-write ratios start around 10:1 for ordinary apps and reach 100:1 or far more for catalogue and content sites.

### Numbers for a mid-sized shop

- 5 million products; 200 million product page views a day, 2,300 per second on average, 25,000 per second at a sale peak.
- 500,000 price or stock changes a day (6 per second), with bursts of 50,000 in a minute when a sale starts.
- A product view document of about 20 KB (text, image URLs, attributes, rating summary).
- Requirement: a price change is visible on every page within 5 seconds.
- Across the whole catalogue that is about 400 views per change. For the most popular products, which are viewed constantly but repriced rarely, the ratio reaches a million to one, and those are the pages that matter most for load.

## Step 1: make each read cheap

- **Denormalise the page model.** A product page usually joins products, prices, inventory, images, ratings and seller data, often across several services. Precompute a single "product view" document whenever any part changes, so a read is one key lookup rather than six queries. This is the **materialised view** idea: pay at write time, which is rare, to save at read time, which is constant.
- **Index** on product ID, and on category or seller for listing pages.
- **Read replicas** take reads off the primary. They lag by milliseconds to seconds, which matters for prices (see below).

Listing pages work the same way. For a best-sellers page, a batch job over sales logs computes each category's top sellers hourly into a `sales_rank` table. Reads (`GET /api/v1/popular?category_id=1234`) are then just an indexed lookup, with a memory cache in front for popular categories. At 1 billion transactions and 100 billion reads a month, that is about 400 writes and 40,000 reads per second, and the reads never touch the transaction tables.

## Step 2: cache in layers

~~~dg-layers The read path for a product page, outermost first; each layer can hold an old price, so each needs a freshness rule.
CDN / edge | whole page or static parts; long TTL plus purge on change
*App cache (Redis) | product view by ID; overwritten on price change
In-process cache | hottest products for 1 to 2 s on each app server
Read replicas | absorb cache misses; may lag by a second or two
Primary database | the source of truth; writes and fallback reads
~~~

Each layer multiplies the protection of the one below. With a 90% CDN hit rate and a 98% Redis hit rate, 25,000 page views per second become 2,500 requests to the app and 50 to the database. The difficulty is that **each layer may hold an old price**, and each needs its own way of being told about a change.

## Step 3: keep prices fresh

### Split what changes from what does not

Descriptions, images, specifications and reviews change rarely; cache them long (minutes or hours, at the CDN). Price and stock change often; serve them separately, from a small, fast lookup or a short-TTL fragment. Many shops render the page shell from the CDN and fetch price and stock with a separate small API call (or an edge-side include) that is cached for only a second or two. The 20 KB document stays cached; only a 100-byte price object needs to be fresh.

### Invalidate on write, do not wait for TTLs

A TTL alone gives you a bad choice: a long TTL means stale prices; a short one means low hit rates and more load. Instead, push changes from the write path. When a price changes:

1. Write the new price to the primary database, the source of truth.
2. Publish a `price_changed` event, either from the application (using an outbox table written in the same transaction, so the event cannot be lost) or by reading the database's change log (**change data capture**, CDC, such as Debezium reading the Postgres WAL).
3. A consumer **overwrites** (or deletes) the cache entry for that product and **purges** the CDN URL for its price fragment.
4. The next read gets the new price.

A TTL (say 60 seconds) stays as a safety net in case an invalidation message is lost.

~~~dg-flow A price change: the write path pushes the new value outward instead of waiting for caches to expire.
[Write] Pricing admin -(UPDATE price, outbox row)-> Primary DB -(CDC)-> Event stream
[Push] Event stream -(consumer)-> *Overwrite product:123 in Redis -(and)-> Purge CDN /api/price/123
[Read] Next shopper -(edge miss)-> Price API -(Redis hit, new value)-> Shopper sees new price
~~~

~~~dg-timeline How long a price change takes to reach shoppers along the push path; the 5-second target is met with room to spare.
0 ms | *Price UPDATE commits on the primary
50 ms | CDC reads the change from the WAL
150 ms | price_changed event published to the stream
300 ms | Consumer overwrites product:123 in Redis
800 ms | CDN purge of /api/price/123 completes at all edges
2,800 ms | Last in-process caches expire (2 s TTL)
3 s | Every shopper sees the new price
~~~

### Beware replica lag after invalidation

If you **delete** the cache entry and the next miss refills it from a replica that has not yet received the update, you cache the old price again, with a fresh TTL. Now the stale price lives for the whole TTL, and no further invalidation is coming.

~~~dg-seq The stale refill race: the cache is deleted, then refilled from a replica that has not yet seen the update.
Admin > Primary : UPDATE price = 40 (was 50)
Primary > Consumer : price_changed(123)
Consumer > Redis : DEL product:123
Shopper > App : GET /p/123
App > Redis : GET product:123
Redis -> App : miss
App > Replica : SELECT price (replica 800 ms behind)
Replica -> App : 50 (old)
App > Redis : SET product:123 price 50, TTL 60 s
Primary > Replica : replicate price = 40 (too late)
~~~

Fixes, from simplest:

- **Overwrite instead of delete**: the invalidation message carries the new value (or the consumer reads it from the primary) and writes it straight into the cache. The entry is never empty, so no refill race.
- **Refill from the primary** for a short window (a few seconds) after a change, using a "recently changed" marker.
- **Delayed second delete**: delete now, and again after a delay longer than the worst replica lag. Crude but effective.

### Order and races between updates

Two quick changes (\$10, then \$12) can be processed out of order by parallel consumers, leaving \$10 in the cache. Give each price a **version number** (incremented on every write, or the database's commit position) and make the cache write conditional: only overwrite if the incoming version is newer. In Redis this is a small Lua script comparing versions. Partitioning the event stream by product ID also keeps one product's events in order.

## Step 4: protect against stampedes and hot keys

When a hot product's cache entry is deleted or expires, thousands of concurrent requests can miss at once and all hit the database. This is a **cache stampede** (lesson 05.05). Fixes:

- **Request coalescing** (single-flight): only one request per key reloads; the others wait for its result.
- **Refresh instead of delete**: overwrite the entry with the new value, so it is never empty.
- **Jittered TTLs**, so many keys do not expire at the same moment.
- **Early refresh**: when a hot entry is close to expiry, one request refreshes it in the background while others keep using the old value.

**Hot keys**: a flash-sale product can overload the single Redis shard holding it. If one product draws most of the 25,000 views per second, its 20 KB entry alone is about 500 MB/s of network traffic from one shard, more than a typical network card carries. Keep a tiny in-process cache (a second or two) on each app server, so 100 servers each ask Redis at most once a second, or replicate the hot key across shards under suffixed names.

<div class="ex">
<div class="lab">Worked example</div>

**a price drop at the start of a sale**

At 12:00:00 the price of product 123 changes from \$50 to \$40, along with 50,000 other products in the same minute.

The writes commit on the primary. CDC publishes 50,000 events, partitioned by product ID across 32 partitions; 32 consumers process them in parallel at about 2,000 per second each, so the whole batch is pushed within about a second.

For product 123 the consumer writes `{price: 40, version: 17}` into Redis with a conditional Lua script (it would refuse version 16 if that arrived late), and purges `/api/price/123` from the CDN. The page HTML at the CDN has a 5-minute TTL, but the price comes from `/api/price/123` with a 2-second edge TTL.

Within about 3 seconds every shopper sees \$40. The database sees a handful of reads for product 123, not 25,000 per second.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what checkout does**

Even with all this, some shopper somewhere loaded the page at 11:59:59 and clicks "buy" at 12:00:30 with \$50 on screen. The checkout service never trusts the displayed price: it reads the current price from the primary (or a strongly consistent price service) and, if it differs from what the cart shows, tells the user before charging. Caches serve the browsing; the source of truth serves the money.

</div>

## Common mistakes

- **TTL-only freshness**, forcing a choice between stale prices and low hit rates.
- **Caching the whole page with the price inside it**, so every price change purges a 20 KB document from every edge.
- **Delete-then-refill from a lagging replica**, re-caching the old price.
- **Unordered invalidations**, letting an older price overwrite a newer one.
- **Trusting the cached price at checkout.**

## In an interview

Start with the ratio and the freshness requirement, then present: denormalised product view, layered caches with hit-rate arithmetic, split of slow and fast-changing data, push invalidation via CDC or an outbox, overwrite with versions to avoid replica-lag and ordering races, coalescing for stampedes, and checkout reading from the source of truth.

<div class="take">
<div class="lab">Key takeaway</div>

for a million reads per update, precompute a denormalised product view and cache it at the app and CDN layers. Keep prices fresh by pushing new values from the write path (CDC or an outbox), not by waiting for TTLs. Separate fast-changing price and stock from slow-changing content, overwrite with version checks to avoid replica-lag and ordering races, coalesce to prevent stampedes, and let checkout read the source of truth.

</div>

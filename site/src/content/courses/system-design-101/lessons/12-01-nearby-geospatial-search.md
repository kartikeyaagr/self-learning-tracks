---
id: "12.01"
course: "system-design-101"
module: "12"
slug: "nearby-geospatial-search"
title: "Nearby geospatial search"
keyIdeas: ["Nearby search answers \"what's within X km of me?\". Indexing latitude and longitude separately doesn't work well: a range on each returns a huge strip that must be intersected.", "Use a spatial index that keeps nearby points close in the index: geohash, quadtree, R-tree, S2 or H3 cells.", "Query: find cells covering the search circle, fetch candidates in those cells, then filter by exact distance and rank.", "Static places (restaurants) suit a precomputed index; moving objects (drivers) need an in-memory index updated every few seconds.", "PostGIS, Elasticsearch geo queries and Redis GEO all provide this."]
question: "Why can't you index lat and lng separately and query both ranges efficiently?"
answer: "A B-tree sorts on one value, so a latitude index returns a band around the whole globe and a longitude index a band from pole to pole; even a composite (lat, lng) index only orders by the second column within equal first values, which almost never happen for real numbers. Either way the database reads tens or hundreds of thousands of entries to find the few hundred in the small area where the two ranges intersect, because a one-dimensional order cannot keep two-dimensional neighbours together. Spatial trees (quadtrees, R-trees, BKD trees) or cell encodings along a space-filling curve (geohash, S2, H3) keep nearby points together so only a small region is read, followed by an exact distance filter."
sources: [{"title": "Hello Interview: Yelp breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/yelp", "note": "Nearby search with geo indexes."}, {"title": "Hello Interview: Proximity search deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/proximity-search", "note": "Geohash, quadtrees, R-trees."}]
---

## The query every location product needs

"Restaurants within 2 km of me." "Drivers near this rider." "Friends who are in this city right now." Every one of these is a **proximity query**: given a point (the user's latitude and longitude) and a radius, return the objects whose positions fall inside the circle, usually sorted by distance and cut to the top few dozen.

Each stored object is just a row with two numbers: **latitude** (degrees north or south of the equator, from −90 to 90) and **longitude** (degrees east or west of the prime meridian, from −180 to 180). The naive SQL is a **bounding box**, a rectangle that contains the circle:

`SELECT * FROM places WHERE lat BETWEEN 40.70 AND 40.74 AND lng BETWEEN -74.02 AND -73.98`

You then compute the true distance for each returned row (with the **haversine formula**, the great-circle distance on a sphere) and keep those under 2 km. The box is a cheap over-approximation; the exact distance check trims its corners.

To make the numbers concrete, imagine a Yelp-like system: about **100 million daily users** and **10 million businesses**. Nearby search is the main read path, so it might run at 5,000 to 20,000 queries per second at peak. Each query must return in tens of milliseconds. The question for this lesson is why the obvious way to index that table does not work, and what does.

## How a B-tree index answers a range query

Start from first principles. A **B-tree index** (lesson 03.04 covers it in detail) keeps keys in **sorted order** across a balanced tree of disk pages. To answer `lat BETWEEN 40.70 AND 40.74`, the database descends the tree to the first key at or above 40.70 (about 3 or 4 page reads for millions of rows), then walks the leaf pages in order until the key passes 40.74. The cost is roughly **log N to find the start, plus the number of matching entries**.

That is excellent when the range is selective. The problem is that "selective in one dimension" is not the same as "selective in two".

### A latitude index returns a strip around the planet

Every place between 40.70 and 40.74 degrees north matches the latitude condition, wherever it is on Earth. That band runs right round the planet at New York's latitude: through central Spain, southern Italy, Turkey, Central Asia, northern China and the western United States. The index gives you a **horizontal strip about 4.4 km tall that circles the whole globe**.

### A longitude index returns a strip from pole to pole

The longitude condition alone matches a **vertical strip** about 3.4 km wide at New York's latitude, from the Arctic to Antarctica, passing New York, the Caribbean near Haiti, and Colombia and Peru.

### Combining them still reads too much

The database has three plans, and none is good:

1. **Use one index, filter the other condition row by row.** It reads the whole latitude strip and discards everything outside the longitude range.
2. **Intersect the two indexes** (a bitmap AND, which Postgres can do). It reads both strips in full, builds two sets of row IDs, and intersects them. You still touch both strips.
3. **Full table scan.** Read all 10 million rows.

### A composite index does not save you

A **composite index** on `(lat, lng)` sounds like a 2D index, but it is not. A B-tree sorts by the first column and uses the second only to **break ties**. Latitudes are real numbers, so ties almost never happen. The index is effectively sorted by latitude alone; within the range 40.70 to 40.74 the longitudes are scattered in no useful order, and the database must still scan the whole strip and check each longitude.

The underlying reason is simple: **a one-dimensional sort order cannot preserve two-dimensional closeness**. Two places on the same latitude, one in New York and one in Madrid, sit right next to each other in the index. Two places 50 m apart, one just north and one just south of 40.70, can be far apart.

~~~dg-compare What each index shape reads for one 4 km box. The B-tree strategies read whole strips; a spatial index reads only the neighbourhood.
[Latitude B-tree]
lat 40.70 to 40.74 -> Strip around the globe -> Filter lng row by row
- Reads about 100,000 entries
- Grows with the whole world's data
[Composite (lat, lng)]
Sort by lat -> Ties broken by lng -> Same strip
- lng order only helps on exact ties
- Behaves like the lat index
[*Spatial index]
Point and radius -> Few cells or nodes -> Exact distance
+ Reads about 600 entries
+ Cost tracks local density, not world size
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what the bad query costs**

10 million businesses spread around the world. A 0.04 degree latitude band (about 4.4 km tall) contains maybe 100,000 businesses worldwide; a 0.04 degree longitude band maybe 50,000. The box where they cross holds about 300.

Plan 1 reads 100,000 index entries and, if the table is not covered by the index, up to 100,000 random heap page reads at roughly 0.1 ms each on SSD: about 10 seconds of I/O in the worst case for 300 results. Even fully cached, scanning 100,000 entries costs tens of milliseconds of CPU per query.

At 10,000 queries per second, that is a billion index entries scanned per second. With a spatial index, each query descends to a handful of leaf nodes or cells covering the box and reads about 300 matches plus a few hundred near misses that the exact distance filter removes: under 1,000 entries, a reduction of about 100 times.

</div>

~~~dg-bars Index entries read per nearby query in the example above. Scale is logarithmic.
Full table scan | 10000000 | 10,000,000
Latitude B-tree | 100000 | 100,000
Longitude B-tree | 50000 | 50,000
*Spatial index | 600 | about 600
Actual results | 300 | 300
~~~

## The fixes: make space searchable

There are two families of solution. Both rest on the same insight: you need a structure where things that are close in 2D are close in the index.

### Family 1: spatial trees

These understand two dimensions directly by recursively dividing space or grouping objects.

**Quadtrees.** Take a square region (the whole map). If it holds more than some threshold of points (say 100), split it into four equal quadrants: north-west, north-east, south-west, south-east. Repeat for any quadrant that is still over the threshold. Each leaf holds at most 100 points. The tree **adapts to density**: Manhattan becomes a deep tree with tiny leaves a few hundred metres across, while the middle of the Pacific stays one huge leaf at depth 2. A query descends only into quadrants that overlap the search circle, then scans the points in the overlapping leaves.

~~~dg-tree A point quadtree over a city: dense areas split deeper, sparse ones stay as large leaves. Leaves hold at most 100 points.
World
  NW quadrant | 40 points, leaf
  NE quadrant | split
    NE.NW | 80 points, leaf
    *NE.NE: Manhattan | split again
      NE.NE.NW | 95 points
      *NE.NE.NE | 100 points, query lands here
      NE.NE.SW | 70 points
      NE.NE.SE | 60 points
    NE.SW | 30 points, leaf
    NE.SE | 12 points, leaf
  SW quadrant | 3 points, leaf
  SE quadrant | 0 points, leaf
~~~

Quadtrees are easy to build in memory and are a common answer in interviews for a "nearby businesses" service: build the tree from the business table at start-up, keep it in RAM on each search server, and rebuild or patch it as businesses are added. With 10 million businesses at about 100 per leaf, you get about 100,000 leaves; at roughly 1 KB per node plus 24 bytes per point (ID plus coordinates), the whole tree fits in well under 1 GB.

**R-trees.** An R-tree wraps each object in its **minimum bounding rectangle** (MBR), the smallest axis-aligned box that contains it, then groups nearby rectangles into larger rectangles, level by level, up to a root. Like a B-tree, it is balanced and its nodes are sized to disk pages (a node may hold 50 to 200 children). A query checks which children's rectangles overlap the search area and descends only into those. Rectangles at the same level may **overlap**, so a query sometimes descends several branches. PostGIS implements R-trees on Postgres's **GiST** (Generalised Search Tree) framework, and the **R\*-tree** variant it draws on chooses splits that keep overlap and total area low. R-trees handle shapes, not just points: polygons, lines, delivery zones.

**k-d trees and BKD trees.** A **k-d tree** splits space alternately by x and y at the median point. Elasticsearch and Lucene store geo fields in **block k-d (BKD) trees**: k-d trees packed into disk-page-sized blocks and built in bulk per index segment. Because Lucene segments are write-once, the tree is built in a batch when a segment is flushed or merged, which is efficient for search engines that index in batches but means updates arrive with a short delay.

### Family 2: encoded cells

These take a different route: convert each 2D position into a **1D key** in such a way that nearby points usually get nearby keys, then store the key in an **ordinary** B-tree or sorted set.

- **Geohash** (lesson 12.02): interleave the bits of latitude and longitude and write them in base 32. Nearby points usually share a prefix, so a prefix scan finds them.
- **S2 and H3** (lesson 12.03): Google's sphere-based square cells and Uber's hexagonal cells, each with a 64-bit cell ID.
- **Redis GEO** (lesson 12.04): stores a 52-bit geohash as the score in a sorted set.

The general recipe is the same in every case.

~~~dg-flow Encoded cells: convert the point to a cell key, scan nearby keys with an ordinary index, then filter by exact distance.
Rider at 40.72, -73.99 -(encode)-> *Cell dr5ru -(scan cell plus neighbours)-> Candidates -(exact distance under 2 km)-> Results
~~~

1. **Encode** the query point into its cell at a resolution roughly matching the search radius.
2. **Expand** to the neighbouring cells, because the circle may cross a cell boundary (lesson 12.02 explains why this is essential).
3. **Scan** each cell's key range in the ordinary index to get candidates.
4. **Filter** candidates by exact distance and sort.

Cells give up a little precision (the candidate set is always a superset of the answer) in exchange for using infrastructure every database already has.

## The space-filling curve idea underneath

Why do encoded cells work at all, when a sort by latitude does not? The trick is a **space-filling curve**: a path that visits every cell of a grid exactly once and tends to stay in one neighbourhood before moving on. Geohash uses the **Z-order curve** (the path traced by interleaving bits); S2 uses the **Hilbert curve**, which never jumps across the grid between consecutive cells. Sorting by position along such a curve means that a small square region usually corresponds to a few **contiguous ranges** of keys.

"Usually" is the important word. Any mapping from 2D to 1D must break some neighbourhoods: there will always be places where two adjacent cells are far apart on the curve. That is why every cell-based search queries several cells and post-filters. The fix is not perfection; it is making the breaks predictable and cheap to work around.

## Choosing between them

| Situation | Typical choice | Why |
| --- | --- | --- |
| Static listings: businesses, hotels, stores | Search engine with geo (Elasticsearch BKD) or PostGIS R-tree | Combines distance with text, category and rating filters; writes are rare |
| Polygons: zones, roads, parcels | PostGIS R-tree, S2 coverings | Needs containment and intersection, not just distance |
| Moving points: drivers, couriers, phones | In-memory cells (Redis GEO, H3 keyed maps) | Hundreds of thousands of cheap overwrites per second |
| Analytics and pricing by area | H3 or S2 cell IDs as a column | Group by a cell ID like any other key |
| A quick feature on an existing SQL database | Geohash column with a B-tree | No new infrastructure |

~~~dg-matrix How the main spatial index types compare on the properties that drive the choice.
Index | Points | Polygons | Cheap updates | Adapts to density | Ordinary B-tree
Quadtree | yes | partial, via bounding boxes | partial, split or merge leaves | yes | no
R-tree (PostGIS) | yes | yes | partial, rebalancing cost | yes | no
BKD (Elasticsearch) | yes | yes | no, batch built per segment | yes | no
Geohash column | yes | partial, as cell coverings | yes, recompute key | no, fixed cells | yes
S2 or H3 cell IDs | yes | yes, as coverings | yes, recompute key | partial, mixed resolutions | yes
~~~

The rule of thumb: **spatial trees for geometry and mostly static data, encoded cells for points that move constantly**. Writes to an R-tree may split and rebalance nodes; writes to a cell index just recompute a key, which is a few arithmetic operations.

<div class="ex">
<div class="lab">Worked example</div>

**designing "nearby businesses" at 10,000 queries per second**

Data: 10 million businesses at about 1 KB of metadata each is 10 GB, but the spatial index needs only ID, latitude and longitude: 24 bytes each, about 240 MB. That fits in RAM on every server.

Choice: an in-memory quadtree (or a geohash table) replicated on 10 search servers, each handling 1,000 queries per second. Businesses change rarely (a few thousand edits a day), so rebuild the index nightly and apply edits as small patches, accepting minutes of staleness for new listings.

Query: the rider's point is located in its leaf (about 25 node visits), neighbouring leaves overlapping the 2 km circle are added (typically 4 to 9 leaves of up to 100 points each), 600 candidates are distance-filtered to 300, the top 20 by a distance and rating score are returned, and the full business records are fetched from a cache by ID. Total server time: about 1 to 3 ms.

</div>

## What goes wrong

- **Radius too large for the structure.** A 50 km query in a city expands to hundreds of cells or leaves and tens of thousands of candidates. Cap the radius, or switch to a coarser resolution and rank by relevance instead of returning everything.
- **Empty areas.** In rural areas, a 2 km radius may return nothing. Products usually expand the radius step by step (2 km, 5 km, 20 km) until they have enough results, which means your index must handle wide queries efficiently too. A quadtree does this naturally because sparse areas have large leaves.
- **Distance on a flat map.** Treating degrees as a flat grid is wrong away from the equator: at 60 degrees north, a degree of longitude is half as long as at the equator. Always do the final filter with haversine (or a projection appropriate to the area).
- **Forgetting the boundary.** Querying only the single cell or leaf that contains the point misses neighbours just across its edge. Always include neighbouring cells.

## In an interview

When asked to design proximity search, say the problem out loud: "A B-tree is one-dimensional, so lat and lng indexes return strips, not neighbourhoods." Then ask the questions that pick the structure: Are the objects static or moving? Points or shapes? How many queries and updates per second? Do results need text or category filters? A static business directory points to a geo-enabled search engine or an in-memory quadtree; a fleet of moving drivers points to an in-memory cell index (lesson 12.04). Finish with the exact distance filter, because it shows you know cells are approximations.

<div class="take">
<div class="lab">Key takeaway</div>

a B-tree orders one dimension, so separate or composite lat/lng indexes return huge strips, not small neighbourhoods. Use a spatial tree (quadtree, R-tree, BKD) for shapes and mostly static data, or encode positions into cells (geohash, S2, H3) along a space-filling curve for moving points, query the cell and its neighbours, then filter candidates by exact distance.

</div>

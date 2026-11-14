---
id: "12.02"
course: "system-design-101"
module: "12"
slug: "geohash-prefix-spatial-index"
title: "Geohash prefix spatial index"
keyIdeas: ["A geohash encodes a lat/lng into a string by repeatedly halving the world and interleaving longitude and latitude bits, then base32-encoding.", "Longer prefixes mean smaller cells: 5 characters ≈ 5 km × 5 km, 6 ≈ 1.2 km × 0.6 km.", "Points sharing a prefix are close, so a B-tree index on the geohash string turns \"nearby\" into prefix range scans.", "Edge problem: two points on either side of a cell boundary can be very close but share a short prefix. Always query the cell plus its 8 neighbours.", "Cells are rectangles of varying size by latitude, which can skew density."]
question: "Two nearby points can have very different geohashes. When, and how do you handle it?"
answer: "It happens whenever two points sit on opposite sides of a cell boundary: they can be a metre apart yet differ from an early character, because the boundary belongs to a coarse level of the halving (the north neighbours of dr5ru start dr72), and at the equator or prime meridian they may share no prefix at all because the very first bits differ. The antimeridian and poles cause the same effect. Handle it by choosing a precision whose cells are at least as large as the search radius, scanning the query point's cell plus its eight neighbouring cells (a 3 by 3 block that always covers the circle), and then filtering the candidates by exact haversine distance."
sources: [{"title": "Wikipedia: Geohash", "url": "https://en.wikipedia.org/wiki/Geohash", "note": "Encoding and precision table."}, {"title": "Movable Type: Geohash demo", "url": "https://www.movable-type.co.uk/scripts/geohash.html", "note": "Interactive."}]
---

## The idea in one sentence

A **geohash** turns a latitude and longitude into a short string, such as `dr5ru`, so that points in the same small area share the same leading characters. Once you have that, "everything near here" becomes "every key that starts with these characters", and an ordinary B-tree index or sorted set can answer it with a range scan. Lesson 12.01 explained why plain latitude and longitude indexes fail; geohash is the simplest fix that works on any database.

Geohash was published by Gustavo Niemeyer in 2008. The underlying idea, interleaving the bits of two coordinates to get a single number (a **Z-order curve** or **Morton code**), was described by G. M. Morton in 1966.

## How a geohash is built

The construction is repeated halving. You keep two intervals, one for longitude (starting at −180 to 180) and one for latitude (starting at −90 to 90), and you write down a bit each time you halve one of them.

1. Take the **longitude** interval and split it at its midpoint. If the point is in the upper half, write **1** and keep the upper half; otherwise write **0** and keep the lower half.
2. Do the same for the **latitude** interval.
3. Keep alternating: longitude, latitude, longitude, latitude. The bits are therefore **interleaved**: even positions come from longitude, odd positions from latitude.
4. Group the bits into fives and write each group as one **base-32** character from the alphabet `0123456789bcdefghjkmnpqrstuvwxyz` (digits and letters, leaving out a, i, l and o so the strings are hard to misread).

Each character carries 5 bits, so each extra character divides the current cell into 32 smaller cells. Because 5 is odd, the cells alternate in shape: odd-length geohashes give cells that are square in degrees (the first character splits the world 8 ways in longitude and 4 in latitude, giving 45° × 45° cells), and even-length ones give cells twice as wide as they are tall.

~~~dg-tree Encoding 42.605 N, 5.603 W: each halving keeps one half and writes one bit. The first five bits, 01101, become the character e.
Whole world
  Bit 1, lng: west half | 0, keep -180 to 0
    Bit 2, lat: north half | 1, keep 0 to 90
      Bit 3, lng: east of -90 | 1, keep -90 to 0
        Bit 4, lat: below 45 | 0, keep 0 to 45
          *Bit 5, lng: east of -45 | 1, so 01101 = e
            Bits 6 to 10: 11111 | z
              Bits 11 to 25 | s, 4, 2
~~~

<div class="ex">
<div class="lab">Worked example</div>

**encoding a point by hand**

Take 42.605 N, 5.603 W. Longitude −5.603 is below the midpoint 0 of −180 to 180, so bit 1 is 0 and the interval becomes −180 to 0. Latitude 42.605 is above the midpoint 0 of −90 to 90, so bit 2 is 1 (interval 0 to 90). Longitude is above −90, bit 3 is 1 (−90 to 0). Latitude is below 45, bit 4 is 0 (0 to 45). Longitude is above −45, bit 5 is 1 (−45 to 0).

The first five bits are 01101, which is 13, and character number 13 of the alphabet (counting from 0) is `e`. Continuing gives 11111 (31, `z`), 11000 (24, `s`), 00100 (4, `4`) and 00010 (2, `2`). The geohash is `ezs42`, a cell about 4.9 km across that contains the point.

Decoding runs the same process backwards: each bit tells you which half to keep, and the final pair of intervals is the cell. Its centre is the decoded position, and half the cell size is the error.

</div>

## Precision: how big is a cell?

Each character shrinks the cell by a factor of 32 in area. The table gives approximate cell sizes near the equator; the north-south size is fixed, but the east-west size in kilometres shrinks with the cosine of latitude, so at 60 degrees north cells are about half as wide.

| Length | Bits | Cell size (width × height) | Approximate error |
| --- | --- | --- | --- |
| 1 | 5 | 5,000 km × 5,000 km | ±2,500 km |
| 2 | 10 | 1,250 km × 625 km | ±630 km |
| 3 | 15 | 156 km × 156 km | ±78 km |
| 4 | 20 | 39 km × 19.5 km | ±20 km |
| 5 | 25 | 4.9 km × 4.9 km | ±2.4 km |
| 6 | 30 | 1.2 km × 0.61 km | ±610 m |
| 7 | 35 | 153 m × 153 m | ±76 m |
| 8 | 40 | 38 m × 19 m | ±19 m |
| 9 | 45 | 4.8 m × 4.8 m | ±2.4 m |

~~~dg-bars Geohash cell width by length, near the equator. Every character divides the area by 32, so the scale is logarithmic.
Length 3 | 156000 | 156 km
Length 4 | 39000 | 39 km
*Length 5 | 4900 | 4.9 km
Length 6 | 1200 | 1.2 km
Length 7 | 153 | 153 m
Length 8 | 38 | 38 m
Length 9 | 4.8 | 4.8 m
~~~

Store the geohash at the finest precision you will ever need (say 9 or 12 characters), and query at whatever precision suits the radius. A prefix of a long geohash is the geohash of the containing coarser cell, so one column serves every zoom level.

~~~dg-widget geohash Geohash as nested cells: each character zooms into one of 32 sub-cells, and every point inside shares the prefix. Tap a cell to zoom in.
~~~

## Why prefixes mean nearness

Points in the same cell share every character that identified that cell. All points in the 4.9 km cell `dr5ru` (which covers Midtown Manhattan, including the Empire State Building at `dr5ru6j`) have geohashes beginning `dr5ru`. "Everything in this cell" is therefore a **prefix query**:

- In SQL: `WHERE geohash LIKE 'dr5ru%'`, which a B-tree can serve as the range `geohash >= 'dr5ru' AND geohash < 'dr5rv'`.
- In a key-value store with ordered keys (Cassandra clustering columns, DynamoDB sort keys, Bigtable row keys), a range scan between the same two bounds.
- In Redis, the bits are kept as a 52-bit integer score in a sorted set and the range is a score range (lesson 12.04).

That is the whole trick: a 2D square becomes a 1D range, and a 1D range is what every database is good at.

## The catch: close points need not share a prefix

The relationship only works one way. **Points that share a long prefix are close. Points that are close do not necessarily share a long prefix.** The reason is cell boundaries.

Look at what the 3 × 3 block of cells around `dr5ru` is called:

~~~dg-matrix The eight neighbours of dr5ru. The row to the north starts dr72, not dr5r: those cells are adjacent on the map but far away in key order.
Row | West | Middle | East
North | dr725 | dr72h | dr72j
Middle | dr5rg | **dr5ru**, query | dr5rv
South | dr5re | dr5rs | dr5rt
~~~

The cells to the east and west share four characters with `dr5ru`. The cells to the north share only three, because a boundary at a coarser level (between the 4-character cells `dr5r` and `dr72`) runs along the top edge of `dr5ru`. A rider standing 10 m south of that line is in `dr5ru`; a driver 10 m north of it is in `dr72h`. A query for `dr5ru%` misses the driver completely.

How bad can it get? It depends on how coarse the boundary is:

- **Ordinary cell edges** at the query precision: neighbours differ in the last character or two.
- **Edges of coarse cells**: neighbours can differ from the second or third character on, as with `dr5r` and `dr72`.
- **The equator and the prime meridian**: these are the very first halvings, so points a metre apart can share **no prefix at all**. A point just north of the equator at longitude 10 E encodes as `s0p058`; one just south encodes as `kpzpgx`. At 51.5 N, the Greenwich meridian gives `u10hbp` just east of it and `gcpuzz` just west.
- **The 180° meridian**: the far east and far west of the longitude range are physically adjacent across the Pacific but sit at opposite ends of the key space.
- **The poles**: all longitudes converge, cells become slivers, and "neighbour" stops meaning much.

There is also a distortion issue: because longitude degrees shrink towards the poles, a geohash cell of a given length covers less ground east to west at high latitudes. A precision that suits Lagos is too fine for Oslo by a factor of two.

~~~dg-compare Querying one cell versus the cell and its eight neighbours, for a rider 10 m from a cell edge.
[Single prefix]
Rider in dr5ru -> Scan dr5ru% -> Driver in dr72h missed
- Misses anything across the nearest edge
- Results depend on where the cell lines fall
[*Cell plus 8 neighbours]
Rider in dr5ru -> Scan 9 prefixes -> Haversine filter
+ The 3 by 3 block always covers the circle
+ Correct regardless of cell lines
- Up to 9 range scans and more candidates
~~~

## The fix: query the neighbours too

The standard solution is the **3 × 3 search**:

1. Choose a precision whose cell size is **at least the search radius** in both directions, so the circle can extend at most one cell beyond the query cell in any direction.
2. Compute the query point's cell at that precision.
3. Compute its **8 neighbouring cells**. Libraries provide `neighbours(hash)`; it works by decoding the cell, stepping one cell width or height in each direction, and re-encoding (with wrap-around at 180°).
4. Run **9 prefix scans** (they can be issued in parallel, or merged into fewer ranges where neighbours happen to be contiguous in key order).
5. **Post-filter** every candidate by exact distance (haversine), because nine cells cover far more area than the circle, then sort by distance.

Why does a 3 × 3 block always cover the circle? If each cell is at least r wide and r tall, then a circle of radius r centred anywhere inside the middle cell cannot reach past the outer edges of its neighbours. The worst case is a point in a corner of the query cell; even then the circle stays inside the block.

How much extra area do you scan? Nine cells of side r cover 9r², while the circle covers πr², about 3.14r². Roughly a third of the candidates survive the distance filter. If cells are bigger than r, the ratio gets worse: with cells of side 2r, nine cells cover 36r², and fewer than 9% of candidates survive.

<div class="ex">
<div class="lab">Worked example</div>

**choosing precision for "within 1 km"**

Precision 6 cells are about 1.2 km × 0.6 km near the equator. Their height (0.6 km) is smaller than the 1 km radius, so the circle can reach two cells away north or south and a 3 × 3 block may not cover it. Precision 5 cells (4.9 km × 4.9 km) are larger than the radius in both directions, so 9 of them always do.

So: compute the point's precision-5 cell and its 8 neighbours and scan 9 prefixes. The block is about 14.7 km × 14.7 km, roughly 216 km², while the circle is 3.14 km². Only about 1.5% of the area is useful.

In a dense city with 100 places per km², that is about 21,600 candidates for about 314 results. Better: use precision 6 and a larger block of cells, about 3 wide and 5 tall (15 cells, roughly 3.6 km × 3.0 km, 11 km²), which yields about 1,100 candidates. More scans, far fewer rows. Many implementations pick the block size automatically from the radius and the cell dimensions at that latitude.

</div>

## Storing geohashes in practice

A few patterns appear in real systems:

- **One column, many precisions.** Store a 12-character geohash (precise to a few centimetres) and query any prefix length. Some systems also store the 5- and 6-character prefixes as separate indexed columns so they can use equality (`geohash6 IN (...)`) instead of `LIKE`.
- **Partition keys.** In a distributed store, use a coarse prefix (say 4 characters) as the partition key and the full geohash as the sort key. All places in one roughly 40 km cell live on one partition, and a radius query touches one to four partitions. Watch for **hot partitions**: the cell over central Tokyo holds vastly more rows than one over Siberia.
- **Search engines.** Solr and Elasticsearch supported geohash prefix filters and grid aggregations; the same neighbour-list idea applies there, even though modern versions use BKD trees for point queries.
- **Map clustering.** Grouping points by a geohash prefix is a quick way to draw "142 restaurants here" bubbles at low zoom: group by the first 4 or 5 characters and count.

## Pitfalls

- **Querying only the containing cell.** The most common bug. It passes tests when test points are in the middle of cells and fails in production at the edges.
- **Precision chosen in degrees, not metres.** A precision that works at the equator is wrong at high latitude because cells are narrower. Compute block size from the real cell dimensions at the query latitude.
- **Skipping the distance filter.** The 3 × 3 block is a square up to three times the radius on each side. Without the haversine filter you return places 3 km away for a 1 km query.
- **Using string comparison on mixed lengths.** `dr5ru` sorts before `dr5ru6j`, which is what you want for prefix ranges, but make sure the upper bound of the range is the next prefix (`dr5rv`), not `dr5ru` plus a high character, or you will miss rows.
- **Expecting good behaviour near the poles or the antimeridian.** If your product serves Fiji, the Aleutians or polar research stations, handle wrap-around explicitly or use S2 (lesson 12.03).

## In an interview

Geohash is often the right first answer for "find nearby X" because it needs no special infrastructure. Explain the encoding briefly (interleaved bits, base 32), say that a prefix is a cell and a cell is a range scan, and then volunteer the boundary problem before the interviewer asks. Describe the 3 × 3 neighbour search and the exact distance filter, and mention how you would choose the precision from the radius. If the interviewer pushes on uneven density (Manhattan versus a desert), mention that a fixed grid does not adapt, and that a quadtree (lesson 12.01) or mixed-resolution cells do.

<div class="take">
<div class="lab">Key takeaway</div>

a geohash interleaves longitude and latitude bits and writes them in base 32, so points in the same cell share a prefix and a 2D area becomes a 1D range scan. Points near a cell edge (or the equator, prime meridian, antimeridian and poles) can have very different prefixes, so always query the cell plus its 8 neighbours at a precision where cells are at least the radius, then filter by exact distance.

</div>

---
id: "12.03"
course: "system-design-101"
module: "12"
slug: "grid-systems-h3-s2-geohash"
title: "Grid systems: H3, S2, geohash"
keyIdeas: ["Grid systems divide the earth into cells with IDs, to index and aggregate location data.", "**Geohash**: rectangles, simple, string prefixes.", "**S2** (Google): projects the sphere onto a cube and uses a Hilbert curve; cells are roughly square, with good locality, and region coverings are cheap to compute.", "**H3** (Uber): hexagons at 16 resolutions. Every neighbour of a hexagon is the same distance away, which makes movement, smoothing and radius queries cleaner.", "Uber uses H3 for things like surge pricing zones and supply/demand analysis."]
question: "Why do hexagons have more uniform neighbour distances than squares?"
answer: "A hexagon's six neighbours all share a full edge with it, and by symmetry all six centres are the same distance away; no neighbour touches only at a corner. A square has four edge neighbours at distance d and four corner neighbours at about 1.41d, so its neighbours sit at two different distances (triangles have three). With one distance, rings of hexagons approximate circles, smoothing can give every neighbour the same weight, movement always crosses into exactly one neighbour, and spatial maths needs a single set of coefficients; squares make diagonal directions behave differently from axis directions."
sources: [{"title": "Uber: H3", "url": "https://www.uber.com/blog/h3/", "note": "Hexagonal grid and why hexagons."}, {"title": "S2 Geometry", "url": "https://s2geometry.io/", "note": "Google's sphere-based cells."}]
---

## Why another grid?

A **discrete global grid** is a way of cutting the whole surface of the Earth into cells, each with an ID, usually at many resolutions nested inside one another. Geohash (lesson 12.02) is one. It works, but it has three structural weaknesses:

- **Distortion.** Its cells are rectangles in latitude and longitude. Because lines of longitude converge towards the poles, a geohash cell at 60 degrees north is half as wide on the ground as one at the equator, and near the poles cells become thin slivers.
- **Seams.** The equator, the prime meridian and the 180° meridian are boundaries of the very first bits, so neighbours across them have unrelated keys.
- **Two neighbour distances.** Square-ish cells have edge neighbours and corner neighbours at different distances, which makes "the cells around me" a lopsided idea.

Two production systems fix these problems in different ways: Google's **S2**, which uses squares projected onto a sphere, and Uber's **H3**, which uses hexagons. This lesson explains how each is built, why hexagons behave better for some jobs, and how to choose.

## S2: a cube wrapped around the sphere

S2 models the Earth as a **sphere** rather than a flat map, so there is one coordinate system for the whole planet with **no seams or singularities**. It is built in four steps.

1. **Enclose the sphere in a cube.** Each point on the sphere is projected outward onto one of the cube's **six faces**. A face covers about one sixth of the Earth.
2. **Correct the distortion.** A straight projection would make cells near the face corners much smaller than cells near the face centres. S2 applies a quadratic transform to the face coordinates so that cells at the same level are within a factor of about 2 of each other in area everywhere on Earth.
3. **Subdivide each face like a quadtree.** Level 0 is the face. Each cell splits into four children, down to **level 30**. A level-30 leaf cell is under a square centimetre.
4. **Number the cells along a Hilbert curve.** A **Hilbert curve** is a space-filling curve that visits every cell of a grid once, always stepping to an adjacent cell. Because consecutive cells on the curve are physically adjacent, a compact region usually maps to a few contiguous ranges of IDs.

~~~dg-tree The S2 hierarchy: six cube faces, each a quadtree down to level 30. Truncating a cell ID gives its parent.
Earth: 6 cube faces
  Face 2 | level 0, about 85 million km2
    Child 0 to 3 | level 1, 4 per parent
      ... | each level splits into 4
        Level 10 cell | about 80 km2
          *Level 13 cell | about 1.3 km2, neighbourhood
            Level 16 cell | about 20,000 m2, a block
              Level 30 leaf | under 1 cm2
~~~

### The 64-bit cell ID

Every S2 cell has a 64-bit ID laid out as: **3 bits** for the face, then **2 bits per level** saying which of the four children was chosen, then a single **1 bit** marking where the meaningful bits end, then zeros. A level-30 cell uses all 3 + 60 bits plus the marker. To get a cell's parent at any level, you clear the low bits and move the marker up, which is a couple of bit operations. So the hierarchy is **exact**: every child lies entirely inside its parent.

### The region coverer

S2's most useful tool is the **region coverer**: give it any region (a circle, a polygon, a country) and limits such as "at most 8 cells, levels 10 to 16", and it returns a set of cells that covers the region, using big cells for the interior and small cells along the edges. Because every cell is a contiguous range of leaf IDs, a covering is a short list of **ID ranges**, and "find points in this region" becomes a handful of range scans on an ordinary sorted index. Coverings can also be **interior** (cells entirely inside the region), which lets you skip exact tests for points that land in them (lesson 12.05).

S2 is the cell system under MongoDB's `2dsphere` index, and it is used widely in Google's mapping products. Because it works on the sphere, it handles polygons that cross the antimeridian or contain a pole, which flat latitude and longitude grids mangle.

<div class="ex">
<div class="lab">Worked example</div>

**an S2 covering for a 2 km radius search**

The rider is at 40.72 N, 73.99 W. The circle has an area of about 12.6 km². Ask the coverer for at most 8 cells between levels 12 and 15. It might return 2 level-12 cells (each about 5 km²) for the middle and 6 level-13 cells (about 1.3 km² each) around the edges: about 18 km² in total, roughly 5 km² more than the circle.

Each cell becomes one range of leaf IDs, so the query is 8 range scans on a B-tree column holding each point's level-30 cell ID. Adjacent cells along the Hilbert curve often merge into a single range, so it might be 5 scans in practice. Every candidate is then checked against the exact 2 km distance.

</div>

## H3: hexagons on an icosahedron

Uber built **H3** to support **dynamic pricing** (measuring supply and demand per hexagon in each city), **dispatch** decisions, and city-wide analysis and visualisation of trips. Its construction is different from S2's:

1. **Project the Earth onto an icosahedron**, a solid with **20 triangular faces**. The icosahedron is the Platonic solid closest to a sphere, so distortion is low.
2. **Tile the faces with hexagons.** At the coarsest resolution there are **122 base cells**.
3. **Accept 12 pentagons.** A sphere cannot be tiled with hexagons alone (a consequence of Euler's formula for polyhedra), so there are exactly **12 pentagons** at every resolution, one centred on each icosahedron vertex. Uber orients the icosahedron (following Buckminster Fuller's Dymaxion map) so all 12 land in the ocean, where almost nobody calls a ride.
4. **Refine in steps of 7.** There are **16 resolutions**, 0 to 15. Each finer resolution has hexagons of about **one seventh** the area of the previous. This is called **aperture 7**.

A hexagon cannot be split exactly into 7 smaller hexagons, so each resolution's grid is rotated slightly relative to the previous one, and the 7 children are only **approximately** contained in their parent: bits of the children stick out past the parent's edge and bits of the parent are covered by children of a neighbour. Indexing at a **single resolution** is exact; rolling up to a parent introduces a small, bounded error. If your analytics sum resolution-9 counts into resolution-7 parents, a few events near hexagon edges are attributed to a neighbouring parent.

~~~dg-bars Average H3 cell area at selected resolutions. Each step divides the area by about 7.
Resolution 5 | 252.9 | 253 km2
Resolution 6 | 36.13 | 36 km2
Resolution 7 | 5.161 | 5.2 km2
*Resolution 8 | 0.737 | 0.74 km2
Resolution 9 | 0.1053 | 0.11 km2
Resolution 10 | 0.01504 | 15,000 m2
Resolution 11 | 0.00215 | 2,150 m2
~~~

### The H3 index and its operations

Each cell is a **64-bit integer**: a few mode and reserved bits, 4 bits for the resolution, 7 bits for the base cell, then fifteen 3-bit digits (each 0 to 6) naming which child was taken at each level. Key operations:

- `latLngToCell(lat, lng, res)`: point to cell ID (formerly `geoToH3`).
- `gridDisk(cell, k)`: all cells within **grid distance** k (formerly `kRing`). k = 1 gives the cell plus its 6 neighbours (7 cells); in general a disk has 1 + 3k(k + 1) cells, so k = 2 gives 19.
- `cellToParent` and `cellToChildren`: move between resolutions.
- `polygonToCells`: all cells whose centres lie inside a polygon.
- `compactCells`: replace each complete set of 7 children with their parent, so a region is stored as a mix of resolutions. A city-sized region of about 10,000 fine hexagons typically compacts to under a thousand cells, because its interior is replaced by a few large parents.

## Why hexagons: one neighbour distance

In any grid, the **neighbours** of a cell are the cells it touches. The question that matters is how far away their centres are.

- A **square** with side d has **8 neighbours**: 4 share an edge, with centres at distance **d**, and 4 touch only at a corner, with centres at **d√2 ≈ 1.41d**. That is **two** different distances.
- A **triangle** has neighbours sharing an edge and neighbours sharing only a vertex, at **three** different distances.
- A **hexagon** has exactly **6 neighbours**, and every one shares a full edge. By symmetry, all six centres are the **same distance** away (√3 times the edge length). There is no such thing as a corner-only neighbour.

The deep reason is geometric: hexagons are the regular polygon with the most sides that still tiles the plane, which makes them the closest a tile can get to a circle. A hexagonal tiling is also the only regular tiling in which neighbouring cells never meet at a single point.

~~~dg-compare Neighbour structure of the two grid shapes. Hexagon neighbours are all equivalent; square neighbours come in two kinds.
[Square grid]
Centre -(4 edge neighbours)-> distance d
Centre -(4 corner neighbours)-> distance 1.41 d
- Two distances, two kinds of neighbour
- Rings look like squares, not circles
[*Hexagon grid]
Centre -(6 edge neighbours)-> distance d
+ One distance, one kind of neighbour
+ Rings approximate circles
- Children only approximately nest
~~~

### Why it matters in practice

- **Rings approximate circles.** `gridDisk(cell, 2)` is a near-circular blob of 19 cells. With squares, the cells within two steps are either a diamond (if you only count edge neighbours) or a square whose corners are 41% further away than its sides (if you count diagonals). Radius searches, catchment areas and "how far can a driver reach in 5 minutes" all treat every direction alike on hexagons.
- **Smoothing is uniform.** Surge pricing blends each cell's demand with its neighbours' to avoid sharp price cliffs between adjacent cells. On hexagons, one weight serves all six neighbours. On squares you must decide whether diagonals count and how much less, and any choice creates directional artefacts.
- **Movement is cleaner.** A car crossing a city leaves a trail of cells. On a square grid it can pass exactly through a corner, touching two diagonal cells in an ambiguous way. On hexagons, movement always crosses an edge into exactly one neighbour, which **minimises quantisation error** in flow and trajectory analysis.
- **Simpler maths.** Gradients, diffusion and flow calculations on a grid need coefficients per neighbour type. One type means one coefficient.

<div class="ex">
<div class="lab">Worked example</div>

**how circular is a ring?**

Take resolution-8 hexagons (about 0.74 km² each, centres about 0.8 km apart). `gridDisk(cell, 2)` gives 19 cells, about 14 km². The equivalent circle (14 km² area) has a radius of about 2.1 km, and every cell centre in the disk is within about 1.6 km of the centre, roughly evenly in all directions.

Now a square grid with 0.86 km sides (the same area per cell). "Within 2 steps including diagonals" is a 5 × 5 block, 25 cells and 18.5 km². Its edge centres are 1.7 km away but its corner centres are 2.4 km away, so a "2-step" search reaches 41% further towards the diagonals than along the axes.

</div>

## The trade-off of hexagons

Hexagons are not free. H3 cells are **not laid along a space-filling curve** in a way that keeps neighbours close in ID order, so you cannot answer "everything near here" with a single range scan. Instead you **compute the exact set of cell IDs** you need (a `gridDisk`) and query `WHERE h3_cell IN (...)` or do several key lookups. For a disk of 19 cells, that is 19 lookups, which is fine in memory or in a key-value store and acceptable in SQL. The approximate parent-child nesting, as described above, also makes hierarchical roll-ups slightly lossy.

S2 makes the opposite trade: squares with an exact hierarchy and Hilbert ordering, so regions become ID ranges, but with two neighbour distances and cells whose shapes vary somewhat across a face.

~~~dg-matrix Geohash, S2 and H3 on the properties that usually decide the choice.
Grid | Cell shape | Low distortion | Exact hierarchy | Range-scan friendly | Uniform neighbours
Geohash | lat and lng rectangles | no, high near poles | yes, prefixes | yes, Z-order | no, two distances
S2 | squares on cube faces | yes, low | yes, ID truncation | yes, Hilbert curve | no, two distances
H3 | hexagons plus 12 pentagons | yes, low | partial, approximate | no, compute cell sets | yes, one distance
~~~

## Choosing a grid

- **Geohash** when you want something simple on an existing database, mostly for point search in populated latitudes, and you are happy to handle neighbours and edges yourself.
- **S2** when you need **region coverings** and polygons, global correctness (antimeridian, poles), and range scans on a sorted store. Good for storage indexes and "which documents fall in this region" queries.
- **H3** when the job is **aggregation and analysis by area**: demand and supply per cell, heat maps, pricing, ETAs, movement between cells, machine-learning features keyed by cell. The uniform neighbours make smoothing and spatial statistics behave.

Many systems use more than one: H3 for analytics and pricing, a geohash or S2-backed index for storage queries, and an R-tree in PostGIS for polygons.

<div class="ex">
<div class="lab">Worked example</div>

**surge pricing in H3**

Uber buckets ride requests and available drivers by resolution-8 hexagon (about 0.74 km² each). Every minute, for each hexagon, it computes a demand-to-supply ratio: 42 requests and 14 drivers gives a ratio of 3.0.

Raw ratios are noisy, and a price that jumps between adjacent blocks invites drivers to game the boundary. So each cell's ratio is smoothed with its `gridDisk(cell, 1)` neighbours: for example 50% weight on the cell itself and the remaining 50% split equally across the 6 neighbours, about 8.3% each. Because all six are equally far away, equal weights are correct. A square grid would need one weight for the 4 edge neighbours and a smaller one for the 4 diagonal ones, and the result would still show faint diagonal artefacts.

</div>

## Pitfalls

- **Forgetting the pentagons.** At every resolution, 12 cells have 5 neighbours, not 6. Code that assumes `gridDisk(cell, 1)` always returns 7 cells breaks near them. They are in oceans, but global datasets do touch them.
- **Mixing resolutions carelessly.** Joining a resolution-8 table to a resolution-9 table needs an explicit parent or child conversion; and for H3 that conversion is approximate.
- **Choosing too fine a resolution.** Resolution 12 cells (about 300 m²) give one or two events each per hour; statistics become noise. Pick the coarsest resolution that still resolves the effect you care about.
- **Assuming close IDs mean close cells.** That holds roughly for S2 along the Hilbert curve and not at all for H3.

## In an interview

If you propose a grid, say which and why. "Geohash for the business search, because it is a column on our existing database, with a 3 × 3 neighbour query" is a fine answer for a directory. For anything about pricing, demand, ETAs or heat maps, H3 is the natural choice, and you should be ready to explain the hexagon argument: one neighbour distance, so rings are near-circular and smoothing is uniform. If the interviewer asks about polygons or global coverage, mention S2's region coverer and exact hierarchy.

<div class="take">
<div class="lab">Key takeaway</div>

S2 projects the sphere onto a cube and subdivides each face into a quadtree of square cells numbered along a Hilbert curve, giving low distortion, no seams, exact 64-bit hierarchical IDs and range-scan-friendly coverings. H3 tiles an icosahedron with hexagons (plus 12 pentagons) in 16 resolutions of aperture 7. Hexagons have a single centre-to-neighbour distance (squares have two, triangles three), so rings approximate circles and smoothing and movement behave the same in every direction, at the cost of approximate nesting and no range scans.

</div>

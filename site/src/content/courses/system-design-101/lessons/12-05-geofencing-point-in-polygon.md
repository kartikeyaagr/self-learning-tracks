---
id: "12.05"
course: "system-design-101"
module: "12"
slug: "geofencing-point-in-polygon"
title: "Geofencing: point in polygon"
keyIdeas: ["Geofencing checks whether a location is inside defined areas (delivery zones, airports, surge zones), usually polygons.", "The core test is point-in-polygon.", "With many geofences, don't test every polygon: index polygons by their bounding boxes or by the grid cells (S2/H3) they cover. For each location, look up candidate polygons in that cell, then run the exact test only on those.", "PostGIS does this with spatial indexes (`ST_Contains`).", "For enter/exit events, remember each object's previous zones and compare."]
question: "You have 100k geofences. How do you avoid testing every polygon per location update?"
answer: "Use a two-stage filter and refine. Index the geofences' bounding boxes in a spatial index (an R-tree, which PostGIS uses automatically for ST_Contains) so each update finds only the handful of fences whose boxes contain the point in a few dozen comparisons, then run the exact point-in-polygon test on those few. For very high rates, precompute a covering of each fence with grid cells (H3, S2 or geohash) into a cell-to-fences map marking interior and boundary cells: a point in an interior or empty cell is answered by one hash lookup, and only boundary cells need an exact test against one or two polygons. Caching each vehicle's last cell and emitting only enter and exit events cuts the work further."
sources: [{"title": "Wikipedia: Point in polygon", "url": "https://en.wikipedia.org/wiki/Point_in_polygon", "note": "Ray casting and winding number."}, {"title": "PostGIS: ST_Contains", "url": "https://postgis.net/docs/ST_Contains.html", "note": "In-database geofence checks."}]
---

## The problem

A **geofence** is a virtual boundary drawn on a map, almost always stored as a **polygon**: an ordered list of vertices (latitude and longitude pairs) whose edges enclose an area. Products use them everywhere:

- an **airport pickup zone**, where drivers must queue and riders get special instructions;
- a **delivery area** for a restaurant or a dark store;
- a **surge or pricing region**;
- a **tax or regulatory region** (a city that charges a congestion fee);
- a **no-fly or no-parking zone** for drones and scooters.

The core operation is: for each incoming location update, **which geofences contain this point?** Often the product cares about the change, not the state: "driver entered the JFK zone", "scooter left the permitted area".

Now the scale. Suppose there are **100,000 geofences** worldwide and **250,000 location updates per second** (the driver fleet from lesson 12.04). Testing every polygon for every update would be 25 billion point-in-polygon tests per second. Each test is itself proportional to the polygon's number of edges (lesson 12.06), and a detailed city boundary may have 2,000 vertices. That is not a system you can build. The whole lesson is about not doing that.

## The principle: filter, then refine

Almost every spatial query engine uses the same two-stage pattern, called **filter and refine**:

1. **Filter**: use a cheap, approximate test, supported by an index, to find the small set of fences that **might** contain the point.
2. **Refine**: run the exact, expensive test only on those candidates.

The cheap approximation is the **bounding box** (or minimum bounding rectangle, MBR): the smallest axis-aligned rectangle containing the polygon, stored as four numbers (min and max latitude, min and max longitude). Testing whether a point is in a box is four comparisons. A point outside the box is certainly outside the polygon. A point inside the box may or may not be inside the polygon (it could be in a corner of the box that the polygon does not reach), so it goes to the refine stage.

~~~dg-flow Filter then refine: the spatial index narrows 100,000 fences to a few bounding boxes; exact tests run only on those.
Location update -(R-tree on bounding boxes)-> *3 candidate fences -(exact point in polygon)-> 1 or 2 fences contain it -(compare with last result)-> Enter and exit events
~~~

## The filter stage: an R-tree of boxes

Checking 100,000 boxes is still too many per update, so the boxes themselves go into a spatial index. The usual choice is an **R-tree** (lesson 12.01): leaves hold fence bounding boxes; each internal node holds a box that encloses all of its children's boxes. To find boxes containing a point, start at the root, descend into every child whose box contains the point, and collect the leaves you reach.

~~~dg-tree An R-tree over geofence bounding boxes. A point at JFK descends only into the boxes that contain it; whole continents are skipped.
Root: world
  Europe box | skipped
  *North America box
    West coast box | skipped
    *New York area box
      *JFK pickup zone | leaf, candidate
      LaGuardia zone | leaf, box misses
      *Queens surge zone | leaf, candidate
      Manhattan delivery | leaf, box misses
    *New York state tax region | leaf, candidate
  Asia box | skipped
~~~

With a fan-out of about 100 children per node, 100,000 boxes form a tree of 3 levels. A lookup touches a handful of nodes on each level. Boxes can overlap, so occasionally the search descends two or three branches, but the work is still a few dozen box tests instead of 100,000.

### In PostGIS

PostGIS does all of this for you. Create a **GiST** index on the geometry column, and the containment functions use it automatically:

```
CREATE INDEX fences_geom_idx ON geofences USING GIST (geom);

SELECT id, name
FROM geofences
WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(-73.78, 40.64), 4326));
```

`ST_Contains` first performs a **bounding-box comparison that uses any available spatial index** (the filter), then runs the exact geometric test on the survivors (the refine). An internal variant, `_ST_Contains`, skips the index step, which is useful only for debugging. The same applies to `ST_Intersects`, `ST_Covers` and `ST_DWithin`.

## Semantics: contains versus covers

Edge cases at the boundary are where geofencing bugs live. `ST_Contains(A, B)` is true when no point of B lies in A's **exterior** and at least one point of B's **interior** lies in A's interior. A consequence that surprises people: **a polygon does not contain a point exactly on its boundary.** If a point on the fence line should count as inside, use `ST_Covers`, which is more inclusive.

For moving vehicles with GPS noise of several metres, a point lying exactly on a line is rare. It matters for:

- **Adjacent fences sharing an edge** (two delivery areas that meet along a street): a point on the shared edge is in neither under `ST_Contains`, and in both under `ST_Covers`. Pick deliberately, and if each point must belong to exactly one zone, add a tie-break rule.
- **Unit tests**, which love points on corners.
- **Snapped coordinates**, such as addresses geocoded to a street centreline that also forms a zone boundary.

## A faster alternative: precompute cells

The R-tree approach does a little index work and a few exact tests per update. At 250,000 updates per second that is fine on a cluster, but you can do better by moving the expensive work from **per update** to **per fence**. Fences change rarely (a few edits a day); updates arrive constantly. So precompute.

1. **Cover each geofence with grid cells** at a chosen resolution, using H3, S2 or geohash (lessons 12.02 and 12.03). S2's region coverer and H3's `polygonToCells` exist for exactly this.
2. **Classify each cell** for each fence: **interior** (the whole cell is inside the fence) or **boundary** (the fence's edge passes through the cell).
3. **Build a map** from cell ID to a short list of (fence ID, interior or boundary). Cells touched by no fence are simply absent.
4. For each location update, **compute the point's cell ID** (a few arithmetic operations) and look it up in the map.

The lookup has three possible outcomes:

- **Absent**: no fence contains the point. Done.
- **Interior entries**: those fences contain the point. No geometry needed.
- **Boundary entries**: run the exact point-in-polygon test against just those fences.

| Cell type for a fence | What the update does | Cost |
| --- | --- | --- |
| Not in the map | Nothing: no fence here | One hash lookup |
| Interior | Report the fence directly | One hash lookup |
| Boundary | Exact test on that fence | Lookup plus one O(n) test |

Most of the area of a typical fence is interior, and most of the Earth is in no fence at all, so the vast majority of updates are answered by **one hash lookup**. Only points near fence edges pay for geometry.

~~~dg-bars Approximate work per location update for 100,000 fences. Precomputed cells answer most updates with one lookup.
Test every polygon | 100000 | 100,000 polygon tests
R-tree filter then refine | 30 | about 30 box tests plus 2 exact tests
*Cell map, interior cell | 1 | 1 hash lookup
Cell map, boundary cell | 2 | 1 lookup plus 1 exact test
~~~

### Choosing the resolution

The resolution trades memory against how often you hit boundary cells.

- **Coarser cells** mean fewer entries in the map, but a bigger share of each fence is boundary cells, so more updates need exact tests.
- **Finer cells** mean almost every update lands in an interior or empty cell, but the map grows with the fence's area divided by the cell area.

A common compromise is **mixed resolution**: big cells for the interior of a fence and small cells along its edge, exactly what S2's coverer and H3's compaction produce. The lookup then checks the point's cell at a few resolutions (or walks its parents).

<div class="ex">
<div class="lab">Worked example</div>

**sizing a cell map**

Say 100,000 fences averaging 5 km² each (500,000 km² in total, overlapping in places). At H3 resolution 9 (about 0.105 km² per cell) that is about 4.8 million cell entries. At around 40 bytes per entry (64-bit cell ID, fence ID and a flag in a hash map) the map is about 200 MB: small enough to keep in memory on every geofence server.

Boundary share: a 5 km² fence of roughly circular shape has a perimeter of about 8 km. Resolution-9 cells are about 0.35 km across, so the edge passes through roughly 25 to 45 cells, out of about 48 cells in total for the fence. At this resolution half or more of the fence's cells are boundary cells.

Go one resolution finer for the edges only (resolution 10, about 0.13 km across): keep the roughly 20 interior resolution-9 cells and split the edge band into about 175 resolution-10 cells, of which only some 60 to 100 still touch the boundary. Those thin boundary cells cover only about a tenth of the fence's area, so a point picked at random in the fence has roughly an 85% chance of landing in an interior cell and needing no polygon test at all.

</div>

## Further optimisations

- **Check only on change.** A vehicle that is still in the same interior or empty cell as last time cannot have crossed a fence boundary. Cache each vehicle's last cell and last result, and skip the work when nothing changed. At city speeds, a car stays in a resolution-9 cell for tens of seconds, which is many updates.
- **Emit enter and exit events, not states.** Compare the new set of fences with the previous set for that vehicle and publish only the differences ("entered JFK pickup zone"). Downstream consumers then process a tiny event stream instead of every update.
- **Add hysteresis.** GPS jitters by 5 to 20 m. A vehicle parked on a fence line can flip in and out every few seconds, producing a flood of events. Require the vehicle to be inside for two consecutive updates, or 20 m past the line, before declaring an entry, and similarly for exits.
- **Simplify polygons.** A city boundary traced at 2,000 vertices can often be simplified to 200 with an error of a few metres, making exact tests ten times faster. Buffer slightly outward or inward if the product prefers false positives or false negatives.
- **Shard by region.** Keep each city's fences and cell map on the servers that process that city's vehicles, routed by a coarse cell of the update.

~~~dg-state One vehicle against one fence, with hysteresis so GPS jitter on the boundary does not produce a storm of events.
[*] -> Outside
Outside -> Entering : first update inside
Entering -> Outside : next update outside
Entering -> *Inside : second update inside, emit ENTER
Inside -> Leaving : first update outside
Leaving -> Inside : next update inside
Leaving -> Outside : second update outside, emit EXIT
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an airport zone, both ways**

A car at JFK sends an update. **R-tree approach**: the search descends to three leaf boxes containing the point: the JFK pickup zone, the Queens surge zone and the New York state tax region. Exact tests (about 300 edges in total) confirm the car is inside all three. About 30 box comparisons and 3 polygon tests.

**Cell approach**: the point's resolution-9 cell is looked up in the map. It is an interior cell of the tax region and of the surge zone (no test needed) and a boundary cell of the pickup zone (one exact test on its 80 edges). Compared with the vehicle's previous result (tax region and surge zone only), the set gained one fence, so after the second consecutive inside update the service emits `ENTER jfk_pickup` and the driver app shows the queue instructions.

</div>

## Keeping fences up to date

Fences do change: a new delivery area launches, an event closes some streets. Because the cell map is derived data, treat it like any other index:

- Store the polygons in a durable database (PostGIS is a good fit) as the **source of truth**.
- On a change, recompute that fence's covering and publish a diff to the geofence servers, or rebuild the whole map periodically and swap it in atomically.
- Version the map, so you can tell which fence definitions produced a given event when investigating a complaint.

## Pitfalls

- **Testing everything.** The naive loop works in a demo with 20 fences and collapses with 20,000.
- **Ignoring boundary semantics.** Decide whether "on the line" is inside, and be consistent across services.
- **No hysteresis.** Fence-entry notifications that fire five times while a driver waits at a kerb.
- **Longitude wrap and big polygons.** A fence crossing the 180° meridian, or a very large one where edges should follow great circles, behaves wrongly in planar geometry. Use geography types or S2 for those.
- **Stale cell maps.** Changing a polygon in the database but not regenerating its covering means the fast path keeps answering with the old shape.

## In an interview

Say "filter and refine" early. Describe the bounding-box R-tree (or PostGIS doing it for you) for a moderate load, then show you can go further by precomputing a cell-to-fences map so most updates are a single hash lookup, with exact tests only for boundary cells. Add the product details that show experience: emit enter and exit events, cache per-vehicle state, add hysteresis for GPS noise, and choose contains or covers deliberately for shared edges.

<div class="take">
<div class="lab">Key takeaway</div>

never test every polygon. Filter with a spatial index on bounding boxes (PostGIS ST_Contains does this automatically) and run exact point-in-polygon tests only on the few candidates, or precompute a cell-to-fences map so most updates resolve with a single lookup and only boundary cells need exact tests. Emit enter and exit events with hysteresis, and mind boundary semantics: ST_Contains excludes the boundary, ST_Covers includes it.

</div>

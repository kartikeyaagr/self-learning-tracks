---
id: "12.06"
course: "system-design-101"
module: "12"
slug: "ray-casting-point-in-polygon"
title: "Ray casting point in polygon"
keyIdeas: ["Ray casting tests if a point is inside a polygon: draw a ray from the point in one direction and count how many polygon edges it crosses.", "Odd number of crossings = inside; even = outside. Each time the ray crosses an edge it switches between inside and outside, and far away it's outside.", "Edge cases: the ray passing exactly through a vertex, and points on an edge. Implementations use half-open rules to count vertices consistently.", "Runs in O(number of edges) per test, so pair it with an index to limit how many polygons you test."]
question: "Why does an odd number of crossings mean \"inside\"?"
answer: "A simple polygon's boundary divides the plane into an inside and an outside (the Jordan curve theorem), and the only way to move from one region to the other is to cross the boundary. The far end of the ray, beyond every edge, is certainly outside. Walking back along the ray from there to the point, each edge crossing flips you between outside and inside, so after an even number of crossings you are back outside and after an odd number you are inside, whatever the polygon's shape. Vertices lying on the ray are handled with a half-open rule so each true crossing is counted exactly once."
sources: [{"title": "Wikipedia: Point in polygon, ray casting", "url": "https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm", "note": "The algorithm."}]
---

## The question behind every geofence

Lesson 12.05 narrowed 100,000 geofences down to two or three candidates per location update. The **refine** step still has to answer, exactly: **is this point inside this polygon?** That is the **point-in-polygon** problem. It sounds easy when you look at a drawing, because your eyes see the shape. A computer only has a list of vertices, and the polygon might be concave, have hundreds of edges and wiggle in and out like a coastline.

The standard answer is a short algorithm called **ray casting**. This lesson explains how it works, why it is correct, how to implement it so that awkward cases do not break it, and when you need the alternative, the winding number.

## The algorithm

From the point, draw a **ray**: a half-line that starts at the point and goes off to infinity in one fixed direction. By convention it goes horizontally to the right (towards positive x). Then count how many times the ray **crosses the polygon's edges**.

- **Odd** number of crossings (1, 3, 5, ...): the point is **inside**.
- **Even** number (0, 2, 4, ...): the point is **outside**.

That is all. It is also called the **even-odd rule** or the **crossing number** algorithm. It works for any **simple polygon**, meaning one whose edges do not cross each other, whether convex or concave.

## Why odd means inside

The correctness rests on one theorem and one observation.

**The Jordan curve theorem** says that a simple closed curve in the plane (and a simple polygon's boundary is one) divides the plane into exactly **two regions**: a bounded **inside** and an unbounded **outside**. Every point not on the curve is in exactly one of them, and **you cannot get from one region to the other without crossing the curve**.

**The observation**: the polygon is bounded, so if you go far enough along the ray, past the right-most vertex, you are certainly **outside**.

Now put them together. Walk along the ray **backwards**, from the far end (known to be outside) towards the point you are testing.

1. You start outside.
2. Between crossings you stay in the same region, because crossing the boundary is the only way to change region.
3. Each time you cross an edge, you switch: outside to inside, or inside to outside.
4. When you reach the test point, you have switched once per crossing. After an even number of switches you are back where you started: outside. After an odd number, you are inside.

So the **parity** of the crossing count tells you the region. The direction of the ray does not matter, and neither does the shape: a concave polygon may make the ray cross many times (in, out, in, out, in), but each crossing still flips the state exactly once.

~~~dg-state Walking back along the ray from far away: every edge crossing flips the region. Three crossings means the test point is inside.
[*] -> Outside
Outside -> Inside : cross edge at x = 6
Inside -> Outside : cross edge at x = 4
Outside -> *Inside : cross edge at x = 2, reach the point
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a concave U shape**

Take the U-shaped polygon with vertices (0,0), (6,0), (6,6), (4,6), (4,2), (2,2), (2,6), (0,6). It is a square block with a notch cut out of the top middle, between x = 2 and x = 4, down to y = 2.

**Point P = (1, 4)**, in the left arm. Its ray goes right along y = 4. It crosses the notch's left wall (x = 2, from (2,2) to (2,6)), the notch's right wall (x = 4, from (4,6) to (4,2)) and the outer right wall (x = 6). The outer left wall at x = 0 is to the left of P, so it does not count. **Three crossings: odd, inside.** Correct.

**Point Q = (3, 4)**, inside the notch, which is outside the polygon. Its ray crosses x = 4 and x = 6. **Two crossings: even, outside.** Correct, even though Q is surrounded by the polygon on three sides.

**Point R = (3, 1)**, in the bottom bar below the notch. The ray along y = 1 crosses only the right wall at x = 6 (the notch walls only span y = 2 to 6). **One crossing: inside.**

</div>

~~~dg-timeline The ray from P = (1, 4) through the U shape, read from the point outwards. The region alternates at each crossing.
x = 1 | *Start at P, region unknown
x = 2 | Cross notch left wall
x = 4 | Cross notch right wall
x = 6 | Cross outer right wall
x = 7 and beyond | Certainly outside, so 3 crossings means P is inside
~~~

## Implementing it

For each edge from (x1, y1) to (x2, y2), with the test point at (px, py) and the ray going right along y = py:

1. **Does the edge straddle the ray's line?** It can only cross the horizontal line y = py if one endpoint is above it and the other is not.
2. **Where does it cross?** Solve for the x coordinate of the edge at height py: `x = x1 + (py − y1) × (x2 − x1) / (y2 − y1)`. This is just linear interpolation along the edge.
3. **Is that crossing to the right of the point?** If x > px, the ray hits this edge: flip the `inside` flag.

```
def inside(px, py, poly):
    n = len(poly); c = False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > py) != (y2 > py):            # straddles the ray's line
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x > px:
                c = not c
    return c
```

The cost is **O(n)** for n edges: one comparison per edge, plus a multiply and divide for the few that straddle. A 200-vertex delivery zone takes well under a microsecond. Because the straddle test guarantees y1 ≠ y2, the division never divides by zero.

Two cheap speed-ups are common in production:

- **Bounding box first.** Check the point against the polygon's min and max coordinates before looping. Most candidates from a coarse filter fail this test.
- **Avoid the division.** Compare `(px − x1) × (y2 − y1)` with `(py − y1) × (x2 − x1)`, flipping the comparison when y2 < y1. Same answer, no floating-point division, and slightly more robust.

## The edge cases

Ray casting is short, but most buggy implementations fail on the same handful of cases.

### The ray passes exactly through a vertex

Imagine the ray hitting the exact point where two edges meet. A naive implementation counts both edges, giving two crossings where there should be one, or one where there should be none. Two situations look identical locally but need different answers:

- **Passing through**: the boundary goes from below the ray to above it via the vertex. That is a real crossing: count **once**.
- **Touching**: the boundary comes down to the ray and goes back up (a peak or a valley touching the line). That is not a crossing: count **zero** (or two, which has the same parity).

The fix is a **half-open rule**: treat every vertex that lies exactly on the ray as if it were **slightly above** it. Then an edge is counted only if one endpoint is strictly above py and the other is at or below. The condition `(y1 > py) != (y2 > py)` in the code does exactly this. A vertex shared by two edges is then counted for exactly one of them when the boundary passes through, and for neither or both when it only touches.

<div class="ex">
<div class="lab">Worked example</div>

**a ray through two vertices of a diamond**

Diamond with vertices (2,0), (4,2), (2,4), (0,2). Test point P = (−1, 2), clearly outside on the left. The ray along y = 2 runs exactly through the vertices (0,2) and (4,2).

Apply the rule with py = 2. Edge (2,0)→(4,2): is 0 > 2? No. Is 2 > 2? No. Same side, skip. Edge (4,2)→(2,4): 2 > 2 no, 4 > 2 yes. Straddles; it meets the line at x = 4, which is right of −1: **crossing 1**. Edge (2,4)→(0,2): 4 > 2 yes, 2 > 2 no. Straddles; meets the line at x = 0 > −1: **crossing 2**. Edge (0,2)→(2,0): no, no. Skip.

Two crossings: **outside**. Correct. A naive version counting every edge that touches y = 2 would have counted all four edges and also got an even answer here, but would fail for P = (1, 2), inside the diamond: the rule gives exactly one crossing (at x = 4), so inside.

Now the tangent case: P = (−1, 4), level with the top vertex (2,4). Edge (4,2)→(2,4): 2 > 4 no, 4 > 4 no. Edge (2,4)→(0,2): no, no. **Zero crossings: outside.** The ray grazed the peak and correctly did not count it.

</div>

### Horizontal edges

An edge lying exactly along the ray (both endpoints at y = py) never straddles under the half-open rule, so it is skipped. Its two end vertices are handled by the neighbouring edges, which gives the right parity.

### Points exactly on the boundary

For a point lying on an edge, the answer depends on implementation details and rounding, and may be "inside" for some edges and "outside" for others. If boundary points matter (shared edges between zones, as in lesson 12.05), test for them explicitly first: check whether the point is within a small tolerance of any edge, and return "on the boundary" so the caller can apply its own rule (like `ST_Covers` versus `ST_Contains`).

### Floating point

Near an edge, rounding errors in the intersection formula can flip the answer for points a few nanometres from the line. For geofences with GPS noise of metres, this does not matter. For exact geometry (CAD, map overlays, legal boundaries), robust libraries use **exact predicates**: orientation tests computed with enough precision to always give the correct sign.

~~~dg-matrix The edge cases of ray casting and how a correct implementation handles each.
Case | Naive count | Half-open rule | What to do
Ray through a pass-through vertex | no, counts 2 | yes, counts 1 | Rule handles it
Ray grazes a peak or valley | partial, counts 2 | yes, counts 0 | Rule handles it
Horizontal edge on the ray | no, may divide by zero | yes, skipped | Rule handles it
Point exactly on an edge | maybe | maybe | Test for boundary first
Rounding near an edge | maybe | maybe | Tolerance or exact predicates
~~~

## Polygons with holes and multiple parts

Real geofences are often **multipolygons**: several separate pieces (an airport with two terminals) or a shape with **holes** (a city zone that excludes a park). Ray casting extends naturally: run the same loop over **every ring** (outer boundaries and hole boundaries) and count all crossings together. A point in a hole crosses the hole's boundary once more than a point just outside the hole, so its parity comes out even: outside. No special case is needed.

## The alternative: the winding number

The **winding number** of a point with respect to a closed curve counts how many times the curve travels **around** the point, anticlockwise counting +1 and clockwise −1. The point is inside if the winding number is **non-zero** (the **nonzero rule**). It can be computed as cheaply as the crossing number: walk the edges, and for each edge that crosses the ray, add +1 if it crosses upwards and −1 if downwards. A 2001 formulation showed it needs no trigonometry and runs as fast as crossing counting.

For simple polygons, both rules always agree. They differ for **self-intersecting** polygons, where the edges cross each other.

~~~dg-compare The two inside rules on a five-pointed star drawn as one self-intersecting path. They agree on simple polygons and differ here.
[Even-odd rule]
Ray from the centre -> crosses 2 edges -> even
- Centre pentagon counts as outside
+ Matches crossing count directly
[Nonzero winding rule]
Ray from the centre -> 2 crossings, same direction -> winding 2
+ Centre pentagon counts as inside
+ Matches how most people read the shape
~~~

A pentagram drawn as one continuous path has a pentagon in the middle. A ray from the centre crosses two edges, so even-odd says **outside** (a hollow star), while both crossings go the same way, so the winding number is 2 and nonzero says **inside** (a solid star). SVG's `fill-rule` exposes exactly these two options, `evenodd` and `nonzero`, and graphics libraries do the same.

For geofencing, self-intersecting polygons are almost always **data errors** (a zone someone drew with a twisted edge). The right response is usually to **validate polygons on input** (PostGIS has `ST_IsValid` and `ST_MakeValid`) rather than to pick a rule.

## Beyond a single test

Ray casting is O(n) per point. If you test **one polygon against millions of points**, preprocessing helps: split the polygon's bounding box into horizontal **slabs** at every vertex's y coordinate, sort the edges within each slab, and a point test becomes a binary search, O(log n). Or overlay a grid and mark cells that are fully inside, fully outside or on the boundary, which is the cell-covering idea from lesson 12.05 applied inside one polygon. The general lesson: do expensive work once per shape when the shape is tested many times.

Also remember that ray casting works on a **flat plane**. Latitude and longitude are not a plane, but for polygons a few kilometres across the error is negligible. For huge polygons, ones crossing the 180° meridian, or ones near the poles, use a spherical library (S2, or PostGIS geography types).

## In an interview

Give the rule (odd inside, even outside), then the reason in two sentences: the boundary separates inside from outside, the far end of the ray is outside, and each crossing flips the region. Mention the vertex problem and the half-open fix, the O(n) cost, and that you only run it on candidates after a bounding-box filter. If the interviewer asks about self-intersecting shapes, explain the winding number and the pentagram.

<div class="take">
<div class="lab">Key takeaway</div>

cast a ray from the point and count edge crossings; odd means inside. By the Jordan curve theorem the boundary splits the plane into inside and outside, every crossing flips between them, and the far end of the ray is outside, so parity gives the answer. Handle vertices on the ray with a half-open rule, check boundary points explicitly, count all rings for holes, and use the winding number for self-intersecting polygons.

</div>

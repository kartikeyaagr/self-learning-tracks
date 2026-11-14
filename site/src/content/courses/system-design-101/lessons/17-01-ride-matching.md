---
id: "17.01"
course: "system-design-101"
module: "17"
slug: "ride-matching"
title: "Ride matching"
keyIdeas: ["Ride matching: drivers send location every few seconds to a location service that keeps an in-memory geo index (Redis GEO, H3 cells).", "A rider request finds nearby available drivers, ranks them by ETA (not straight-line distance), and offers the ride to one or a few drivers.", "Prevent double-matching: when offering a ride, atomically lock the driver (conditional update or Redis lock with TTL). If the driver declines or times out, release and try the next.", "Partition by city or region, since matching is local.", "Trip state machine and payments live in durable storage; location updates are high-volume and short-lived."]
question: "How do you stop two riders being matched to the same driver?"
answer: "Lock the driver before offering. The matcher runs an atomic SET lock:driver:ID ride NX EX 10 in Redis: only one matcher can create the key, so only one rider's offer goes to that driver, and any other matcher moves on to the next candidate. If the driver does not answer within 10 seconds, the key expires and the driver is free again without any cleanup job. When the driver accepts, the assignment is written to the database conditionally (only if the driver has no active ride and the ride is still in the matching state), so even if Redis loses a lock or a paused matcher acts on an expired one, the database stays the final guard against double assignment."
sources: [{"title": "Hello Interview: Uber breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/uber", "note": "Location updates, matching, locking drivers."}]
tags: ["Design"]
---

## What we are building

Ride matching in the style of Uber or Lyft. A rider opens the app, sees a price, taps "request", and within seconds a nearby driver is offered the trip, accepts it, and drives over. Under the friendly map are three hard problems: tracking millions of moving drivers in real time, finding the right driver fast, and guaranteeing that one driver is never given two riders at the same time.

### Functional requirements

- Riders get a **fare estimate** (price and pickup ETA) for a trip.
- Riders **request** a ride and are **matched** with a nearby available driver.
- Drivers **accept or decline** an offer, then navigate to the pickup and destination.
- Out of scope: payments (lesson 16.03), ratings, scheduled rides, pooled rides.

### Non-functional requirements

- Matching finishes (a match or a clear failure) within about **1 minute**.
- **Strong consistency for matching**: a driver must never be assigned two rides at once, and a rider must never get two drivers.
- High throughput at local peaks, such as **100,000 requests from one area** when a stadium empties.
- Location updates may be slightly stale (a few seconds is fine); losing some is acceptable.

## Entities and API

- **Rider** and **Driver** (with vehicle and status: offline, available, offered, on trip).
- **Fare**: an estimated price and ETA for a pickup and destination, valid for a few minutes.
- **Ride**: the lifecycle from request to completion, with rider, driver and timestamps.
- **Location**: a driver's latest position and the time it was reported.

```
POST  /fare {pickup, destination}          -> fare_id, price, eta
POST  /rides {fare_id}                     -> ride_id (status: requested)
POST  /drivers/location {lat, lng}         -> 204
PATCH /rides/{id} {action: accept|decline} -> ride
```

Two security habits matter here. The rider or driver ID always comes from the session token, never the request body, otherwise anyone could accept rides as any driver. And the server must not trust client-supplied fares or timestamps: `POST /rides` takes a `fare_id` that the server created and stored, so a modified app cannot request a ride at a price of 1 cent.

## The ride lifecycle

Before the architecture, pin down the states. Most bugs in matching systems are illegal transitions: a ride assigned twice, or cancelled after it was completed.

~~~dg-state A ride's lifecycle. Each transition is a conditional update, so two actors cannot move it in conflicting directions.
[*] -> Requested
Requested -> Matching : queued
Matching -> *Accepted : driver accepts
Accepted -> In progress : rider picked up
In progress -> Completed : drop-off
Matching -> No driver : 1 minute passes
~~~

A rider can cancel from Requested, Matching or Accepted. Every transition is written as `UPDATE rides SET status='accepted', driver_id=D WHERE id=R AND status='matching'`, so if the rider cancelled a millisecond earlier, the driver's accept finds zero rows updated and is told the ride is gone.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing the location stream**

10 million drivers online at a global peak, each reporting position every 5 seconds: 10,000,000 / 5 = 2 million writes per second.

Each update is about 50 bytes (driver ID, latitude, longitude, timestamp, heading). That is 100 MB per second of incoming data, about 8.6 TB a day if you kept it all.

But matching only needs the latest position per driver: 10 million x about 100 bytes in an index is 1 GB. The whole live state fits in memory on a handful of machines. The challenge is the write rate, not the size.

Ride requests: perhaps 20 million rides a day, about 230 per second on average, but with sharp local spikes (100,000 requests in 15 minutes around one stadium is over 100 per second from a few city blocks).

</div>

## High-level design

~~~dg-flow The main services: locations in an in-memory geo index, matching behind a queue, offers pushed to drivers.
[Locations] Driver app -(every 5 s)-> Location service -(GEOADD)-> *Redis geo index
[Requests] Rider app -(POST /rides)-> Ride service -(enqueue)-> Kafka -(consume)-> Matching service
[Offer] Matching service -(GEOSEARCH nearby)-> Redis geo index -(lock + offer)-> Notification to driver
~~~

- **Ride service**: computes fares (calling a maps service for route distance and time, then applying pricing including surge), stores Fares and Rides, and enqueues ride requests for matching.
- **Location service**: receives driver location updates and answers "which available drivers are near this point?".
- **Matching service**: chooses drivers by pickup ETA, availability, vehicle type and other factors, and runs the offer loop.
- **Notification service**: pushes offers to drivers. While the driver app is in the foreground it usually holds a persistent connection, so offers arrive in under a second; APNs and FCM are the fallback.

## Driver locations

Two million writes per second into a normal relational database, each updating a row and a spatial index, is far too much. And "drivers near this point" needs a spatial index (lessons 12.01 to 12.04). A good fit is **Redis geo commands**:

- `GEOADD drivers:available <lng> <lat> driver:123` stores the driver as a member of a sorted set whose score is a 52-bit geohash of the position. Updating a driver's position is an O(log N) sorted-set update.
- `GEOSEARCH drivers:available FROMLONLAT <lng> <lat> BYRADIUS 3 km ASC COUNT 20` finds the nearest drivers by scanning the geohash ranges that cover the circle and filtering by exact distance (lesson 12.04 covers how).

Practical details:

- **Remove stale drivers.** Keep a second sorted set scored by last-update time, and every few seconds remove drivers whose last update is older than, say, 30 seconds (the app was closed or the phone died). Otherwise matchers offer rides to ghosts.
- **Separate sets by status.** Keep only available drivers in the searchable set; remove a driver when an offer is made or a trip starts, and add them back when it ends.
- **Tolerate loss.** If a Redis node fails, a few seconds of locations are lost, but every active driver sends a fresh update within 5 seconds, so the index heals itself. Replication with automatic failover limits the gap. This is one of the few places where in-memory, eventually consistent state is exactly right.
- **Reduce updates on the client.** Send less often when the driver is parked or the app is idle, more often when moving fast or close to a pickup. Adaptive reporting can cut the write rate by half.
- **Shard by area.** One Redis instance handles perhaps 100,000 to 200,000 writes per second. Split the world into regions (city or H3 cell groups) with their own instances: 2 million writes per second across 20 to 40 shards.

## Never matching two riders to one driver

This is the lesson's central question. Two matching workers, each handling a different rider near the stadium, both search nearby and both pick driver D as the closest. If both send offers, D might accept both, or tap accept on one while the other rider waits on a driver who is already taken. Three ways to prevent it:

~~~dg-compare Three ways to stop two matchers offering the same driver.
[Lock in each process]
Matcher -(local mutex)-> Offer
- Other machines cannot see it
- Crash leaves it held
[Status column]
UPDATE driver where free -> Offer
+ Atomic and durable
- Timeout lives in process memory
[*Redis lock with TTL]
SET NX EX 10 -> Offer
+ Atomic across all matchers
+ Expires on its own
- Depends on Redis being up
~~~

- **A lock inside each matching process** (bad). Workers on different machines do not see each other's locks, so the race remains.
- **A status column in the database** such as `outstanding_offer` (good). A conditional `UPDATE drivers SET offer=ride17 WHERE id=D AND offer IS NULL` is atomic, so only one matcher wins. But who clears it if the driver never responds? A timer in the matcher's memory is lost if that process crashes, leaving D unavailable forever unless a sweeper runs.
- **A distributed lock in Redis with a TTL** (best). Before offering a ride to driver D, the matcher runs `SET lock:driver:D ride-17 NX EX 10`. Only one matcher can succeed, because `NX` sets the key only if it does not exist. The driver has **10 seconds** to respond; if they do not, the key expires on its own and D becomes available again. No sweeper, no timer to lose.

~~~dg-seq Two matchers want the same driver; the atomic lock lets only one make an offer.
Matcher A > Redis : SET lock:driver:D rideA NX EX 10
Redis -> Matcher A : OK
Matcher B > Redis : SET lock:driver:D rideB NX EX 10
Redis -> Matcher B : nil, try next driver
Matcher A > Driver D : offer ride A (10 s)
Driver D > Matcher A : accept
Matcher A > DB : assign D if ride open and D free
~~~

### The database is still the final guard

Redis locks are leases, and leases can fail: a Redis failover can lose a recently written key, and a matcher paused by garbage collection for 12 seconds might still believe it holds a lock that expired (lesson 06.09 covers fencing tokens for exactly this). So the final assignment is a conditional database write: assign D to ride A **only if** D has no active ride **and** ride A is still in `matching`. If that write finds zero rows, the matcher treats it as a failed offer and moves on. Redis keeps the common case fast; the database guarantees correctness in the rare case.

<div class="ex">
<div class="lab">Worked example</div>

**the race traced with timestamps**

18:02:00.000 Matchers A and B both receive requests from riders at the stadium gate and both get driver D as nearest (300 m away).

18:02:00.004 Matcher A's SET NX succeeds. 18:02:00.005 Matcher B's SET NX returns nil; B immediately tries driver E (450 m), whose lock succeeds, and offers E the ride.

18:02:03.200 D accepts. Matcher A runs the conditional assignment; one row is updated. Ride A is accepted; D is removed from the available set.

If instead D had ignored the offer, at 18:02:10.004 the lock key would expire, and at 18:02:10.050 Matcher A (still holding the ride) would lock the next candidate. Total time lost to an unresponsive driver: 10 seconds.

</div>

## Not dropping requests

If a matching instance holds a rider's request only in memory and crashes, the rider waits forever staring at "finding your driver". Put requests on a **queue** (Kafka), and have the matcher commit the offset only after the request is matched or definitively failed. A crash then means another instance picks the request up. Partition the queue **by geography** so each matcher owns an area and its candidates rarely overlap with another matcher's, which also reduces lock contention.

### The offer loop as a durable workflow

Drivers who do not respond are normal. After 10 seconds the matcher must offer the ride to the next driver, then the next, until someone accepts or the minute is up. That is a long-running, multi-step process with timers. Keeping it in one process's memory means a deploy or crash loses every in-flight ride. Model it as a **durable workflow** (Temporal, AWS Step Functions, or a state table plus a timer service): "offer, wait up to 10 s, on timeout or decline try the next" is persisted step by step and survives restarts.

~~~dg-timeline One ride request offered to drivers in turn until one accepts, within the one-minute budget.
0 s | Request queued, 8 candidates found
0.1 s | Lock driver 1, offer sent
10 s | No answer, lock expires
10.1 s | Lock driver 2, offer sent
14 s | Driver 2 declines
14.1 s | Lock driver 3, offer sent
19 s | *Driver 3 accepts, ride assigned
~~~

## Hot spots and surge

When a stadium empties, 100,000 requests arrive from a few city blocks while only a few hundred drivers are nearby. Matching more cleverly does not create drivers. The system responds in several ways:

- **Batch matching**: instead of greedily matching each request to its nearest driver, collect requests for a second or two and solve the assignment for the batch, minimising total pickup time. Greedy matching gives the first rider a good driver and leaves later riders with far worse ones.
- **Widen the radius** progressively as nearby supply runs out.
- **Surge pricing** raises fares in the area, which reduces demand and draws drivers in. Fare estimates must be computed from current supply and demand per area, which is another per-region aggregate updated every few seconds.
- **Queue and inform**: tell riders their position or expected wait rather than spinning.

## Scaling and failure handling

- **Shard by geography.** Riders and drivers in one city mostly interact locally. Map regions to shards (consistent hashing over region IDs spreads load). Queries near a region boundary ask both shards and merge results (scatter-gather).
- **Read replicas** help with ride history and fares; the ride table's writes stay on the primary.
- **Redis failure** loses locks and positions for a few seconds. In-flight offers may be duplicated, but the conditional database write prevents double assignment, and positions refill within 5 seconds.
- **Driver app disconnects mid-offer**: the lock expires and the next driver is tried.
- **Matcher crash**: the workflow engine resumes the offer loop on another worker from its last persisted step.

## Common mistakes

- Writing every location update to the primary database.
- Locks without expiry, or locks only in local memory.
- No conditional database write behind the lock.
- Keeping offer timers in process memory.
- Trusting the client for price or identity.
- Matching purely greedily during surges.

## In an interview

Do the location write-rate calculation early: it justifies the in-memory geo index. Draw the ride state machine, then go deep on the double-assignment question: Redis `SET NX EX`, why it beats the alternatives, and the conditional database write as the final guard. Finish with durability (queue plus workflow) and hot spots. Ask: how long should drivers have to respond, can a driver receive several offers at once, and is pooling in scope?

<div class="take">
<div class="lab">Key takeaway</div>

keep driver locations in an in-memory geo index (Redis GEOADD and GEOSEARCH, sharded by area) that tolerates losing a few seconds of updates. Put ride requests on a queue so nothing is dropped. Before offering a ride, take a Redis lock on the driver with SET NX and a 10-second TTL so only one rider's offer can reach them, back it with a conditional database write, and drive the offer, timeout and retry loop with a durable workflow.

</div>

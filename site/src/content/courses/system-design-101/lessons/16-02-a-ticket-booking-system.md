---
id: "16.02"
course: "system-design-101"
module: "16"
slug: "a-ticket-booking-system"
title: "A ticket booking system"
keyIdeas: ["Ticket booking must never sell one seat twice, and must survive huge demand when popular events go on sale.", "Seat states: available → reserved (held with an expiry, e.g. 10 minutes) → booked.", "Reserve atomically: conditional update (`WHERE status = 'available'`), or a distributed lock with TTL per seat in Redis. Release expired holds by TTL or a sweeper.", "Payment confirms the booking; payment failure or timeout releases the seat.", "For big on-sales, put users in a virtual waiting queue and let them in at a controlled rate.", "Seat maps are read-heavy: cache availability and accept slight staleness, then validate on reserve."]
question: "How do you hold a seat for 10 minutes and release it if unpaid?"
answer: "Reserve with a claim that expires on its own. For example, Redis SET seat:A12 user42 NX EX 600 succeeds for only one buyer and Redis deletes the key after 10 minutes; or a reserved_until column where the claiming UPDATE treats a reservation whose reserved_until is in the past as available, so release is instant without depending on a cleanup job. When payment succeeds, an idempotent webhook marks the ticket sold in the database with a conditional write (only if not already sold), which stays the final guard even if Redis loses a hold. Extend the hold when payment starts so it does not expire mid-checkout. If the user never pays, the key or timestamp simply expires and the next buyer can claim the seat immediately."
sources: [{"title": "Hello Interview: Ticketmaster breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/ticketmaster", "note": "Seat holds, virtual queue."}]
tags: ["Design"]
---

## What we are building

A ticket booking system in the style of Ticketmaster: people browse events, search for them, pick seats on a venue map and pay. Most of the time it is a quiet, read-heavy catalogue. A few times a year it faces something closer to a denial-of-service attack from its own customers: ten million fans arriving at 10:00:00 for a stadium tour with 50,000 seats.

The interesting design problems are therefore **correctness under contention** (never sell one seat twice) and **surviving a stampede** (keep the site up when demand is 200 times supply), while keeping the boring part (browsing) fast and cheap.

### Functional requirements

- Users can **view events** (date, venue, performers, seat map and prices).
- Users can **search** events by keyword, date, location and performer.
- Users can **book tickets**: pick specific seats, hold them briefly, pay, and receive a confirmed booking.
- Out of scope: resale marketplace, dynamic pricing, venue management tools.

### Non-functional requirements

- **Consistency for booking**: no seat may ever be sold twice. If you have to choose, refuse a sale rather than double-sell.
- **Availability for viewing and search**: a slightly stale price or seat count on the browse page is fine.
- **Hot events**: up to **10 million** concurrent users for one on-sale.
- Search p99 under 500 ms. Roughly 100 reads for every write.

This is a deliberate split of the CAP trade-off (lesson 06.04). The catalogue can be served from caches and replicas; the seat claim must be linearisable, meaning there is one agreed order of claims per seat.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**how hot is a hot event?**

A 50,000-seat stadium goes on sale. 10 million people are waiting. If each refreshes the seat map every 5 seconds, that is 2 million seat-map reads per second, and if each tries to reserve, millions of claim attempts on 50,000 rows.

Supply runs out fast: if an average checkout takes 3 minutes and buyers take 4 seats each, 12,500 orders clear the venue. Admitting 5,000 shoppers at a time, 3 rounds (under 10 minutes) would clear it if every admitted shopper bought. In practice many find no seats they like or abandon checkout, so it takes more like 8 rounds, roughly 25 minutes.

So 9.9 million of the 10 million will leave without tickets no matter what you build. The design goal is that they leave quickly and fairly, not that they all get to click on seats.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**normal storage**

100,000 events a year, each with up to 50,000 ticket rows of about 100 bytes: worst case 500 GB of ticket rows a year, in practice far less because most venues are small. This fits comfortably in a sharded or even a single large relational database. Storage is not the hard part.

</div>

## Entities and data model

- **Event**: id, venue_id, performer_ids, start time, description.
- **Venue**: id, address, and a **seat map** (sections, rows, seats, coordinates for drawing).
- **Performer**: id, name, genre.
- **Ticket**: one row per seat per event: event_id, seat_id, price, **status**, and who holds or owns it.
- **Booking**: a user's order: id, user_id, ticket_ids, total, status, payment reference.

The ticket row is the unit of contention. Making it one row per seat means "claim seat A12" is a single-row operation, which databases do atomically and quickly. Sharding by `event_id` keeps all seats of an event on one shard, so multi-seat claims stay inside one database transaction.

~~~dg-tree The data model. Tickets are the rows everyone fights over, so they are kept small and sharded by event.
Event
  Venue | seat map
  Performers
  Tickets | one per seat
    status | available, held, sold
    holder, hold expiry
  Bookings | groups tickets
    payment reference
~~~

## The API

```
GET  /events/{id}                 -> event details (cacheable)
GET  /events/{id}/seats           -> seat map with availability
GET  /search?q=&date=&city=       -> list of events
POST /bookings {event_id, seat_ids}  -> booking_id, hold expires_at
POST /bookings/{id}/pay           -> payment intent client secret
```

The user ID always comes from the auth token. `POST /bookings` creates a booking in state `pending` and places holds on the seats; it returns the time the hold expires so the UI can show a countdown.

## Booking has two steps

The user picks seats and the system **reserves** them for a few minutes while they type card details. Then payment **confirms** the booking. Between those two moments, nobody else may take the seats, and if the user walks away, the seats must come back. So the central question: how do you hold a seat for 10 minutes and release it automatically if unpaid? There are four classic answers, from bad to best.

~~~dg-state A ticket's life. A hold is a claim with a timer; only a successful payment makes it permanent.
[*] -> Available
Available -> *Held : claim succeeds
Held -> Sold : payment confirmed
Held -> Available : hold expires
Sold -> Available : refund or cancel
~~~

### Option 1: a long database lock (bad)

Start a transaction, `SELECT ... FOR UPDATE` the seat rows, and keep the transaction open while the user checks out. It feels natural: the lock really does stop others. But it ties up a database connection per shopper for minutes (a pool of 200 connections supports 200 shoppers), invites deadlocks when two users lock overlapping seats in different orders, and a client that crashes or closes the tab leaves the lock held until some timeout. Never hold database locks across human think time.

### Option 2: a status column plus a cleanup job (good)

Add `status` (available, reserved, sold) and `reserved_until` to each ticket. Reserving is a short transaction that only succeeds for seats that are still free:

```
UPDATE tickets SET status='reserved', reserved_until=now()+interval '10 min', user_id=42
WHERE event_id=9 AND seat_id IN ('A12','A13') AND status='available';
```

The database checks the condition and changes the row atomically (lesson 03.06 covers why a single conditional UPDATE does not race). If fewer rows were updated than requested, someone else got a seat, so roll back. A cron job sets expired reservations back to `available` every minute. The flaw: release happens only when the job runs, so a seat whose hold expired at 10:10:01 stays unavailable until the 10:11:00 run. For a hot event with buyers waiting, a minute of dead stock on thousands of seats matters.

### Option 3: expiry checked in the query (great)

Do not rely on the job for correctness. Treat a ticket as claimable if it is available **or** its reservation has expired:

```
UPDATE tickets SET status='reserved', reserved_until=now()+interval '10 min', user_id=42
WHERE event_id=9 AND seat_id='A12'
  AND (status='available' OR (status='reserved' AND reserved_until < now()));
```

The moment a hold expires, the next buyer can claim it. The seat-map read uses the same rule (`reserved_until < now()` shows as free). A cleanup job can still tidy rows, but correctness no longer depends on it. This is a pattern worth remembering in general: **make expiry part of the predicate, not a background process**.

### Option 4: a Redis lock with a TTL (the best fit at scale)

Keep the database for sold tickets only, and hold reservations in Redis, which handles hundreds of thousands of simple operations per second per node:

- `SET seat:evt9:A12 user42 NX EX 600`. **NX** means only set if the key does not exist, so exactly one buyer wins, atomically. **EX 600** means Redis deletes the key after 600 seconds on its own.
- If the SET returns OK, the user holds the seat for 10 minutes. If it returns nil, someone else does.
- For the seat map, the service needs the set of held seats per event. A **sorted set** per event (`ZADD holds:evt9 <expiry_ts> A12`) lets you list current holds with `ZRANGEBYSCORE holds:evt9 now +inf` and drop expired ones with `ZREMRANGEBYSCORE holds:evt9 0 now`.

~~~dg-seq Holding a seat with a self-expiring Redis key, then confirming on payment through a webhook.
User > Booking : reserve seat A12
Booking > Redis : SET seat:A12 u42 NX EX 600
Redis -> Booking : OK (held 10 min)
User > Stripe : pay (card tokenised in browser)
Stripe > Booking : webhook: payment succeeded
Booking > DB : A12 sold if not already sold
Booking > Redis : DEL seat:A12
~~~

~~~dg-compare The four ways to hold a seat, from worst to best fit for a hot event.
[Long DB lock]
Open txn -(FOR UPDATE)-> Wait minutes
- Ties up a connection per shopper
- Deadlocks and orphaned locks
[Status + cron]
UPDATE where available -> Cron frees
+ Short transactions
- Seats stuck until the next run
[Expiry in query]
UPDATE where free or expired
+ Release is instant
+ Database stays the only truth
[*Redis TTL hold]
SET NX EX 600 -> DB write on pay
+ Very high claim rate
+ Expiry built in
- Second system to keep in sync
~~~

## Edge cases that interviewers probe

- **Several seats at once.** A family wants A12 to A15 together. Claim them in a fixed order (sorted seat IDs) so two buyers cannot each hold half. If any claim fails, release the ones you got. In Redis a short Lua script can check and set all four atomically, which avoids partial holds.
- **The hold expires during payment.** The user enters card details at minute 9:50 and the bank's 3-D Secure check takes 30 seconds. Extend the hold when payment starts (for example to 15 minutes total), and make the final database write conditional: `UPDATE tickets SET status='sold', booking_id=... WHERE event_id=9 AND seat_id='A12' AND status <> 'sold'`. If that update finds the seat already sold to someone else, refund automatically and tell the user. The database is the final guard even if Redis was wrong.
- **Redis fails or loses keys.** Holds vanish and some users may lose their seat mid-checkout, but nobody is double-sold, because the conditional database write still decides. You trade a bad user experience in a rare failure for guaranteed correctness.
- **Release on success.** After the sale, delete the Redis key. Deleting only if the value is still `user42` (a compare-and-delete in Lua) avoids deleting a new holder's key in a race.

<div class="ex">
<div class="lab">Worked example</div>

**a race at expiry**

At 10:10:00 Alice's hold on A12 expires and Redis deletes the key. At 10:10:01 Bob claims A12 with SET NX: OK. At 10:10:02 Alice's payment webhook finally arrives.

The webhook handler runs `UPDATE tickets SET status='sold', owner=alice WHERE seat='A12' AND status <> 'sold'`. It succeeds, because Bob has not paid yet. A12 is sold to Alice. Bob's later payment then fails the conditional update, and his booking is refunded with an apology.

Was that the right outcome? It is a policy choice. Many systems instead check at payment start that the hold is still owned by the payer and refuse to take money otherwise. Either way, the database's conditional update guarantees one owner; the policy only decides who.

</div>

## Payment

- The browser sends card details straight to the payment provider (Stripe.js turns them into a token), so your servers never see card numbers and stay out of most PCI compliance scope.
- The server creates a Stripe **PaymentIntent** for the booking amount, tagged with the booking ID.
- Stripe calls your **webhook** when payment succeeds. In one database transaction, the handler marks the tickets `sold` and the booking `confirmed`.
- **Stripe retries webhooks** (it may send the same event several times over hours), so the handler must be idempotent: if the booking is already confirmed, return 200 and do nothing. Store processed event IDs to make this cheap.
- If no webhook arrives (lost or delayed), a reconciliation job queries the provider for pending bookings older than a few minutes. Lesson 16.03 covers payments in depth.

## Hot events: the virtual waiting room

When 10 million people arrive for one concert, letting them all at the seat map means endless failed reservations, a crushed booking service and a terrible experience: you click a seat, it is gone, you click another, gone again. The answer is to **meter admission**:

- Before the seat map, users join a **queue**: a Redis sorted set scored by arrival time (`ZADD queue:evt9 <ts> user42`). Their position is `ZRANK`.
- A process **admits users in batches**, for example 5,000 every minute or whenever the number of active shoppers drops, moving them into an "admitted" set with a TTL (say 15 minutes). Each admitted user gets a signed token.
- The booking service rejects any request without a valid admission token. This is enforced on the server; hiding the button in the UI is not enough.
- Waiting users receive their position over **Server-Sent Events** (lesson 11.02), because updates only flow from server to client and SSE is simpler than WebSockets.

~~~dg-flow A virtual waiting room meters how many people reach the booking step.
[Queue] 10M users -(join)-> *Waiting room -(admit 5,000 per batch)-> Admitted set with TTL
[Book] Admitted users -(reserve)-> Booking service -(SET NX EX)-> Redis holds
~~~

The waiting room page itself must be extremely cheap to serve: a static page from the CDN plus a lightweight position endpoint. Many real systems put it at the edge entirely. Fairness questions come up here too: randomising order among everyone who arrived before the on-sale time (rather than rewarding the fastest bot) is common, and bot detection runs before users are admitted.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the waiting room**

10 million users each hold an SSE connection. At about 10 KB of memory per idle connection, that is 100 GB across the gateway fleet, perhaps 50 servers with 200,000 connections each.

Position updates every 10 seconds means 1 million small messages per second. You can cut that by broadcasting the "now serving" number once to everyone and letting the client compute "you are about N away", which turns per-user updates into one broadcast per batch.

</div>

## Viewing and search

- Event, venue and performer data barely changes: cache it aggressively (read-through with TTLs of minutes to hours) and on the CDN, and scale the stateless event service horizontally.
- **Seat availability** changes quickly during an on-sale, so cache it only for a second or two, or compute it from the Redis hold set plus a cached list of sold seats. A user seeing a seat as free that was taken 2 seconds ago is fine; the claim will simply fail and they pick another.
- **Search**: a B-tree index cannot do "contains this word" efficiently (`LIKE '%taylor%'` scans everything). PostgreSQL full-text search (a `tsvector` column with a GIN index) is a good step; **Elasticsearch** is best for fuzzy matching, typo tolerance and relevance ranking. Feed it from the main database by change data capture so the search index follows the database without dual writes. Cache popular search results briefly in Redis, keyed by every query parameter.

## Common mistakes

- Holding database transactions open while the user types.
- Relying on a cron job for correctness of expiry.
- Trusting the client to say which seats it holds.
- No final conditional write in the database, so a Redis failure double-sells.
- Treating the payment webhook as delivered exactly once.
- Letting 10 million users hit the booking service directly, then trying to autoscale out of it.

## In an interview

State the consistency split early: available browsing, strictly consistent claiming. Then walk the four hold options quickly, land on the expiry-in-predicate or Redis TTL answer, and spend your remaining time on the edge cases (multi-seat, expiry during payment, Redis loss) and the waiting room. Ask: how long should a hold last, may a user hold seats in two events at once, and what is the expected peak for the biggest on-sale?

<div class="take">
<div class="lab">Key takeaway</div>

hold seats with a short, self-expiring reservation, either a Redis `SET NX EX 600` or a `reserved_until` column checked inside the claiming query, never a long database lock or a cron job you depend on. Confirm through an idempotent payment webhook with a conditional database write as the final guard, and put a virtual waiting room in front of hot events so only as many shoppers reach the seat map as there are seats to sell.

</div>

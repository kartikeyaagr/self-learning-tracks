---
id: "08.10"
course: "system-design-101"
module: "08"
slug: "flash-sale-inventory-locking"
title: "Flash sale inventory locking"
keyIdeas: ["Flash sales concentrate huge demand on a tiny inventory in seconds. The main risk is overselling; the second is crashing.", "Atomic decrement prevents overselling: `UPDATE stock SET qty = qty - 1 WHERE id = ? AND qty > 0` and check rows affected. Redis `DECR` with a check works too.", "A single hot row becomes a bottleneck. Options: split inventory into buckets, pre-allocate tokens in Redis, or put buyers in a queue and process in order.", "Reserve with a timeout: hold the item for N minutes, release it if payment doesn't complete.", "Put a waiting room or rate limiter in front so only as many buyers enter as the system can serve."]
question: "100 items, 100,000 buyers in 1 second. How do you avoid overselling?"
answer: "Put a CDN and a virtual waiting room or rate limit in front so only a manageable batch of users reaches checkout, each with an admission token. Decrement stock atomically and conditionally, for example with a Redis Lua script that checks stock > 0 and decrements in one step, or UPDATE ... SET stock = stock - 1 WHERE stock > 0 and check rows affected, so the count can never go below zero (a CHECK constraint and unique order constraint in the database act as a final guard). A successful decrement creates a reservation with a short TTL; an idempotent payment webhook confirms it, and an expired reservation returns the unit to stock for the next person in line. Once stock hits zero, set a sold-out flag so everyone after the 100th gets an immediate response without touching the counter."
sources: [{"title": "Hello Interview: Dealing with contention", "url": "https://www.hellointerview.com/learn/system-design/patterns/dealing-with-contention", "note": "Locks, OCC, queues for hot inventory."}, {"title": "Hello Interview: Ticketmaster breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/ticketmaster", "note": "Reservations with timeouts."}, {"title": "Hello Interview: Flash sale breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/flash-sale", "note": "Full design."}]
---

## The problem in one sentence

You have 100 items and 100,000 people press "Buy" in the same second. You must sell exactly 100: not 101 (overselling means cancelling orders and angry customers, sometimes legal trouble), and not 95 because of lost or stuck reservations. And the system must not fall over while doing it.

There are really two problems here. **Correctness**: never let the stock count go below zero. **Load**: 100,000 requests in a second against one hot row or key, almost all of which will be told "sold out". A good design solves correctness with an atomic operation and solves load by stopping most requests cheaply before they reach that operation.

## The race on the last item

The core bug is a **read-check-write** sequence without coordination:

```
stock = SELECT stock FROM inventory WHERE sku = 'X';   -- reads 1
if stock > 0:
    UPDATE inventory SET stock = 0 WHERE sku = 'X';      -- writes stock - 1
    create_order()
```

Two buyers run this at the same moment. Both read "1 left", both pass the check, both write "0 left", and both are charged for one item. This is a **lost update**: the second write overwrites the first without seeing it. Under normal traffic, the window is microseconds and you might never see it. Under a flash sale, with thousands of requests a millisecond on one row, it happens constantly. The default isolation level of most databases (Read Committed) does not prevent it.

~~~dg-seq The race: both buyers read 1 left before either writes, so both succeed and one item is sold twice.
Buyer A > Database : read stock = 1
Buyer B > Database : read stock = 1
Buyer A > Database : write stock = 0, create order
Buyer B > Database : write stock = 0, create order
Database -> Buyer B : both charged for one item
~~~

## Making the decrement safe

There are three standard ways to make "check then decrement" atomic.

### 1. Atomic conditional update

Do the check and the write in one statement:

```
UPDATE inventory SET stock = stock - 1
WHERE sku = 'X' AND stock > 0;
```

Then check the **rows affected** count. 1 means you got an item; 0 means sold out. The database takes a row lock for the update, so concurrent updates on the row are serialised; each re-evaluates `stock > 0` against the latest value. The count can never go below zero. A `CHECK (stock >= 0)` constraint is a good extra safety net.

The weakness is throughput. Every buyer contends for **one row**. Each update holds the row lock until its transaction commits, which includes a durable log flush, maybe 1 to 5 ms. That caps you at a few hundred to about a thousand decrements a second on that row, no matter how big the database is.

### 2. Pessimistic locking

`SELECT stock FROM inventory WHERE sku = 'X' FOR UPDATE`, then check and update in the same transaction. Correct, but it holds the lock across your application logic, which is even slower, and connections pile up waiting.

### 3. Optimistic concurrency

Read the row with its version number, then `UPDATE ... SET stock = stock - 1, version = version + 1 WHERE sku = 'X' AND version = 17`. If someone else changed it first, zero rows are affected and you retry. Excellent when conflicts are rare. In a flash sale, conflicts are the whole workload: almost every attempt fails and retries, so it's the worst choice here.

~~~dg-matrix Ways to make the stock decrement safe, and how they behave under flash-sale contention.
Approach | Never oversells | Throughput on one item | Behaviour under contention
Read, check, write | no | high | oversells
Conditional UPDATE | yes | partial, ~1k/s per row | serialised on row lock
SELECT FOR UPDATE | yes | no, slower | waits pile up
Optimistic version | yes | no | retry storm
Redis Lua decrement | yes | yes, ~50-100k/s | serialised, in memory
Pre-split tokens | yes | yes | spread over keys
~~~

### Moving the counter to Redis

Redis runs commands on a single thread, one at a time, so every command is atomic. `DECR` alone can go negative, but a small **Lua script** runs atomically as a whole:

```
-- KEYS[1] = stock:X, KEYS[2] = holds:X, ARGV[1] = user id, ARGV[2] = hold expiry
local s = tonumber(redis.call('GET', KEYS[1]))
if s and s > 0 then
  redis.call('DECR', KEYS[1])
  redis.call('ZADD', KEYS[2], ARGV[2], ARGV[1])  -- hold with expiry time as score
  return 1
end
return 0
```

A single Redis node handles roughly 50,000 to 100,000 such script calls a second with sub-millisecond latency, enough for one hot item. The database still records the actual orders. Redis's state must be reconciled with the database: after the sale (or continuously), check that units sold in the database plus active holds plus remaining stock in Redis equals the starting stock. Configure Redis persistence and replication so a failover doesn't lose decrements, or treat the database as the final check (insert orders with a constraint that rejects order 101).

### Splitting the hot key

For bigger sales, split the stock across keys: 10,000 units become 100 buckets of 100, and each buyer is routed to a random bucket (falling back to others when it's empty). Now the load spreads across many Redis shards or database rows. Another form is **pre-generated tokens**: put 100 item tokens in a list and have each buyer `LPOP` one. Getting a token is the reservation; there's nothing to decrement.

## Reserve, then pay

Payment takes seconds to minutes: card entry, 3-D Secure, the bank's response. You can't hold a database lock that long, and you shouldn't sell the item outright before payment. So split it:

1. **Reserve**: atomically take a unit and create a **hold** with an expiry (now plus 5 or 10 minutes).
2. **Pay**: the user completes checkout. The payment provider confirms, usually through a webhook.
3. **Confirm**: an idempotent handler, keyed by the reservation or booking ID, marks the unit sold and the order confirmed in one transaction.
4. **Expire**: if payment doesn't complete in time, the hold lapses and the unit returns to stock.

~~~dg-state A unit's life during a sale. A hold expires on its own unless payment confirms it.
[*] -> Available
Available -> Held : atomic reserve, 10 min expiry
Held -> Sold : payment webhook confirms
Held -> Available : hold expired
Held -> Available : user cancels
Sold -> Refunded : order cancelled later
~~~

### Ticketmaster-style reservations

A ticketing system handling 10 million users for one big event uses the same idea per seat:

- **Reservation with expiry in the database**: a short transaction moves a ticket to RESERVED with `reserved_until = now() + 10 minutes`, but only `WHERE status = 'AVAILABLE' OR (status = 'RESERVED' AND reserved_until < now())`. No background job is needed to free expired holds: the condition itself treats an expired hold as available.
- **Or a distributed lock**: `SET ticket:{id} {user} NX EX 600` in Redis. `NX` (only set if not exists) makes it atomic, so exactly one user gets the key; `EX 600` frees it automatically after 10 minutes if they abandon checkout. The database still has the final say when the booking is written.
- **Double booking is prevented** by a transactional database with row locks or optimistic concurrency and a unique constraint on (event, seat) for confirmed bookings.
- **Payment**: the client hands card details directly to the payment provider (such as Stripe) and gets a token; the provider's webhook confirms payment; the handler marks the ticket sold and the booking confirmed in one transaction, keyed by booking ID so a repeated webhook is harmless.

## Admission control: protect the hot path

Correctness is the easy part. The load is what takes systems down. 100,000 requests a second hitting checkout means 100,000 page renders, session lookups, fraud checks and database connections, almost all for nothing.

### The virtual waiting room

For huge events, put a **waiting room** in front:

- Users who arrive before or at the start are placed in a queue, for example a Redis sorted set keyed by event with arrival time (or a random number, for fairness among early arrivals) as the score.
- They hold a lightweight connection (server-sent events or WebSocket) or poll a cheap, cacheable status endpoint showing their position.
- A controller **admits users in batches** at a rate the checkout can handle, say 2,000 a minute, adding them to an `admitted:{event}` set with a signed, expiring token.
- The checkout and reservation services **reject requests without a valid admission token**, so nobody can skip the queue by calling the API directly.
- When stock reaches zero, the waiting room tells everyone still queued "sold out" immediately.

~~~dg-flow The flash sale funnel: most buyers are stopped cheaply before the stock counter, and the database sees only real orders.
[Edge] 100,000 buyers -(CDN-cached page)-> Waiting room -(batches)-> 2,000 admitted
[Reserve] 2,000 admitted -(atomic Lua decrement)-> *100 holds -(sold out flag)-> Rest told sold out
[Pay] 100 holds -(pay within 5 min)-> Payment webhook -> Orders in DB
~~~

### Other load tactics

- **Serve static content from a CDN.** The product page, images and countdown are cached at the edge; only the "Buy" call reaches your servers.
- **Sold-out short circuit.** As soon as stock hits zero, set a flag that every API server (and the CDN) checks first, so the remaining 99,900 requests return "sold out" in a millisecond without touching Redis or the database.
- **Rate limit per user and per device** to stop bots firing thousands of requests. Use CAPTCHAs or a pre-registration lottery for the most contested items.
- **Queue the orders**: instead of decrementing inline, put purchase requests on a queue partitioned by SKU and let a single consumer per SKU allocate units in order. No contention at all, at the cost of an asynchronous "you're in, we'll confirm shortly" response.
- **Pre-scale** everything before a known sale time. Autoscaling reacts in minutes; the sale is over in seconds.

~~~dg-timeline The first minutes of a 100-unit sale with a waiting room and five-minute holds.
11:55:00 | Waiting room opens; 100,000 users queued
12:00:00 | Sale starts; first batch of 2,000 admitted
12:00:01 | *100 holds created; sold-out flag set
12:00:01 | Remaining users told sold out
12:05:01 | 12 holds expire unpaid; units back in stock
12:05:02 | Next 12 in the waiting list get holds
~~~

<div class="ex">
<div class="lab">Worked example</div>

**100 items, 100,000 buyers**

The sale opens at noon. The product page is served from the CDN. A waiting room admits 2,000 users in the first batch. Each "Buy" runs the Redis Lua script: if stock is above zero, decrement it and add a hold with a 5-minute expiry; otherwise return sold out. The first 100 calls succeed within about 50 ms of each other. The 100th success sets `soldout:X = 1`; from then on, API servers check that flag first, and the other 99,900 users see "sold out" without touching Redis or the database.

Eighty-eight buyers pay within 5 minutes; the payment webhook confirms each order idempotently. Twelve don't. A sweeper (or the next reserve call) finds their holds past expiry, increments stock by 12 and clears the sold-out flag, and the waiting room offers those units to the next 12 people in its list. Final result: exactly 100 sold, never 101.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why the plain database version struggles**

Without Redis, every buy runs the conditional `UPDATE` on one row. Each takes about 2 ms holding the row lock, so at most about 500 succeed or fail per second. 100,000 requests arrive in one second; even if 99,900 would fail, they all queue for the same lock. Connection pools fill up within milliseconds, requests time out after 30 seconds, and users retry, adding more load. The correct answer (100 sold) is reached, but the site is down for everyone for minutes. Correctness without admission control is not enough.

</div>

## Common mistakes

- Read-check-write in application code, which oversells under any real concurrency.
- Optimistic locking on the hot row, which turns contention into a retry storm.
- Holding database locks during payment.
- Holds with no expiry, so abandoned carts lock stock forever and the sale ends with unsold items nobody can buy.
- No reconciliation between the Redis counter and the database.
- Forgetting idempotency on the payment webhook, so a repeated webhook creates two orders.

## In an interview

Split the answer into correctness and load. Correctness: atomic conditional decrement (SQL with rows-affected, or a Redis Lua script), reservations with a TTL, idempotent payment confirmation, and a database constraint as the final guard. Load: CDN for static content, a waiting room with admission tokens, a sold-out flag that short-circuits, per-user rate limits, and pre-scaling. Ask how many units, how many buyers, how long checkout takes, and whether fairness (first come first served versus lottery) matters.

<div class="take">
<div class="lab">Key takeaway</div>

overselling comes from unsynchronised read-check-write. Make the decrement atomic and conditional, hold units with expiring reservations until payment confirms them, and put admission control and a sold-out short circuit in front so the hot item isn't hammered by everyone at once.

</div>

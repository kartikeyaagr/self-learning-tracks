---
id: "18.02"
course: "system-design-101"
module: "18"
slug: "stripe-s-idempotency-keys"
title: "Stripe's idempotency keys"
keyIdeas: ["Stripe's API lets clients pass an `Idempotency-Key` header on POST requests.", "The server stores the key with the request's parameters and the result. A retry with the same key gets the saved result instead of a second charge. Reusing a key with different parameters is rejected.", "Keys are kept for at least 24 hours.", "Brandur's design for Postgres adds \"recovery points\": the request is split into atomic phases, each committed with progress, so a crashed request can resume from the last completed phase instead of restarting.", "Clients should retry with exponential backoff and jitter."]
question: "What does Stripe store per idempotency key, and for how long?"
answer: "Per key, scoped to the account by a unique index on account plus key, the server stores the original request's method, path and parameters (to reject a reuse with different parameters), a lock while a request is working on it (a concurrent duplicate gets 409 Conflict), the progress so far as a recovery point, and once finished the response status code and body, which are returned unchanged to every retry. Stripe saves the first result of a request that started executing whether it succeeded or failed, including 500s; requests rejected before executing save nothing. Keys are short-lived: Stripe prunes them after they are at least 24 hours old, after which the same value counts as a new key; a typical implementation lets a reaper delete them at about 72 hours so stranded requests can still be completed after a weekend."
sources: [{"title": "Stripe: Designing robust APIs with idempotency", "url": "https://stripe.com/blog/idempotency", "note": "The source."}, {"title": "Brandur: Implementing Stripe-like idempotency keys in Postgres", "url": "https://brandur.org/idempotency-keys", "note": "Full implementation with recovery points."}]
tags: ["Case"]
tools: ["Stripe"]
---

## The problem

Stripe's API moves money, and its clients talk to it over networks that fail. As lesson 16.03 described, a call can fail before reaching Stripe, partway through, or after it succeeded but before the response arrived. From the client's side those look identical: a timeout or a dropped connection. Retrying blindly can charge a customer twice; not retrying can lose a payment.

Stripe's answer is the **idempotency key**. The client sends a header `Idempotency-Key: <unique value>` on a `POST` request, and reuses the same value whenever it retries that same operation. The server guarantees that, for one key, the operation's side effects happen at most once, and every retry gets back the same response as the first attempt. (GET and DELETE requests are idempotent by definition, so keys only matter for POST.)

This lesson looks inside a production-quality implementation: what is stored per key, how long it is kept, and how a request that calls other systems can crash halfway and still be safely retried.

## What the server stores per key

A full implementation in PostgreSQL uses an `idempotency_keys` table like this:

| Column | Why it is there |
| --- | --- |
| `user_id` + `idempotency_key` | the key is scoped per account; a **unique index** on the pair means only one request can create it |
| `request_method`, `request_path`, `request_params` | what the original request was, so a retry with **different parameters** is caught as a client bug |
| `locked_at` | set while a request is working on this key; a second concurrent request gets **409 Conflict** |
| `recovery_point` | how far the request got: `started`, `ride_created`, `charge_created`, `finished` |
| `response_code`, `response_body` | the final response, returned unchanged to any later request with this key |
| `created_at`, `last_run_at` | for expiry, and for finding abandoned requests |

So the answer to "what does Stripe store per key?" is: the key scoped to the account, a record of the original request's parameters, a lock, the progress so far, and **the response** once there is one.

Two rules about the stored response are worth knowing precisely:

- Stripe saves the status code and body of the first request that **starts executing** for a key, **whether it succeeded or failed**, including 500 errors, and returns that same result for every retry. A retry of a request that failed with a 500 returns the same 500; to try again, the client must use a new key. This prevents a retry from running a half-understood failure a second time.
- Requests **rejected before executing** (failed validation, or a conflict with a concurrent request using the same key) do not save a result, so they can simply be retried with the same key.

~~~dg-tree The idempotency key record, grouped by what each part protects against.
Key record
  Identity | user_id + key, unique
  Request fingerprint | method, path, params
    Guards against reuse with new params
  Lock | locked_at
    Guards against concurrent retries
  Progress | recovery_point
    Lets a retry resume
  *Result | response code + body
    Returned on every retry
~~~

<div class="ex">
<div class="lab">Worked example</div>

**three requests with one key**

10:00:00 A client sends `POST /v1/charges {amount: 5000}` with key `k-1`. The server inserts the key row (unique index succeeds), sets `locked_at`, runs the charge, stores `200 {id: ch_9}` and clears the lock. The response is lost in transit.

10:00:03 The client retries with `k-1` and the same body. The server finds the row with a stored response and returns `200 {id: ch_9}` without touching the card network. One charge.

10:00:05 A buggy client retries with `k-1` but `{amount: 6000}`. The server compares the stored parameters, sees a mismatch, and returns an error saying the key was reused with different parameters. Nothing is charged.

</div>

## Concurrency: the unique index and the lock

Two copies of the same request can arrive at almost the same moment (a client with an aggressive timeout retries while the first attempt is still running). The design handles this with two mechanisms:

- The **unique index** on `(user_id, idempotency_key)` means only one `INSERT` can create the row. The second request's insert fails with a unique violation, so it reads the existing row instead.
- If that row is **locked** (`locked_at` is set and recent), the second request returns **409 Conflict** straight away. The client waits and retries later, by which time the first attempt has finished and stored its response.

A lock older than a timeout (say 90 seconds, longer than any request should take) is considered abandoned: the process that held it probably crashed. A later request may take it over and resume from the recovery point.

~~~dg-state One idempotency key's lifecycle. A key is locked while a request works on it and finished once a response is stored.
[*] -> Created
Created -> Locked : request starts
Locked -> Unlocked : error, lock released
Unlocked -> Locked : retry resumes
Locked -> *Finished : response stored
Finished -> Reaped : older than 24 to 72 h
~~~

## How long are keys kept?

Keys are for **short-term** safety against retries, not a permanent ledger. Stripe removes keys once they are at least **24 hours** old, after which the same key value is treated as new. A sensible scheme for your own system recycles keys after about **24 hours**, and has a **reaper** job that deletes rows at around **72 hours**. The extra two days are deliberate: if a bug strands requests mid-flight on a Friday night, the rows are still there on Monday so engineers can fix the bug and push those requests to completion.

Why not keep keys forever? Storage grows without bound (at millions of requests a day, each row with a stored response body of a few kilobytes), the unique index gets bigger and slower, and no legitimate retry happens days later. Clients retry within seconds or minutes.

~~~dg-flow Life of an idempotency key: created, locked while working, finished with a stored response, reaped later.
[First call] POST with key k-1 -(unique insert, lock)-> *Key row: started -(atomic phases)-> Finished + response
[Retry] POST with key k-1 -(same params)-> Stored response returned
[Later] Reaper job -(older than 24 to 72 h)-> Key deleted
~~~

## Atomic phases and recovery points

The most instructive part of this design is how a request that touches **other systems** recovers after a crash. The worked example is a ride-sharing API ("Rocket Rides") whose `POST /rides` endpoint must:

1. create the idempotency key;
2. create a ride record and an audit record;
3. **charge the rider's card through Stripe**;
4. record the charge on the ride;
5. send a receipt email;
6. store the response.

The charge is a **foreign state mutation**: once it happens, your database cannot roll it back. If you wrapped all six steps in one database transaction and crashed after step 3, the transaction would roll back, the ride would vanish, and the customer would still be charged. So the work is split into **atomic phases**, each one database transaction, with the foreign calls between them:

- `tx1`: insert or lock the key (`recovery_point = started`).
- `tx2`: create the ride and audit records, set `recovery_point = ride_created`.
- Call Stripe to charge, with **its own idempotency key** derived from this one (for example `rocket-rides-k-1`).
- `tx3`: save the charge ID on the ride, set `recovery_point = charge_created`.
- `tx4`: **stage** the receipt email job in a `staged_jobs` table (so the job exists only if this transaction commits), store the response, set `recovery_point = finished`.

Each phase updates the recovery point **in the same transaction** as its work, so progress and work commit together or not at all. A retry with the same key reads the recovery point and **resumes from the next step**, skipping completed ones.

~~~dg-seq A crash after the charge: the retry resumes from the recovery point and does not charge twice.
Client > API : POST /rides, key k-1
API > DB : tx1 started, tx2 ride_created
API > Stripe : charge (key rides-k-1)
Stripe -> API : ch_9 (then API crashes)
Client > API : retry POST /rides, key k-1
API > DB : read recovery_point = ride_created
API > Stripe : charge again, same key
Stripe -> API : same ch_9, no new charge
API > DB : tx3 charge_created, tx4 finished
~~~

<div class="ex">
<div class="lab">Worked example</div>

**crash at every possible point**

Crash during tx2: the transaction rolls back, recovery point is still `started`. The retry creates the ride.

Crash after tx2 but before the Stripe call returns: recovery point is `ride_created`. The retry calls Stripe with the same derived key. If the first call never reached Stripe, it charges now; if it did, Stripe returns the stored charge.

Crash after tx3: recovery point `charge_created`. The retry skips the charge entirely and runs tx4.

Crash after tx4: recovery point `finished`, response stored. The retry just returns the stored response.

In every case: exactly one ride, exactly one charge, exactly one receipt.

</div>

### Why stage the email?

Sending email is also a foreign side effect, and email providers rarely support idempotency keys. Inserting a row into `staged_jobs` inside tx4 makes "the email will be sent" part of the same atomic commit as "the request is finished". A separate enqueuer moves staged jobs to the real job queue after commit. If tx4 rolls back, the job never existed. This is the transactional outbox pattern again.

## Other details

- Phases run at **SERIALIZABLE** isolation (lesson 03.06), the strictest level, so concurrent requests cannot interleave in ways that break invariants. A serialisation failure returns 409, and the client retries.
- On any error, the lock is released so a retry can proceed.
- A **completer** process finds keys that are locked or unfinished and old, and pushes those requests to completion even if the client has given up. Customers who were charged then still get their ride and receipt.
- Calls to services **without** idempotency support that fail ambiguously (a timeout) are usually marked as failed for manual review rather than retried, because a retry could repeat the effect.

## The client side

Clients have a job too:

- Generate a fresh key per **operation** (a random UUID v4 is typical), not per HTTP attempt.
- Keep the key with the pending operation so a retry, even after a client restart, reuses it.
- Retry with **exponential backoff and jitter**: wait roughly 2^n seconds times a random factor, so thousands of clients do not retry in lockstep after an outage.
- Treat 409 as "try again later" and 400-class validation errors as final.
Stripe's own client libraries generate keys and retry automatically for network errors.

## Common mistakes

- Generating a new key inside the retry loop.
- Storing only "seen this key" without the response, so retries cannot return the original result.
- Not checking that a retry's parameters match.
- Wrapping foreign calls inside one database transaction.
- Reading key rows from a replica, where lag hides the stored response.
- Keeping keys forever.

## In an interview

When asked about idempotency, name what is stored (scoped key, request fingerprint, lock, recovery point, response) and for how long (about 24 hours, reaped by about 72), then explain the concurrent-retry case (unique index plus 409) and the crash-midway case (atomic phases with recovery points). That covers almost every follow-up.

<div class="take">
<div class="lab">Key takeaway</div>

per idempotency key, Stripe-style systems store the account and key (unique together), the original request parameters, a lock, a recovery point, and the final status code and body, which are returned on every retry. Keys are kept around 24 hours, with a reaper deleting them at about 72. Long requests are split into atomic phases between foreign calls, each recording its recovery point in the same transaction, so a retry resumes exactly where the last attempt stopped.

</div>

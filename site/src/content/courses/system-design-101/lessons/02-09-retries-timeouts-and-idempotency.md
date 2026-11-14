---
id: "02.09"
course: "system-design-101"
module: "02"
slug: "retries-timeouts-and-idempotency"
title: "Retries, timeouts, and idempotency"
keyIdeas: ["Every remote call needs a **timeout**. Without one, a slow dependency ties up your threads until you fall over too.", "**Retries** fix transient failures but can amplify an outage. Use exponential backoff with jitter, cap the number of attempts, and only retry errors that are retryable.", "Retrying across several layers multiplies load (3 retries at each of 3 layers = 27x). Retry at one layer.", "**Idempotency** makes retries safe: doing the operation twice has the same effect as once. GET, PUT and DELETE are naturally idempotent; POST isn't.", "Idempotency keys: the client sends a unique key with the request. The server stores the result under that key and returns it for repeats instead of re-executing."]
question: "A payment request times out. Why can't the client just retry, and how do idempotency keys fix it?"
answer: "A timeout cannot tell the client whether the request never reached the server or whether the charge completed and only the reply was lost, so a blind retry may charge the customer twice. With an idempotency key, the client generates a unique key for the payment and sends the same key on every attempt. The server stores the key, the request parameters and the result atomically with the charge, and when the key arrives again it returns the stored result instead of charging again (or an error if the parameters differ), so retries become safe."
sources: [{"title": "AWS Builders' Library: Timeouts, retries and backoff with jitter", "url": "https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter", "note": "The canonical guide."}, {"title": "Stripe: Designing robust APIs with idempotency", "url": "https://stripe.com/blog/idempotency", "note": "Idempotency keys for safe retries."}, {"title": "AWS Builders' Library: Making retries safe with idempotent APIs", "url": "https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/", "note": "How AWS implements it."}]
---

## Every remote call can fail in three places

A local function call either returns or throws. A remote call has a third outcome: **you don't know**. To see why, look at where a call to another service can break:

- **On connection**: the request never reached the server (DNS failure, connection refused, TLS handshake timeout). The server did nothing. **Safe to retry.**
- **During processing**: the server received the request, started the work, and something broke (a crash, a dependency error). If all the work was in one database transaction that rolled back, nothing happened and a retry is safe. If the work had side effects outside the transaction (it called a card network, sent an email), state may be half done.
- **On response**: the server finished the work, but the reply was lost (the connection dropped, a proxy timed out, the client gave up a millisecond too early). The client sees a timeout and has **no idea whether the charge went through**.

The third case is why "just retry" is dangerous. **A timeout does not mean the operation failed.** It means you don't know. Retrying a payment that actually succeeded charges the customer twice.

~~~dg-seq The unknown case: a timeout doesn't tell the client whether the server did the work. The idempotency key makes the retry safe.
Client > Server : POST /charges $50, key k1
Server > Server : charge succeeds, store k1 + result
Server -> Client : reply lost
Client > Client : timeout after 2 s
Client > Server : retry, same key k1
Server > Server : k1 already done
Server -> Client : same 201 result, charged once
~~~

## Timeouts

Set a timeout on **every** remote call. Without one, a caller waits forever on a dependency that has hung, holding a thread, a connection and memory. Enough of those and the caller runs out of resources too (the cascading failure in lesson 02.02).

Use two timeouts:

- A **connection timeout** (how long to wait to establish the connection), usually short: tens to a few hundred milliseconds inside a data centre, since a healthy server accepts connections almost instantly.
- A **request timeout** (how long to wait for the whole response), based on how long the operation normally takes.

Choosing the request timeout is a trade-off:

- **Too long**, and threads pile up waiting on a dead dependency, and users wait for an answer that never comes.
- **Too short**, and you time out healthy requests that were just a little slow, retry them, and add load, sometimes to a server that was only slow because it was busy.

A good method: decide what false timeout rate you can accept (say 0.1%), and set the timeout at the dependency's matching latency percentile (p99.9). Add margin for long network paths. Then check what the timeout actually covers: in some clients, DNS lookup and TLS setup sit outside it, in others inside.

A real failure of this kind: a service had a 20 ms timeout on calls to a dependency, comfortably above that dependency's normal latency. After every deployment, error rates spiked, because new processes had to establish fresh TLS connections, and the TLS handshake alone took longer than 20 ms. The fix was to open and warm connections at startup, before taking traffic.

### Deadlines across a chain

In a chain of calls, a fixed timeout per hop doesn't add up correctly. If the user-facing request has 1 s and service A calls B with a 1 s timeout, which calls C with a 1 s timeout, C may keep working long after the user has gone. Better to pass a **deadline** ("this must finish by 12:00:00.950") down the chain, as gRPC does. Each service uses the time remaining, and drops work whose deadline has already passed.

## Retries are selfish

A retry improves your chance of success by using more of the server's capacity. That is fine for rare, **transient** errors: a dropped connection, a single overloaded instance, a leader election. When the server is failing **because it is overloaded**, retries make it worse: every failed request comes back two or three times, so a server struggling at 100% of capacity is now asked for 200% or 300%, and recovery is delayed or prevented.

Retries also **multiply across layers**. If a request passes through five layers and each layer retries 3 times on failure, a failure at the bottom can produce 3⁵ = 243 attempts on the bottom service for one user request.

~~~dg-bars Worst-case attempts reaching the bottom service when every layer makes up to 3 attempts. Retrying at one layer only keeps it at 3.
1 layer retries | 3 | 3 attempts
2 layers | 9 | 9
3 layers | 27 | 27
4 layers | 81 | 81
*5 layers | 243 | 243
~~~

So:

- **Retry at one layer only**, usually the one closest to the failure or the top-level caller, and let the others fail fast.
- **Cap attempts**: a small maximum, typically 2 or 3 in total.
- **Use a retry budget**: a token bucket where each retry costs a token and successful calls slowly refill it. While tokens last, retry freely; when they run out, stop retrying (or retry only at a slow fixed rate). This caps retries at, say, 10% of traffic, so retries can never triple the load. AWS SDKs have used this approach since 2016.
- **Only retry what can succeed.** A 400 Bad Request or 403 Forbidden will fail the same way next time. Retry connection errors, timeouts on idempotent calls, 503 and 429 (respecting `Retry-After`).
- **Only retry what is safe**: idempotent operations, or ones protected by an idempotency key.

~~~dg-tree Should this failed call be retried? Most of the decision is about whether a retry can succeed and whether it is safe.
Call failed
  4xx client error
    429 Too Many Requests | retry after Retry-After
    400, 401, 403, 404, 409 | don't retry: same answer
  Connection never established | retry: server did nothing
  Timeout or 5xx after sending
    Idempotent (GET, PUT) or has a key | *retry with backoff
    Not idempotent, no key | don't retry blindly: check status first
~~~

## Exponential backoff with jitter

Retrying immediately rarely helps: whatever caused the failure is probably still there. Wait longer after each failure: **exponential backoff**, for example 100 ms, 200 ms, 400 ms, 800 ms, up to a cap such as 10 s.

Backoff alone has a problem. When a server has a brief outage, thousands of clients fail at the same moment. With identical backoff schedules, they all retry at the same moment too: at +100 ms, then +300 ms, then +700 ms. Each wave hits the server like a wall, a **thundering herd**, and can knock it over again just as it recovers.

**Jitter** adds randomness to each wait so retries spread out. A common form is **full jitter**: `sleep = random(0, min(cap, base × 2^attempt))`. Instead of 10,000 retries landing in the same millisecond, they spread evenly across the whole backoff window. The same idea applies to cron jobs, cache expiry times and periodic timers: if every client refreshes at exactly midnight or exactly every 60 seconds, add a random offset.

~~~dg-widget backoff Every client fails at once and retries five times with exponential backoff. Without jitter the retries land together in spikes; jitter spreads them out.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**backoff with and without jitter**

10,000 clients lose their connection to a service at the same instant. Each retries with base 100 ms, doubling, cap 5 s.

No jitter: all 10,000 retries arrive within a millisecond at +100 ms. If the service can take 2,000 requests per 100 ms, 8,000 fail and retry together at +300 ms, then again at +700 ms. The service sees periodic walls of thousands of requests and idles in between.

Full jitter: the first retries are spread evenly over 0 to 100 ms, the second over 0 to 200 ms after that, and so on. The service sees a steady stream it can handle, and most clients succeed within their first two or three attempts.

</div>

## Idempotency keys

Timeouts and retries are only half of the story. The fix for the "lost response" case is making the operation **safe to repeat**, even when it isn't naturally idempotent like a POST that creates a charge. The standard design, used by Stripe and many payment APIs:

- The **client generates a unique key** per logical operation (for example a UUID) and sends it in an `Idempotency-Key` header. Every retry of the same operation reuses the same key.
- The server **stores the key together with the result** of the first attempt.
- If the same key arrives again, the server **returns the stored result** instead of doing the work again. If the first attempt is still in progress, it returns a "try again shortly" (409 or 429). If the first attempt broke halfway, the server resumes or completes it from where it stopped.

~~~dg-state The lifecycle of an idempotency key on the server. A retry finds the record and gets the stored answer instead of a second charge.
[*] -> New key
New key -> *Processing : insert key + request hash, start work
Processing -> Completed : work done, response stored
Processing -> Failed, retryable : dependency error, rolled back
Failed, retryable -> Processing : retry with same key
Completed -> Completed : retry returns stored response
Completed -> Expired : retention window ends
~~~

Further rules, as applied in Amazon EC2's `RunInstances` API and similar systems:

- **Store the key and perform the change atomically**, in one database transaction, so you never record a key without doing the work, or do the work without recording the key.
- **Store the original request parameters** (or a hash of them). If the same key arrives with different parameters, return a validation error: the caller has a bug, and silently returning the old result would hide it.
- **Return a response that means the same as the first one**, not an error like "already exists". The caller should not need special handling for retries.
- **A late retry after the resource was deleted** still gets the original resource back (now shown as terminated), not a newly created one.
- **Don't derive the key from a hash of the parameters.** Two identical requests may both be intended: a user really might buy the same \$5 item twice, or launch two identical servers.
- **Keys can't be kept forever.** Keep them for at least as long as clients might retry (Stripe keeps them 24 hours), and long enough to cover the resource's life plus a window after which late retries are assumed gone.

<div class="ex">
<div class="lab">Worked example</div>

**payment API**

The client sends `POST /charges` with `Idempotency-Key: 7f3c...` and `{amount: 2000, currency: "USD"}`.

The server, in one transaction, inserts `(key 7f3c, request hash, status processing)` together with a pending charge record. It then calls the card network (which itself accepts an idempotency reference, so a resumed attempt can't double-charge), and in a second transaction marks the charge succeeded and stores the response (201, charge ID ch_91) on the key's row. The response is lost on the way back.

The client times out after 2 s, waits 200 ms plus jitter, and retries with the same key. The server finds key 7f3c, sees the request hash matches, and returns the stored 201 with charge ID ch_91. The customer is charged once.

If the client had sent the same key with `amount: 3000`, the server would return 422, because the key is tied to the original parameters.

</div>

## Common mistakes

- **No timeout**, or an infinite default in the HTTP client library.
- **Retrying at every layer**, multiplying load during an outage.
- **Retrying non-idempotent writes without a key.**
- **Backoff without jitter**, creating synchronised retry waves.
- **Retrying 4xx errors** that can never succeed.
- **Storing the idempotency key outside the transaction** that does the work.

## In an interview

For any write that matters (payments, orders, bookings), say: "Every call has a timeout. The client retries with exponential backoff and jitter, at most 3 attempts, only at the edge. Writes carry a client-generated idempotency key, stored with the result in the same transaction, so a retry after a lost response returns the original result." That one paragraph covers most of what interviewers look for.

<div class="take">
<div class="lab">Key takeaway</div>

a timeout is not a failure, it is "unknown". Time out every call, pass deadlines down chains, retry sparingly at one layer with exponential backoff, jitter and a retry budget, and make writes safe to repeat with client-generated idempotency keys stored atomically with the result.

</div>

---
id: "08.01"
course: "system-design-101"
module: "08"
slug: "delegation-and-async-work"
title: "Delegation and async work"
keyIdeas: ["Move slow or unreliable work off the request path. The API accepts the request, records it, enqueues a job and returns quickly (often 202 Accepted with a job ID).", "Workers pull jobs from the queue and do the work. Scale them independently of the web tier.", "The client learns the result by polling a status endpoint, a webhook, or a push notification.", "Benefits: fast responses, absorbs spikes, retries without user involvement.", "Costs: more moving parts, eventual results, and you need idempotent workers and a dead-letter queue for jobs that keep failing."]
question: "A request takes 30 s to generate a PDF. How do you redesign it so the user isn't waiting?"
answer: "Turn it into an asynchronous job. The API validates the request, stores a job record with status \"queued\", puts the job ID on a durable queue and returns 202 Accepted with the ID in milliseconds. Workers take jobs off the queue under a lease (visibility timeout), heartbeat while generating the PDF, upload it to object storage and set the status to done with a link. The client polls GET /jobs/{id} every few seconds or receives a WebSocket, webhook or email notification. Failed attempts retry with exponential backoff and go to a dead-letter queue after a few tries, and the job is idempotent (output keyed by job ID) so a retry after a worker crash is safe."
sources: [{"title": "Hello Interview: Long-running tasks pattern", "url": "https://www.hellointerview.com/learn/system-design/patterns/long-running-tasks", "note": "Move slow work off the request path."}, {"title": "microservices.io: Messaging", "url": "https://microservices.io/patterns/communication-style/messaging.html", "note": "Async via brokers."}]
---

## Don't make the user wait for slow work

Start from a contrast. Loading a profile page takes under 100 ms: one or two indexed database reads and a bit of rendering. Generating an annual activity PDF that aggregates millions of rows and renders charts takes 45 seconds or more. Both arrive as ordinary HTTP requests, but they cannot be handled the same way.

Web servers, reverse proxies and load balancers usually time out a request after 30 to 60 seconds. An AWS Application Load Balancer defaults to 60 seconds of idle time; many NGINX setups use 60 seconds; Heroku's router cuts requests at 30 seconds. So the slow request may fail outright even though the server is still working on it. Even when it succeeds, the user stares at a spinner with no progress, refreshes the page out of impatience, and starts a second copy of the same expensive work.

There is a capacity problem too. A web server has a fixed pool of worker threads or processes, say 200. If each PDF request holds a worker for 45 seconds, then just 5 report requests a second keep 225 workers busy, which is more than the whole pool. Fast requests for profile pages now queue behind slow ones, and the whole site slows down because of one feature.

Other everyday examples of work that does not belong in the request path:

- Transcoding an uploaded video into several resolutions (minutes).
- Resizing a photo into thumbnails (hundreds of milliseconds each, times several sizes).
- Sending a newsletter to 2 million subscribers.
- Importing a 500 MB CSV of customer records.
- Calling a slow or unreliable third party, such as a payment provider's fraud check or a shipping label API.

## The pattern: acknowledge now, work later

The fix is called **delegation** (or asynchronous processing): split one request into two phases that run at different times on different machines.

1. **Acknowledge immediately.** The web server validates the request, writes a **job record** (a row describing the work and its status), pushes a message onto a **queue** (Redis, RabbitMQ, SQS) and returns a job ID within milliseconds, usually with HTTP status `202 Accepted`, which means "I've taken this on, but it isn't finished".
2. **Process in the background.** Separate **worker** processes pull jobs from the queue, do the slow work and update the job's status in the database.

A **queue** here is a buffer that holds messages until a consumer is ready to take them. The process that sends messages is the **producer**; the process that takes them is the **consumer** or **worker**. The queue lets the two sides run at different speeds and fail independently.

~~~dg-seq The API answers in milliseconds; the slow work happens on a worker, and the client checks back for the result.
Client > API : POST /reports
API > DB : insert job (queued)
API > Queue : enqueue job id
API -> Client : 202 + job id
Worker > Queue : pull job
Worker > DB : status = running
Worker > Worker : build PDF (45 s)
Worker > DB : status = done + URL
Client > API : GET /jobs/{id}
API -> Client : done + download URL
~~~

~~~dg-compare Synchronous versus asynchronous handling of the same request: the queue removes the slow step from the request path.
[Synchronous]
Client -(waits 45 s)-> API -> PDF builder
- Holds a web worker for 45 s
- Fails when the proxy times out
- A refresh starts the work again
[*Asynchronous]
Client -(202 in 20 ms)-> API -(enqueue)-> Queue -> Worker
+ Web workers freed at once
+ Workers scale separately from the API
- Needs a job table, a queue and status checks
~~~

### What goes in the message

Put the **job ID** in the queue message, not the whole payload. The job record in the database holds the parameters (user, date range, format). This keeps messages small, lets you look a job up by ID from anywhere, and means the database stays the single place that knows the truth about a job. If the payload is large (an uploaded file), store it in object storage such as S3 and put its key in the job record.

### How the client learns the result

The client has four common ways to find out the job finished:

- **Polling**: the client calls `GET /jobs/{id}` every few seconds. Simple and works everywhere. Add a `Retry-After` header so the server can tell clients how long to wait.
- **Push over a live connection**: a WebSocket or server-sent events stream tells the browser the moment the status changes. Better user experience, but you need a component that keeps connections open.
- **Webhook**: for server-to-server APIs, the caller registers a URL and you send an HTTP POST to it when the job ends. Stripe and GitHub use this heavily.
- **Out of band**: for very long jobs (an hour-long export), send an email or in-app notification with a link.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the polling load**

Suppose 2,000 users a minute request reports, each job takes 45 seconds on average, and clients poll every 3 seconds. At any moment about 2,000 × 45 / 60 = 1,500 jobs are in flight. Each in-flight job is polled 1 / 3 times a second, so polling adds 1,500 / 3 = 500 requests a second. Each poll is a primary-key read, perhaps 1 ms, so this is cheap. If you poll every 500 ms instead, the load rises to 3,000 requests a second for no real benefit. Polling at 2 to 5 seconds, or backing off (1 s, 2 s, 4 s, capped at 10 s), is usually right.

</div>

## The job lifecycle

A job record moves through a small set of states. Writing them down early forces you to decide what happens on every failure.

~~~dg-state The lifecycle of a background job. Every arrow is a status update the worker or a sweeper writes to the job table.
[*] -> Queued
Queued -> Running : worker claims it
Running -> Done : success, result URL saved
Running -> Queued : lease expired, retry
Running -> Failed : error, retries left
Failed -> Queued : backoff timer fires
Failed -> *Dead : max attempts reached
Queued -> Cancelled : user cancels
~~~

- **Queued**: the job exists and is waiting for a worker.
- **Running**: a worker has claimed it. The worker writes its own ID and a **lease expiry** (a time after which the claim is no longer valid), for example now plus 2 minutes.
- **Done**: the result is stored, usually as a URL to a file in object storage.
- **Failed**: an attempt threw an error. The job is retried after a delay that grows each time (**exponential backoff**: 10 s, 40 s, 160 s), often with random **jitter** so thousands of failed jobs don't retry at the same instant.
- **Dead**: after a maximum number of attempts (say 5), the job is moved aside, to a **dead-letter queue** or a dead status, so a human can inspect it. Without this, one malformed job retries forever and wastes capacity.

### Leases and crashed workers

What if a worker dies mid-job, from a deploy, an out-of-memory kill or a lost machine? Nobody will ever mark the job done. The usual answer is a **lease** (SQS calls it a **visibility timeout**): when a worker takes a message, it becomes invisible to other workers for a fixed time. If the worker does not acknowledge it in time, the message becomes visible again and another worker takes it. Long jobs must send a **heartbeat** that extends the lease, or the job will be handed out a second time while the first worker is still busy.

This has a big consequence: **a job can run more than once**. The first worker might have been slow, not dead, and both finish. So background jobs must be **idempotent**, meaning running them twice has the same effect as running them once. For the PDF job that is easy: the output file is named after the job ID, so a second run overwrites the same file. For a job that charges a card or sends an email, you need an idempotency key, covered in lessons 02.09 and 08.03.

## Why messaging helps beyond slow requests

Delegation is one use of a broader idea: **asynchronous messaging**. With a synchronous call, the caller and callee are tied together at runtime. Both must be up and fast for the whole request, and a slow callee makes the caller slow. With messaging through a **broker** (the server that stores and forwards messages), the sender only needs the broker to be up.

That buys you:

- **Looser runtime coupling**: the worker fleet can be down for a deploy and the API keeps accepting work.
- **Load levelling**: a burst of 10,000 requests in a minute turns into a queue that workers drain at their steady rate, instead of overloading a downstream database.
- **Independent scaling**: you can run 5 API servers and 50 workers, or the reverse, because they are separate fleets.
- **Retries for free**: a failed message goes back on the queue instead of turning into an error page.

The cost is real. The broker is a new piece of infrastructure that must itself be highly available and monitored. Flows become harder to trace, because a request's work is now spread across processes and time. And you must think about duplicates, ordering and backlogs, which synchronous code never had.

Messaging supports several interaction styles, and it helps to name which one you mean:

| Style | What happens | Example |
| --- | --- | --- |
| One-way notification | sender fires a message, expects no reply | "user signed up" |
| Request with async response | sender gets the answer later via a reply queue or callback | PDF job, webhook |
| Publish/subscribe | one message goes to many independent subscribers | order placed: email, analytics, search |
| Request/response over queues | caller waits on a reply queue with a correlation ID | RPC over RabbitMQ |

~~~dg-flow Where the pieces of a background job system live. The API and the workers never call each other; they share only the queue and the job table.
[Write path] Client -(POST)-> API -(insert)-> Job table
[Hand-off] API -(enqueue id)-> *Queue -(pull)-> Worker pool
[Result] Worker pool -(upload)-> Object store -(URL saved)-> Job table
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the 45-second PDF, end to end**

`POST /reports {year: 2024}` inserts `{id: r_81, user: u_7, status: queued, attempts: 0}`, enqueues `r_81`, and returns `202 Accepted` with `Location: /jobs/r_81` in about 15 ms. The UI shows "Preparing your report" and polls every 3 seconds.

A worker picks up `r_81`, sets `status = running, lease_until = now + 2 min`, and starts building. It heartbeats every 30 seconds to push the lease forward. After 45 seconds it uploads `reports/r_81.pdf` to object storage and sets `status = done, url = ...`. The next poll returns the URL.

Now the failure case: the worker is killed at second 30 by a deploy. No more heartbeats arrive, the lease expires at most 2 minutes later, a sweeper (or the queue's visibility timeout) makes the job available again, and a second worker builds the PDF from scratch. Because the output key is `r_81.pdf`, the retry simply writes the same file. The user waits about 3 minutes instead of 45 seconds, but gets exactly one report.

</div>

## Design choices you should be ready to discuss

- **Deduplicate submissions.** If the user double-clicks, should you build two reports? Accept an `Idempotency-Key` header, or reuse an existing queued job for the same user and parameters.
- **Priorities.** Paying customers or interactive jobs can go on a separate, higher-priority queue with its own workers, so a batch of 100,000 overnight exports doesn't delay one user waiting at a screen.
- **Time limits.** Give every job a maximum run time. A job stuck in an infinite loop should be killed and marked failed, not hold a worker forever.
- **Result expiry.** Generated files cost storage. Expire them after, say, 7 days and let the user regenerate.
- **Progress.** For long jobs, let the worker write a percentage to the job record so the UI can show "40% done" instead of a bare spinner.

## Common mistakes

- **Doing the slow work in a background thread of the web server.** It looks asynchronous, but a deploy or crash loses the work, and nothing retries it. Use a durable queue.
- **Writing the job row and enqueueing as two unsafe steps.** If you enqueue first and the insert fails, a worker gets an ID that doesn't exist. If you insert first and the enqueue fails, the job sits queued forever. Either insert first and have a sweeper re-enqueue old queued jobs, or use the outbox pattern from lesson 08.03, or use the database itself as the queue (lesson 08.06).
- **Assuming exactly-once execution.** Leases, retries and redelivery mean at-least-once. Design for duplicates.
- **No dead-letter handling.** A poison message (one that always fails) retried forever burns capacity and floods logs.
- **Unbounded queues with no alert.** If workers are down for a day, the queue quietly grows. Alert on the age of the oldest message (lesson 08.04).

## In an interview

When a requirement involves anything slower than about a second, or a call to an unreliable third party, say "I'll take that off the request path": return `202` with a job ID, enqueue, process in a worker pool, store status in a job table, notify by polling or push. Then mention the three follow-ups interviewers look for: retries with backoff and a dead-letter queue, idempotency because delivery is at least once, and monitoring queue depth and age. Asking "how fresh does the result need to be, and how long is the user willing to wait?" shows you know the user experience drives the choice between polling, push and email.

<div class="take">
<div class="lab">Key takeaway</div>

for slow work, accept the request, record a job, queue its ID and return 202 at once. Workers with leases do the work in the background, retry with backoff, and must be idempotent because a job can run more than once.

</div>

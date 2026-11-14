---
id: "01.07"
course: "system-design-101"
module: "01"
slug: "how-a-server-handles-many-requests"
title: "How a server handles many requests"
keyIdeas: ["A server handles many requests with either threads/processes (one per request) or an event loop (one thread multiplexing many connections with non-blocking I/O).", "Concurrency = dealing with many things at once (interleaving). Parallelism = doing many things at once (multiple cores).", "Thread-per-request is simple but each thread costs memory (often ~1 MB stack) and context switches, so it struggles at tens of thousands of idle connections. This was the C10K problem.", "Event loops (NGINX, Node.js, Redis) use epoll/kqueue to wait on thousands of sockets in one thread. Great for I/O-bound work.", "The catch: any CPU-heavy or blocking call on the loop stalls every connection. Offload it to a worker pool.", "Modern runtimes mix both: Go's goroutines and Java virtual threads are cheap \"threads\" scheduled onto an event-driven core."]
question: "Why can one NGINX worker handle thousands of connections when a thread-per-request server struggles?"
answer: "Each NGINX worker uses non-blocking sockets and an event loop over epoll, so an idle connection costs only a file descriptor, kernel buffers and a small state machine, and the worker spends CPU only on sockets the kernel reports as ready. With one worker per core there are few context switches. A thread-per-request server spends a full thread (stack memory, scheduling, context switches) on every connection even while it just waits, so it runs out of memory or thread-pool slots at thousands of connections while the CPU sits mostly idle."
sources: [{"title": "Rob Pike: Concurrency is not parallelism", "url": "https://go.dev/blog/waza-talk", "note": "30-minute talk, video and slides on the page."}, {"title": "Inside NGINX: How We Designed for Performance & Scale", "url": "https://blog.nginx.org/blog/inside-nginx-how-we-designed-for-performance-scale", "note": "Event loop vs thread-per-request in a real server."}, {"title": "Node.js: Don't block the event loop", "url": "https://nodejs.org/en/learn/asynchronous-work/dont-block-the-event-loop", "note": "What goes wrong when one request hogs the loop."}, {"title": "Dan Kegel: The C10K problem (optional)", "url": "http://www.kegel.com/c10k.html", "note": "Historical context."}]
---

## The problem

A web server spends most of its life waiting. It waits for the client to finish sending a request over a slow mobile link. It waits for the database to answer. It waits for a disk read, for a downstream service, for the client to read the response. The CPU work per request is often tiny: parsing a few headers, running some business logic, serialising JSON, perhaps 200 µs in total. The waiting can be 100 ms or more.

So the real question is not "how do we compute faster?" but **"how do we keep thousands of mostly idle connections open without wasting resources on them?"** The answer shapes how NGINX, Node.js, Redis, Go and Java servers are built, and it explains limits you will hit in practice: thread pools running dry, one slow call freezing a whole server, memory blowing up at 10,000 connections.

## Concurrency is not parallelism

Two ideas that people blur:

- **Concurrency** is about structure: composing a program out of independently executing pieces, so that while one piece waits, another can make progress. "Dealing with lots of things at once."
- **Parallelism** is about execution: running computations at the same instant on multiple CPU cores. "Doing lots of things at once."

A well-known illustration uses gophers moving a pile of old manuals to an incinerator. Adding more gophers doesn't help on its own; they need carts, and they collide at the pile. Splitting the job into independent steps (load the cart, move it, unload it, bring the empty cart back) gives a concurrent design. Once the structure is right, running more gophers in parallel "falls out" naturally. A concurrent design can run on one core and still be correct; it just becomes easy to parallelise when more cores are available.

~~~dg-timeline Concurrency on one core interleaves two tasks while each waits; parallelism on two cores runs them at the same moment.
[1 core, concurrent] 0 ms | A computes
[1 core, concurrent] 1 ms | A waits on DB, B computes
[1 core, concurrent] 2 ms | B waits, A resumes
[Core 1, parallel] 0 ms | *A computes
[Core 2, parallel] 0 ms | *B computes
[Core 1, parallel] 1 ms | A waits on DB
[Core 2, parallel] 1 ms | B waits on DB
~~~

For servers, concurrency is the important part. Even a single core can serve thousands of clients if it never sits idle while one of them is waiting. Parallelism then multiplies that by the number of cores.

## What a connection actually costs

To compare designs, look at what the operating system keeps for each open connection:

- A **socket** with a **file descriptor** (a small integer the process uses to refer to it) and kernel buffers for incoming and outgoing data, typically a few KB to tens of KB.
- Whatever state the application keeps: the half-parsed request, the user's session, a buffer for the response.

That is cheap. A machine with 16 GB of RAM can hold hundreds of thousands of idle TCP connections. The expensive part is whatever the **concurrency model** adds on top.

## Model 1: a thread (or process) per connection

The classic approach (Apache's prefork and worker modes, traditional Java servlet containers, many Python and Ruby servers) gives each connection its own thread or process. That thread reads the request, calls the database and waits, writes the response, and blocks whenever it has to wait. The code reads top to bottom like a simple script, which makes it easy to write, debug and reason about.

The costs appear with scale:

- **Memory per thread.** Each thread has its own stack. On Linux the default reservation is often 8 MB of virtual memory, and the JVM reserves about 1 MB per thread by default. The physical memory actually touched is smaller, but 10,000 threads still commonly use several GB.
- **Context switches.** When a thread blocks, the kernel saves its registers and loads another thread. A switch costs roughly 1 to 5 µs directly, and more in practice because the new thread finds the CPU caches full of the old thread's data. With thousands of runnable threads, a meaningful share of CPU time goes on switching rather than on work.
- **Scheduler and lock contention.** Thousands of threads competing for the same locks and run queues add overhead that grows with the count.
- **Hard limits.** Thread pools are usually capped (Tomcat's default is 200 threads). Once all threads are busy waiting, new connections queue or are refused, even though the CPU is nearly idle.

This limit was famously called the **C10K problem**: how to serve 10,000 concurrent clients on one machine. In the late 1990s, thread-per-connection servers could not do it comfortably. Event loops were the answer.

## Model 2: an event loop

The alternative is **non-blocking I/O** with an **event loop**. Sockets are put in non-blocking mode, so a read that has no data returns immediately with "would block" instead of putting the thread to sleep. One thread then asks the kernel, through an interface like **epoll** (Linux), **kqueue** (BSD and macOS) or **IOCP** (Windows), "which of these 10,000 sockets are ready to read or write?" The kernel returns only the ready ones. The thread handles each ready event quickly, a few microseconds of work, and loops back to ask again. It never sits waiting for a slow client.

The older `select` and `poll` interfaces did the same job, but they made the kernel scan the whole list of sockets on every call, which costs O(n) per wakeup. epoll keeps the interest list inside the kernel and returns only ready sockets, so the cost is proportional to activity, not to the number of idle connections. That difference is what makes 100,000 mostly idle connections cheap.

~~~dg-compare Thread per connection versus an event loop, for 10,000 connections that are idle most of the time.
[Thread per connection]
10k connections -(one each)-> 10k threads -> mostly blocked
+ Simple, linear code
+ One slow request blocks only its own thread
- GBs of stacks, many context switches
- Pool runs dry while the CPU idles
[*Event loop]
10k connections -(epoll)-> 1 thread per core -> only ready sockets
+ An idle connection costs only a few KB
+ Few context switches
- Any blocking call stalls everyone
- Code is split into callbacks or async steps
~~~

### How NGINX is built

NGINX is the textbook example of this design:

- A **master process** reads the configuration, binds the listening ports (80, 443), and starts the workers. It does no request handling itself.
- It starts **one worker process per CPU core**. Each worker is single-threaded and runs its own event loop.
- Each worker keeps a small **state machine** per connection: where it is in reading the request, talking to the upstream server, sending the response, or waiting for the next request on a keep-alive connection. A worker can manage hundreds of thousands of these.
- Because there are only as many workers as cores, each worker usually has a core to itself, and context switches are rare.
- **Config reloads** have no downtime: the master starts new workers with the new configuration, stops sending them new connections to the old ones, and lets the old workers finish their in-flight connections before exiting. Even upgrading the NGINX binary works this way.

~~~dg-state The small state machine NGINX keeps for each client connection. A worker juggles many of these, advancing whichever one has a ready socket.
[*] -> Reading request
Reading request -> Reading request : more bytes arrive
Reading request -> Waiting on upstream : headers complete, proxy to app
Waiting on upstream -> Sending response : upstream replies
Sending response -> Sending response : socket writable again
Sending response -> *Keep-alive idle : response done
Keep-alive idle -> Reading request : next request
Keep-alive idle -> Closed : idle timeout
~~~

Node.js and Redis follow the same pattern. Node runs your JavaScript on one event loop thread, built on a library called libuv. Redis executes all commands on one main thread, which is why each command is atomic and why one slow command (like `KEYS *` on a large database) blocks every other client.

<div class="ex">
<div class="lab">Worked example</div>

**memory for 50,000 WebSocket connections**

A chat service keeps 50,000 WebSocket connections open per server. Each user sends a message every minute or so, so at any moment almost every connection is idle.

Thread per connection: 50,000 threads × even 256 KB of touched stack = about 12.8 GB, before any application data, plus constant scheduling of 50,000 threads. Most servers would fall over, and the default thread pool limits would refuse connections long before that.

Event loop: each connection costs a socket (kernel buffers of roughly 4 to 16 KB in an idle state) plus perhaps 2 KB of application state. 50,000 × ~20 KB ≈ 1 GB. With 8 cores running 8 loops, each loop manages about 6,000 connections and is woken only for the roughly 800 messages per second the whole server receives.

</div>

## The catch: never block the loop

The event loop's efficiency comes with a golden rule: **don't block the event loop**. One thread serves everyone, so while it is doing any slow piece of work, no other connection on that loop makes progress. Common ways to break the rule:

- **CPU-heavy work whose cost grows with user input**: sorting a user-supplied list of a million items, resizing an image, rendering a large PDF.
- **Regular expressions with catastrophic backtracking (ReDoS)**: a pattern like `/(\/.+)+$/` can take exponential time on a crafted string, freezing the process for seconds or minutes.
- **Synchronous APIs** in a server: `fs.readFileSync`, `crypto.pbkdf2Sync`, or a database driver that blocks.
- **Huge JSON**: parsing or serialising a 50 MB string can take over a second, during which every other request waits.

~~~dg-seq One blocking call stalls every connection the loop serves.
Conn A > Event loop : request
Event loop > Event loop : sync DB call, 200 ms
Conn B > Event loop : request (waits)
Conn C > Event loop : request (waits)
Event loop -> Conn A : response
Event loop -> Conn B : served 200 ms late
Event loop -> Conn C : served 200 ms late
~~~

The fixes:

- **Bound input sizes**: cap request bodies, list lengths and JSON payloads so no single request can demand unbounded work.
- **Partition** long work into small steps that yield back to the loop between them (for example, process 1,000 items, then schedule the next chunk with `setImmediate`).
- **Offload** it to a worker pool or a separate process. Node already offloads file system, DNS lookups and crypto to a libuv thread pool (4 threads by default), but long tasks can clog that pool too, so heavy work often goes to dedicated worker threads or another service.

<div class="ex">
<div class="lab">Worked example</div>

**one slow call, measured**

A Node.js API serves 2,000 requests per second, each needing 0.3 ms of CPU, so the loop is about 60% busy.

A new endpoint calls `crypto.pbkdf2Sync` with 100,000 iterations, taking 80 ms of CPU. Ten such calls arrive per second.

Now the loop needs 600 ms + 800 ms = 1.4 s of CPU per second of wall time. It cannot keep up: queues grow without bound and every endpoint's latency climbs into seconds, even the cheap ones.

Switching to the async `crypto.pbkdf2`, which runs on the thread pool, puts the loop back at 60% busy. The password endpoint still takes 80 ms, but nothing else waits behind it.

</div>

## Modern hybrids

Writing everything as callbacks or async functions is a real cost, and many teams prefer code that looks like plain blocking code. Two popular runtimes give you that while keeping event-loop efficiency underneath:

- **Go's goroutines** start with a stack of about 2 KB that grows as needed, so a server can run hundreds of thousands of them. The Go runtime multiplexes goroutines onto a few OS threads (one per core by default). When a goroutine does a network read that would block, the runtime parks it, registers the socket with epoll, and runs another goroutine. You write `conn.Read(buf)`; the runtime does the event loop.
- **Java's virtual threads** (standard since Java 21) do the same on the JVM: millions of lightweight threads mounted on a small pool of carrier threads, unmounted whenever they block on I/O.

These models also run on all cores at once, so a CPU-heavy task only occupies one OS thread rather than freezing the whole server. You still need to bound concurrency: a million goroutines all querying the same database will overwhelm the database, not the runtime.

~~~dg-layers How a modern runtime serves many connections: cheap user-level tasks on top, a few OS threads in the middle, the kernel's readiness API at the bottom.
Your handlers | goroutines or virtual threads, written as blocking code
*Runtime scheduler | parks tasks that wait, runs ready ones
OS threads | about one per CPU core
Kernel | epoll or kqueue reports ready sockets
~~~

## Choosing a model

- **I/O-heavy work with many connections** (proxies, API gateways, chat, WebSockets): event loops or goroutine-style runtimes.
- **CPU-heavy work** (image processing, ML inference): a pool of threads or processes sized to the core count, with a queue in front.
- **Moderate traffic, simple code wanted**: thread-per-request with a sensibly sized pool is fine. Thousands of requests per second is well within reach.

## Common mistakes

- **Blocking calls on an event loop**, often hidden in a library.
- **Huge thread pools** to "fix" slow dependencies: 2,000 threads waiting on a slow database just means 2,000 queries piling on it.
- **Confusing more threads with more speed.** If the CPU is the bottleneck, threads beyond the core count only add switching.
- **No timeout on downstream calls**, so waiting connections accumulate until memory runs out.

## In an interview

When asked how a server handles load, distinguish waiting from working. Say how many concurrent connections you expect, note that idle connections are cheap with an event loop or lightweight threads, and keep CPU-heavy work off the request path. Little's Law (lesson 01.04) gives the concurrency you need: 5,000 requests per second at 100 ms each means 500 requests in flight, which is fine for an event loop and needs a 500-thread pool in the thread-per-request model.

<div class="take">
<div class="lab">Key takeaway</div>

thread-per-request is simple but costs a full thread for every waiting connection; event loops and lightweight-thread runtimes make idle connections cheap by only spending CPU on sockets that are ready. Match OS threads to cores, never block the loop, and push CPU-heavy work elsewhere.

</div>

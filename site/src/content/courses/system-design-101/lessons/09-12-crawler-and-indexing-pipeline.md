---
id: "09.12"
course: "system-design-101"
module: "09"
slug: "crawler-and-indexing-pipeline"
title: "Crawler and indexing pipeline"
keyIdeas: ["A crawler starts from seed URLs, fetches pages, extracts links and adds new URLs to a frontier queue.", "**Politeness**: respect robots.txt, limit requests per host (one connection per host, delays), and keep per-host queues so you don't hammer one site.", "**Prioritization**: crawl important and frequently changing pages more often.", "**Dedup**: normalize URLs, keep a seen-set (Bloom filter at scale), and detect duplicate content with fingerprints (SimHash).", "Handle traps (infinite calendars), timeouts and huge pages. Fetched pages go to storage, then to the indexing pipeline."]
question: "How does a crawler avoid hammering a single site?"
answer: "It never lets two fetches to the same host happen at once and spaces requests out per host. In the Mercator frontier, each back queue holds URLs for one host only (and each host has only one back queue), and a min-heap records the earliest time each host may be contacted again; a thread takes one URL from the queue whose time has come, fetches it, and the host must then wait a gap of roughly ten times that fetch's duration, so slow or struggling servers are automatically hit less often. Crawl-delay, 429 and 503 responses with Retry-After lengthen the wait, robots.txt is cached per host and checked just before fetching, and the crawl is partitioned by host so one machine enforces each host's limits, with about three times as many back queues as threads so many hosts are in rotation."
sources: [{"title": "IR book ch.20: Web crawling", "url": "https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf", "note": "URL frontier, politeness."}, {"title": "ByteByteGo: Design a Web Crawler", "url": "https://bytebytego.com/courses/system-design-interview/design-a-web-crawler", "note": "Interview version."}]
---

## What a crawler does

A **web crawler** (or spider) is the acquisition stage of web search (lesson 09.01). It starts from a set of **seed URLs**, downloads each page, extracts its links, and adds new links to a queue, repeating until it has covered as much of the web as it should. Then it keeps going, revisiting pages to catch changes. The loop is simple to describe and very hard to run at scale, because the web is enormous, constantly changing, full of traps, and made of servers that belong to other people.

The requirements split into must and should:

- **Must be robust.** Survive **spider traps**: sites that generate infinite URLs (a calendar with a "next month" link forever, session IDs in every URL, endlessly nested paths), malformed HTML, huge files, servers that never finish responding.
- **Must be polite.** Respect each site's rules (robots.txt) and never overload a server. A crawler that sends hundreds of requests a second to a small site is indistinguishable from a denial-of-service attack, and will be blocked.
- **Should be distributed and scalable**, so adding machines adds throughput.
- **Should be efficient**, using bandwidth and CPU well.
- **Should prioritise quality**, fetching important pages first.
- **Should keep pages fresh**, revisiting pages in proportion to how often they change and how much they matter.
- **Should be extensible** to new formats and protocols.

### The scale

To see why each module exists, size it. Suppose the goal is 1 billion pages a week, average 100 KB each.

- 1,000,000,000 / 604,800 seconds ≈ **1,650 pages a second**.
- Bandwidth: 1,650 × 100 KB ≈ 165 MB/s, about **1.3 Gbit/s**.
- Storage: 100 TB of raw pages a week, before compression (HTML compresses about 5 to 10 times).
- If each fetch takes about 1 second end to end (DNS, connect, download), you need about **1,650 fetches in flight** at once, which means thousands of concurrent connections, so asynchronous I/O across several machines.
- And politeness: if you can hit each host at most once every few seconds, you need **tens of thousands of distinct hosts** in rotation at any moment.

## The modules

A standard architecture, based on the Mercator crawler design, has these parts.

~~~dg-flow The crawl loop: take a URL from the frontier, fetch, parse, drop duplicates, and feed new links back.
[Fetch] *URL frontier -(next URL)-> DNS resolve -> Fetch page -> Parse
[Store] Parse -(content seen?)-> Store for indexing
[Links] Parse -(links)-> Filter and robots.txt -> Normalise -(seen URL?)-> URL frontier
~~~

1. **URL frontier**: the queue of URLs to fetch next. It decides both what is important and when each host may be contacted (detailed below).
2. **DNS resolution**: turning hostnames into IP addresses. DNS is a classic bottleneck: lookups can take hundreds of milliseconds or seconds, and standard library resolvers block the thread. Crawlers run their own asynchronous resolvers and cache results aggressively (respecting TTLs), since many URLs share a host.
3. **Fetch**: download the page over HTTP, with timeouts, a maximum size, and conditional requests (`If-Modified-Since`, `ETag`) on revisits so unchanged pages cost almost nothing.
4. **Parse**: extract text, metadata and links from HTML (and PDFs, other formats). Some pages need JavaScript rendering, which is far more expensive, so it's reserved for important sites.
5. **Content seen?**: skip pages whose content was already seen at another URL. Mirrors, copies and parameter variations make a large share of the web duplicate. Exact duplicates are found with a **fingerprint** (a 64-bit hash of the content). **Near-duplicates** (same article with different ads or dates) are found with **shingles**: the set of overlapping k-word sequences in the page, compared by Jaccard similarity, made affordable with MinHash or SimHash sketches.
6. **URL filter**: drop URLs outside the crawl's scope (by domain, file type, or a blocklist of known traps).
7. **robots.txt**: a site's rules on which paths crawlers may fetch, and sometimes a `Crawl-delay`. Fetch it once per host, **cache** it (typically for a day), and **check it just before fetching** each URL, because rules change and a URL may wait in the frontier for days.
8. **Normalisation**: turn relative links into absolute URLs and **canonicalise** them: lower-case the host, remove default ports and fragments (`#section`), sort or strip tracking parameters, resolve `../`. Respect `rel="canonical"` hints. Without this, the same page enters the frontier under dozens of URLs.
9. **Duplicate URL elimination**: don't enqueue a URL already in the frontier or already crawled. With billions of URLs, the "seen" set is too big for memory: store fingerprints of URLs on disk, sorted, keep a cache of recent ones, and batch lookups. A Bloom filter in memory can cheaply say "definitely not seen" for most new URLs.
10. **Checkpointing**: periodically save the frontier and state so a crash doesn't restart the crawl from the seeds.

~~~dg-state A URL's life in the crawler. Most new links are dropped as already seen or out of scope; fetched pages are scheduled for revisits.
[*] -> Discovered
Discovered -> Dropped : filtered or already seen
Discovered -> Queued : enters frontier
Queued -> Fetching : host's turn arrives
Fetching -> Fetched : 200 OK, new content
Fetching -> Queued : timeout, retry later
Fetching -> Dropped : robots.txt disallows
Fetched -> *Queued : revisit scheduled
~~~

## The URL frontier: priority and politeness

The question that matters most for the web's goodwill is how a crawler avoids hammering a single site. A naive frontier is one FIFO queue. But links are highly local: a page on example.com links mostly to other pages on example.com. A FIFO queue fills with runs of URLs from the same host, and many fetcher threads pull them at once, sending a burst of parallel requests to one server.

The Mercator frontier solves this with **two sets of queues** that separate two concerns: **what is important** and **when a host may be contacted**.

~~~dg-flow The Mercator frontier: front queues decide what is important; back queues, one host each, decide when a host may be contacted again.
[Priority] New URL -> Prioritiser -(by importance)-> Front queues 1..F
[Politeness] Front queues 1..F -(biased pick)-> Back queue router -(one host per queue)-> *Back queues 1..B
[Fetch] Back queues 1..B -(heap of next allowed times)-> Fetcher thread
~~~

### Front queues: priority

There are F front queues, numbered by priority. A **prioritiser** gives each URL a priority, based on page quality (link-based importance such as PageRank, the site's reputation) and how often it changes (news home pages change every few minutes; an archived paper never changes). It places the URL in the matching front queue. When the back side needs more URLs, it picks a front queue with a **bias towards higher priorities** (high queues are chosen more often, but low ones aren't starved) and takes from it.

### Back queues: politeness

There are B back queues, with two invariants:

- Each back queue holds URLs from **exactly one host**.
- Each host maps to **only one** back queue at a time (a table records which).

So two threads can never fetch from the same host at once, because a host's URLs are all in one queue and only one thread takes from a queue at a time.

### The heap of next-allowed times

A **min-heap** holds one entry per back queue: the earliest time that queue's host may be contacted again. A fetcher thread:

1. Pops the top of the heap: the back queue whose host is allowed soonest.
2. Waits until that time if it hasn't arrived yet.
3. Takes **one** URL from that queue and fetches it.
4. Computes the host's next allowed time and pushes the queue back into the heap.
5. If the queue is now empty, it refills it from the front queues: it takes URLs until it finds one for a host that has no back queue, and assigns that host to this queue (URLs for hosts that already have a queue are routed there).

### The gap

How long should a host wait between requests? A fixed delay (1 second for everyone) treats a tiny shared server the same as a huge CDN. The Mercator rule sets the gap **relative to how long the last fetch took**: about **10 times** the last fetch duration. A fast server that responded in 0.1 s can be hit every second; a slow server that took 3 s (perhaps because it's struggling) waits 30 s. The crawler automatically backs off from servers under stress. A `Crawl-delay` in robots.txt, or HTTP 429 and 503 responses with `Retry-After`, override it with a longer wait.

The Mercator recommendation is about **three times as many back queues as fetcher threads**, so that whenever a thread is free, some host is allowed to be contacted and the thread doesn't sit idle.

~~~dg-timeline Politeness in action for two hosts: the fast host is fetched every few seconds, the slow one waits ten times its last fetch time.
[Host A] 0 s | fetch, took 0.2 s; next allowed at 2.2 s
[Host B] 0 s | fetch, took 3 s; next allowed at 33 s
[Host A] 2.2 s | fetch, took 0.2 s; next at 4.4 s
[Host A] 4.4 s | fetch, took 0.3 s; next at 7.7 s
[Host B] 33 s | *fetch again, took 1 s; next at 44 s
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two hosts, one slow**

Host A's last page took 0.2 s, so its next allowed time is now + 2 s. Host B's took 3 s, so B waits 30 s. A fetcher pops the heap, gets A's queue (earliest time), fetches one page, and reinserts A at now + 2 s. B's 50 queued URLs sit untouched until its 30 seconds are up. Meanwhile, other threads are busy with thousands of other hosts. A struggling site is never hit by two threads in parallel and sees at most one request every 30 seconds while it's slow; when it speeds up to 1 s responses, its gap shrinks to 10 s.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**how many back queues**

A crawler runs 600 fetcher threads per machine. With B = 3 × 600 = 1,800 back queues, there are 1,800 distinct hosts in rotation per machine. If the average gap is 5 seconds, each host allows 0.2 fetches a second, so 1,800 hosts allow 360 fetches a second, enough to keep a machine at its target of about 300 pages a second. With only 600 back queues, the same machine could do at most 120 a second politely, and most threads would wait on the heap.

</div>

## Freshness: when to revisit

After the first crawl, most of the work is **recrawling**. Pages change at very different rates. Revisit too rarely and the index is stale; revisit too often and you waste bandwidth and the site's resources. Crawlers estimate each page's change rate from history (how often content changed at past visits) and schedule revisits accordingly, weighted by importance. Conditional requests (304 Not Modified) make unchanged revisits cheap. Sitemaps with `lastmod` dates and push protocols where sites notify search engines of changes help too.

## Scaling out

Distribute the crawler by **host**: each crawler node owns a set of hosts, for example by hashing the hostname (consistent hashing so nodes can be added). Politeness for a host is then enforced entirely on one node, with no coordination. When a node extracts a link to a host it doesn't own, it sends the URL to the owning node, batched to reduce messages. Content fingerprints and seen-URL sets are partitioned the same way, and seen-URL checks are batched because the set is too large for memory. Hosting providers with many sites on one IP address need care: politeness may need to apply per IP as well as per host name.

## Common mistakes

- A single FIFO frontier, which bursts requests at one host.
- Fixed per-host delays that ignore server speed and `Retry-After`.
- Checking robots.txt only when a URL is discovered rather than just before fetching.
- No URL canonicalisation or trap detection, so the frontier fills with infinite calendar pages.
- Synchronous DNS lookups that stall every thread.

## In an interview

Draw the loop (frontier, DNS, fetch, parse, content dedupe, filter, robots, normalise, URL dedupe, back to frontier), then focus on the frontier: front queues for priority, back queues with one host each for politeness, a heap of next-allowed times, a gap of about ten times the last fetch, and partition by host across machines. Do the throughput arithmetic to show why you need thousands of hosts in rotation.

<div class="take">
<div class="lab">Key takeaway</div>

a crawler loops frontier, DNS, fetch, parse, dedupe, filter and normalise, then back to the frontier. Politeness comes from the Mercator frontier: back queues hold one host each, a heap enforces each host's earliest next contact time, and the gap scales with the host's response time (about ten times the last fetch), with robots.txt checked just before fetching.

</div>

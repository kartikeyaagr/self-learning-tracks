---
id: "01.06"
course: "system-design-101"
module: "01"
slug: "back-of-the-envelope-capacity-planning"
title: "Back-of-the-envelope capacity planning"
keyIdeas: ["Estimate to decide, not to be precise. You want the order of magnitude: is it 10 QPS or 100k QPS, 1 GB or 1 PB?", "Useful constants: 1 day ≈ 86,400 s ≈ 10^5 s. 1 million requests/day ≈ 12 QPS. Peak is often 2-10x average.", "Latency ladder: memory ~100 ns, SSD read ~100 µs, same-datacenter round trip ~0.5 ms, cross-continent ~100 ms.", "Storage = items per day × size × retention × replication factor.", "Worked example: 100M users × 10 writes = 1B writes/day ≈ 11.6k QPS average. 1B × 1 KB = 1 TB/day ≈ 365 TB/year before replication.", "Then ask what the numbers imply: one database can take a few thousand writes/s, so 12k+ QPS with peaks means sharding or batching."]
question: "100M daily users, 10 writes each, 1 KB per write. Estimate write QPS and storage per year."
answer: "Writes per day = 100M users × 10 = 10⁹. Average write QPS = 10⁹ / ~10⁵ s ≈ 10,000 to 12,000 per second, so plan for about 20,000 to 30,000 at a 2 to 3x peak. Storage = 10⁹ × 1 KB = 1 TB per day, about 365 TB per year of raw data, or roughly 1 PB a year with 3 replicas. That rate is beyond one relational primary, so writes need partitioning or a write-optimised store."
sources: [{"title": "ByteByteGo: Back-of-the-envelope Estimation", "url": "https://bytebytego.com/courses/system-design-interview/back-of-the-envelope-estimation", "note": "Powers of two, latency numbers, a worked QPS and storage example."}, {"title": "Latency numbers every programmer should know", "url": "https://gist.github.com/jboner/2841832", "note": "Memorise the orders of magnitude."}, {"title": "sirupsen/napkin-math", "url": "https://github.com/sirupsen/napkin-math", "note": "Current throughput, latency and cost numbers plus technique."}]
---

## Why estimate at all

Before you choose a database, a cache or a sharding scheme, you need to know roughly how big the problem is. **Back-of-the-envelope estimates** are quick thought experiments, using a handful of standard numbers, that tell you which designs are even possible. Does the data fit on one machine? Can one database take the write rate? Is the bandwidth bill \$100 a month or \$100,000?

The goal is to get the **order of magnitude** right, not the exact figure. Whether a number is 10³ or 10⁵ changes the design completely: 1,000 writes per second fits on one database; 100,000 does not. Whether it is 3 × 10⁴ or 4 × 10⁴ changes nothing. So you round hard, keep the arithmetic simple, and spend your effort on the assumptions.

## Numbers to keep in your head

### Powers of two and data sizes

| Power | Approximate value | Name |
| --- | --- | --- |
| 2^10 | a thousand (10³) | 1 KB |
| 2^20 | a million (10⁶) | 1 MB |
| 2^30 | a billion (10⁹) | 1 GB |
| 2^40 | a trillion (10¹²) | 1 TB |
| 2^50 | a quadrillion (10¹⁵) | 1 PB |

Handy object sizes: a character of ASCII text is 1 byte (UTF-8 for other languages is 2 to 4); a 64-bit integer or timestamp is 8 bytes; a UUID is 16 bytes; a short text post with metadata is about 0.5 to 1 KB; a compressed phone photo is 1 to 3 MB; a minute of 1080p video is roughly 50 to 100 MB.

### Time

One day is 86,400 seconds. Call it **10⁵** for estimates (it is off by about 15%, which is fine). Useful consequences:

- 1 million events a day ≈ 12 per second.
- 100 million a day ≈ 1,200 per second.
- 1 billion a day ≈ 12,000 per second.

A month is about 2.6 million seconds (2.5 × 10⁶), and a year about 31.5 million (3 × 10⁷).

### The latency ladder

Rounded times for common operations:

| Operation | Rough time |
| --- | --- |
| L1 cache reference | 1 ns |
| Main memory reference | 100 ns |
| System call | 100 ns to 1 µs |
| Context switch | 1 to 5 µs |
| Compress 1 KB (fast codec) | 2 µs |
| Read 1 MB sequentially from RAM | 10 to 50 µs |
| SSD random read (4 to 8 KB) | 100 µs |
| Same-zone network round trip | 100 to 500 µs |
| Simple key lookup in Redis or indexed query in MySQL | 0.5 to 1 ms |
| Read 1 MB sequentially from SSD | 0.5 ms |
| HDD seek | 10 ms |
| Blob storage (S3) first byte | 20 to 100 ms |
| Cross-Atlantic round trip | 80 to 150 ms |

~~~dg-widget latency The latency ladder on a log scale. Each step down is roughly ten to a thousand times slower.
~~~

The lessons hidden in the ladder: memory is about 1,000 times faster than an SSD random read, and an SSD is about 100 times faster than a spinning disk seek. A network round trip inside a data centre costs as much as thousands of memory reads, so making 50 sequential calls to a cache is slow even though each call is "fast". Crossing an ocean costs about 100 ms per round trip, a limit set by the speed of light in fibre, so no amount of engineering removes it.

### Throughput

Rough rates for one machine or one device:

~~~dg-bars Sequential throughput of common resources on one machine, in MB per second. Random small reads on a spinning disk are a hundred times slower than sequential ones.
Memory (one thread) | 10000 | ~10 GB/s
NVMe SSD sequential | 3000 | ~3 GB/s
10 Gbit network | 1200 | ~1.2 GB/s
HDD sequential | 200 | ~200 MB/s
*HDD random 8 KB | 2 | ~2 MB/s
~~~

The conclusions: keep hot data in memory, avoid random disk seeks, prefer sequential writes (logs, append-only files), and compress before sending data over the network, because a fast compressor runs at hundreds of MB per second while the network moves about 1 GB per second and costs money.

### What one box can do

Some rough capacities to sanity-check designs against:

- One web or API server: 1,000 to 10,000 simple requests per second, depending heavily on the work per request.
- One Postgres or MySQL primary on good hardware: a few thousand to tens of thousands of simple writes per second; more reads, especially with indexes cached in memory.
- One Redis instance: around 100,000 simple operations per second.
- One Kafka broker: hundreds of MB per second of appends.
- One large server: 64 to 192 cores and 0.5 to 2 TB of RAM are ordinary today.

## The method

Estimates go wrong in predictable ways, so follow a few rules:

- **Fermi decomposition**: break the question into quantities you can guess (users, actions per user, bytes per action), then multiply. Named after a physicist famous for estimating things like the number of piano tuners in Chicago from first principles.
- **Keep units** on every number. "150 million users × 2 tweets per user per day = 300 million tweets per day." If the units don't cancel to what you want, something is wrong.
- **Use exponents**: 1.5 × 10⁸ × 2 × 10³ = 3 × 10¹¹ is much easier than multiplying long numbers.
- **Write each assumption down**, and keep to about six. Too many and the errors compound; too few and you've missed something.
- **Round aggressively**: 99,987 / 9.1 becomes 100,000 / 10 = 10,000.
- **Separate average from peak**. Traffic is not flat over a day. A peak of 2 to 3 times the average is a common assumption; event-driven products (ticket sales, sports) can see 10 to 100 times.
- **Remember the multipliers**: replication (usually 3 copies), indexes and metadata (often 20 to 50% on top of raw data), and growth.

~~~dg-tree A Fermi decomposition of storage for a photo app. Each leaf is a number you can guess; multiply up the tree.
Storage per year
  Photos per day
    Daily active users | 50M
    Uploads per user per day | 0.2
  Bytes per photo
    Original | 2 MB
    Thumbnails | 3 sizes, ~100 KB total
  Days kept | 365
  Copies | 3 replicas
~~~

## Worked examples

<div class="ex">
<div class="lab">Worked example</div>

**Example 1: Twitter-scale writes and media**

Assumptions: 300M monthly users, 50% daily active, 2 tweets per active user per day, 10% of tweets carry 1 MB of media, keep data 5 years.

Daily active users = 300M × 50% = 150M.

Tweets per day = 150M × 2 = 3 × 10⁸. Average write QPS = 3 × 10⁸ / 86,400 ≈ 3,500 per second. Peak at 2x ≈ 7,000 per second.

Media per day = 3 × 10⁸ × 10% × 1 MB = 3 × 10⁷ MB = 30 TB per day.

Five years ≈ 30 TB × 365 × 5 ≈ 55 PB, before replication. Tweet text at 1 KB each is only 0.3 TB a day, tiny by comparison. Media dominates storage completely.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**Example 2: the photo app from the tree above**

Photos per day = 50M users × 0.2 = 10⁷.

Bytes per photo ≈ 2 MB + 0.1 MB ≈ 2.1 MB, call it 2 MB.

Per day = 10⁷ × 2 MB = 2 × 10⁷ MB = 20 TB. Per year = 20 TB × 365 ≈ 7.3 PB, and with 3 copies about 22 PB.

Upload bandwidth = 20 TB per day / 10⁵ s ≈ 200 MB/s on average, perhaps 600 MB/s at peak. That is a handful of machines' worth of network, so ingest is not the hard part; storage cost and serving reads are.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**Example 3: how many servers for the API?**

The app has 50M daily users who each load the feed 20 times a day: 10⁹ feed requests per day, about 12,000 per second on average and say 36,000 at a 3x peak.

If one API server handles 2,000 feed requests per second at acceptable latency, you need 36,000 / 2,000 = 18 servers at peak. Add headroom (run at about 60% so a zone can fail) and you plan for about 30 servers.

That is a modest fleet. The interesting question is the data behind it: 36,000 feed reads per second is far beyond one database, so this number argues for caching or precomputed feeds.

</div>

~~~dg-flow The estimation chain for traffic: users to requests per day, divide by seconds in a day, then apply the peak factor.
Daily active users -(x actions per day)-> Requests per day -(/ 100,000 s)-> *Average QPS -(x 2 to 3)-> Peak QPS -(/ per-server capacity)-> Servers
~~~

## Turn numbers into decisions

The estimate is only useful if you say what it implies. Always finish with "so this means we need...":

- **7,000 writes per second** is too many for one relational primary to take comfortably with headroom, so you need batching, sharding, or a write-optimised store.
- **55 PB of media** means object storage, not a database. Databases store pointers to media, never the bytes themselves.
- **A 20 GB working set** fits in memory on one machine. You might not need a cache cluster at all; one Redis node, or the database's own buffer cache, will do.
- **36,000 reads per second for feeds** suggests precomputation or caching, because assembling a feed from scratch for every request would mean hundreds of thousands of database queries per second.
- **Bandwidth**: 1 GB per second of egress is about 2.6 PB a month. At roughly \$0.05 to \$0.10 per GB to the internet, that is \$130,000 to \$260,000 a month, which is why media-heavy products use CDNs and negotiate prices.

Cost is part of napkin maths too. Rough cloud prices to keep in mind: about \$25 per vCPU per month, \$2.50 per GB of RAM per month, \$0.02 per GB per month of object storage, \$0.10 per GB of SSD block storage per month, and \$0.05 to \$0.10 per GB of internet egress. A cloud object store includes its own replication in the price, so for Example 2 you pay for the 7.3 PB a year of photos: about 7.3 × 10⁶ GB × \$0.02 ≈ \$146,000 a month after the first year, growing by that much again every year. That is why such companies move older photos to cheaper tiers (lesson 01.12).

## Common mistakes

- **Forgetting the peak.** Designing for the average means falling over every evening.
- **Forgetting replication and overhead.** Raw data × 3 copies × 1.3 for indexes is four times the raw number.
- **Mixing bits and bytes.** Network speeds are quoted in bits: 10 Gbit/s is about 1.25 GB/s.
- **False precision.** Writing "3,472 QPS" suggests a confidence you don't have. Say "about 3,500, call it 4k".
- **Stopping at the number.** An estimate with no conclusion is wasted time.
- **Spending ten minutes on it.** In an interview, estimates should take two or three minutes, and only for numbers that change the design.

## In an interview

Ask the interviewer for the numbers they care about (daily users, actions per user, data retention) or propose them and ask if they are reasonable. Write assumptions on the board in a column, do the arithmetic visibly with rounded exponents, and box the conclusion. Many interviewers care less about the precise figures than about seeing you use them: "Peak writes are about 25,000 per second, more than one primary can take, so we'll partition writes by user ID."

<div class="take">
<div class="lab">Key takeaway</div>

aim for the right power of ten, write every assumption down with its units, separate average from peak, add the multipliers for replication and overhead, and always finish with "so this means we need...".

</div>

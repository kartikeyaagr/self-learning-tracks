---
id: "10.13"
course: "system-design-101"
module: "10"
slug: "live-reactions-at-high-throughput"
title: "Live reactions at high throughput"
keyIdeas: ["Live reactions (hearts, emoji on a live video) can hit tens of thousands per second for one stream with a million viewers.", "Don't broadcast each reaction to every viewer: that's 10k × 1M messages per second.", "Instead aggregate per short interval (e.g. 1-2 s) per stream: count reactions by type, then broadcast one summary message to all viewers, who animate it locally.", "Shard by stream ID; for huge streams, aggregate in several workers and merge.", "Use a pub/sub layer with connection servers that each hold many viewers' connections. Dropping some reactions is acceptable."]
question: "1M viewers, 10k reactions/s. How do you aggregate and push without sending every reaction to everyone?"
answer: "Treat reactions as counts, not messages, because a heart from one user looks the same as a heart from any other. Clients POST reactions to stateless ingest servers that count per video and reaction type in memory and emit partial counts every 200 ms; an aggregator per video (Kafka partitioned by video, sharded for the hottest streams) sums the partials and publishes one update per tick, with the tick's counts and the running total, to that video's pub/sub channel. Realtime servers, which hold the video's viewers grouped by consistent hashing at the load balancer, receive the update once each and push it over SSE. Each viewer then gets about 5 small updates per second (about 5 million messages per second in total instead of 10 billion), whether there are 10,000 or a million reactions per second, and the client animates the counts and shows its own taps instantly."
sources: [{"title": "Hello Interview: FB Live Comments", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-live-comments", "note": "Realtime high-fanout events."}]
tags: ["Design"]
---

## The problem

A live stream with 1 million viewers receives 10,000 reactions (hearts, likes, laughing faces) per second. Each viewer should see the reactions float up the screen more or less as they happen, and see a running total.

The naive design treats a reaction like a chat message: when someone taps a heart, send it to every viewer. That's 10,000 × 1,000,000 = **10 billion messages per second**. At even 50 bytes per message, that's 500 GB per second of outbound traffic, 4 terabits per second, for one stream. No system should do that, and it would be pointless anyway: no viewer can perceive 10,000 individual hearts per second. A phone screen shows maybe a few dozen floating hearts at a time.

This lesson pulls together the module's counting ideas (batching in lesson 10.01, time buckets in 10.06, merging partial counts in 10.05 and 10.09) and the delivery ideas of Module 11 into one design.

~~~dg-bars Messages per second delivered to viewers for one stream with a million viewers and 10,000 reactions per second.
Forward every reaction | 10000000000 | 10 billion/s
Aggregate every 100 ms | 10000000 | 10 million/s
*Aggregate every 200 ms | 5000000 | 5 million/s
Aggregate every 1 s | 1000000 | 1 million/s
~~~

## Delivery: what live comments teach

Live comments on a video are the closer cousin of this problem and its design is the base we build on. Requirements: millions of concurrent live videos, thousands of comments per second per video, eventual consistency, and under **200 ms** end-to-end delivery. The key decisions:

- **Polling fails.** To get under 200 ms, each viewer would have to poll every 100 ms or so. A million viewers polling 10 times a second is 10 million requests per second, almost all returning "nothing new", crushing the database.
- **Server-Sent Events over WebSockets.** Viewers mostly read. SSE is a one-way server push over plain HTTP (lesson 11.02); writes (posting a comment or a reaction) go through ordinary HTTP POSTs. A two-way WebSocket channel adds connection complexity that this read-heavy pattern doesn't need.
- **Pub/sub between servers.** Viewers are spread over many realtime servers, each holding tens of thousands of open SSE connections. When a comment is created, it is published to a channel; servers subscribed to that channel forward it to their viewers. Channels are chosen by `hash(liveVideoId) % N` so a server only receives traffic for videos its viewers are watching. Redis pub/sub is a common choice because it is low latency, and its fire-and-forget delivery is acceptable since comments are also stored.
- **Co-locate viewers.** An L7 load balancer uses consistent hashing on the video ID so viewers of the same video tend to land on the same realtime servers, minimising how many servers must subscribe to each video's channel.
- **Mega-streams.** For hundreds of millions of viewers, comments are **sampled** (from 50% down to 1 to 2% as velocity rises) or delivered as periodic snapshots through a CDN (lesson 11.10).

## Aggregate reactions instead of forwarding them

Reactions differ from comments in one crucial way: they are **countable and interchangeable**. A heart from user A and a heart from user B look identical to other viewers. A comment's content matters; a heart's identity doesn't. So the system should forward **totals per time slice**, not events. "312 hearts and 40 wows in the last 200 ms" carries everything a viewer's screen needs.

~~~dg-compare Forwarding every reaction versus forwarding a count per tick. The aggregate design's cost depends on viewers and tick rate, not on how many people tap.
[Forward each reaction]
Tap -(publish)-> Every server -> Every viewer
- 10 billion messages per second
- Cost grows with reactions times viewers
- Phones drown in events they can't draw
[*Count per 200 ms tick]
Taps -(sum per tick)-> One update -> Every viewer
+ 5 updates per second per viewer
+ Cost independent of reaction rate
+ Same animation on screen
- Up to one tick of delay
~~~

~~~dg-flow Reactions aggregated twice: ingest servers count per 200 ms, a per-video aggregator merges the partials, and each viewer gets one small update per tick.
Viewers send reactions, 10k per s -(POST)-> Ingest servers: count per type per 200 ms -(partial counts)-> *Aggregator for video -(5 updates per s)-> Pub/sub channel -> Realtime servers -(SSE)-> 1M viewers
~~~

1. **Ingest.** Reactions arrive as small HTTP POSTs (or as messages on an existing connection) to stateless ingest servers behind a load balancer. Each server keeps in-memory counters per (video, reaction type), and every 200 ms emits a partial count such as `{video: v1, heart: 312, wow: 40}` instead of the 352 individual events.
2. **Aggregate.** Partial counts for the video go to one aggregator, for example through Kafka partitioned by video ID so all partials for v1 reach the same consumer. It sums the partials into a per-tick total and a running total (lessons 10.01 and 10.06). Output: one message per tick, regardless of reaction volume.
3. **Fan out.** The aggregator publishes one update per tick to the video's pub/sub channel. Each realtime server holding viewers of that video receives it once and pushes it to its local viewers over SSE.
4. **Render.** The client animates "312 hearts in the last 200 ms" as a burst of floating hearts, perhaps capped at 20 drawn hearts with the rest implied by the counter. The visual effect is the same as individual events.

### Two levels of aggregation

Why aggregate at both the ingest servers and the aggregator? Because each level shrinks traffic by a large factor. 10,000 reactions per second from 20 ingest servers become 20 × 5 = 100 partial-count messages per second to the aggregator, instead of 10,000. The aggregator then turns 100 messages per second into 5. This is the same tree-shaped combining you saw for top-K merging in lesson 10.09, and it only works because counts are additive.

~~~dg-tree The fan-out tree for one tick: one aggregate message becomes 500 server deliveries and then a million SSE frames, with no node sending more than a few thousand.
Aggregator for v1 | 1 message per tick
  Pub/sub channel v1 | delivers to subscribers
    Realtime server 1 | 2,000 viewers
      Viewer SSE frames | 2,000 per tick
    Realtime server 2 | 2,000 viewers
    *Realtime server 500 | 2,000 viewers
~~~

## The numbers

| Design | Messages per second to viewers |
| --- | --- |
| Forward every reaction | 10,000 × 1,000,000 = 10 billion |
| One aggregate update every 200 ms | 5 × 1,000,000 = 5 million |

The aggregate approach is independent of reaction volume: 10,000 or 1 million reactions per second still produce 5 updates per second per viewer. Between servers, it's 5 messages per second per video per realtime server.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the fan-out**

1 million viewers on 500 realtime servers, 2,000 viewers each (servers can hold far more SSE connections, but the video's viewers are spread out for resilience).

Pub/sub: the aggregator publishes 5 messages per second; the channel delivers 5 × 500 = 2,500 messages per second in total. Trivial for Redis.

Each realtime server receives 5 messages per second and writes 5 × 2,000 = 10,000 small SSE frames per second, which is ordinary.

Bandwidth per viewer: an update like `{"t":1731440400200,"heart":312,"wow":40,"total":4812330}` is about 60 bytes plus SSE framing, say 80 bytes. Five per second is 400 bytes per second, about 3 kilobits per second. Negligible next to the video itself, which is several megabits per second.

</div>

## Details that matter

- **Write path protection.** 10,000 POSTs per second is easy to spread over stateless servers. Rate-limit per user (nobody needs to send 100 hearts a second; cap at, say, 10 per second and count the rest as one), and **drop rather than queue** under overload. A lost heart doesn't matter; a backlog of hearts arriving 30 seconds late looks broken.
- **Exact totals?** If the final count matters (it's shown afterwards or stored), persist the aggregator's per-tick totals to a counter store with batching (lesson 10.01). If not, the in-memory aggregate is enough. Duplicates from retries barely matter for hearts, unlike ad clicks (lesson 10.03).
- **Personal feedback.** The user who tapped sees their own heart instantly, rendered locally, without waiting for the round trip. Otherwise the button feels laggy, and the 200 ms tick plus network makes it worse.
- **Hot videos.** One aggregator per video can become a hotspot for the biggest streams. Shard the ingest stream with random suffixes (`v1:0` to `v1:15`) so 16 aggregators each sum a share, and let a final step sum the 16 totals each tick, as in lessons 10.03 and 10.09.
- **Clock and ordering.** Ticks are numbered by the aggregator. A client that receives tick 1,042 after tick 1,043 (rare, but possible during a reconnect) can simply drop it: for an animation, old data is useless.
- **Late joiners and reconnects.** A client that connects or reconnects needs the running total, not the history. The first SSE message sends the current total, then deltas follow.
- **Mega-streams.** Lower the tick rate (1 per second instead of 5), or move to CDN-delivered snapshots that clients fetch every second from an edge cache.

## Walking one tick through the system

~~~dg-seq One 200 ms tick: ingest servers send partial counts, the aggregator sums them and publishes once, and each realtime server pushes the update to its viewers.
Viewers > Ingest servers : 2,200 reaction POSTs during the tick
Ingest servers > Ingest servers : count per video and type
Ingest servers > Aggregator : 20 partials, e.g. heart 103, wow 7
Aggregator > Aggregator : sum: heart 2,070, wow 130
Aggregator > Pub/sub : publish tick 1,042 to channel v1
Pub/sub -> Realtime servers : 1 message to each of 500
Realtime servers -> Viewers : SSE frame to each of 2,000 viewers
Viewers > Viewers : animate a burst sized to 2,070
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one tick**

Between 20:00:00.000 and 20:00:00.200, 2,070 hearts and 130 "wow" reactions arrive across 20 ingest servers, about 110 per server. At 20:00:00.200, each server sends its partial counts to the video's aggregator, which sums to `{tick: 1042, heart: 2070, wow: 130, totalHearts: 4812330}` and publishes it once.

500 realtime servers each receive that one message and push it to their 2,000 viewers. Each phone draws a burst of hearts sized to 2,070 (capped at, say, 20 visible hearts, with intensity scaled), and updates the total.

A viewer on a train loses the connection for 8 seconds. When SSE reconnects, the server sends the latest total and the client skips the 40 ticks it missed. Nobody notices; there is nothing to catch up on.

</div>

## Common mistakes

- **Treating reactions as messages.** It works in a demo with 50 viewers and fails catastrophically at scale.
- **One message per reaction to the aggregator.** Without ingest-side batching, the aggregator becomes the hot spot at 10,000 messages per second per video.
- **Queueing under overload.** Late reactions are worse than dropped ones.
- **Waiting for the server to show my own heart.** Render locally first.
- **Storing every reaction event to show a total.** Store per-tick totals or a counter; keep raw events only if analytics needs them (lesson 10.07).

## In an interview

Lead with the arithmetic: "forwarding every reaction is 10 billion messages per second; the client can't use them anyway". Then: "reactions are interchangeable, so I'll count them in 200 ms ticks on ingest servers, merge per video, and push one small update per tick through pub/sub to realtime servers that hold the video's viewers over SSE". Close with the numbers: 5 updates per second per viewer, independent of reaction volume, and the knobs for bigger streams (tick rate, sharded aggregators, CDN snapshots).

<div class="take">
<div class="lab">Key takeaway</div>

don't fan out individual reactions. Count them on ingest servers in short time slices, merge per video, and push one small aggregate update per tick through pub/sub to the realtime servers holding the video's viewers over SSE. Fan-out becomes proportional to viewers × tick rate, not viewers × reactions, and each viewer receives a few hundred bytes per second whatever the reaction rate.

</div>

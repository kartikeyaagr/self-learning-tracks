---
id: "11.10"
course: "system-design-101"
module: "11"
slug: "live-commentary"
title: "Live commentary"
keyIdeas: ["Live commentary (sports updates, live blogs, live comments) is a small number of writers broadcasting to a huge number of readers for each event.", "Writers post to a service that stores the update and publishes it to the event's channel.", "Viewers connect via WebSocket or SSE to connection servers subscribed to that event's channel.", "For huge events, fan out in tiers: one publisher → many regional relay servers → many connection servers → viewers. Shard channels by event ID.", "New viewers load recent history from a cache, then receive live updates. Live comments from viewers may be sampled when volume is too high to show."]
question: "How do you shard live commentary pub/sub by event to handle a World Cup final?"
answer: "Partition pub/sub by event: each event's comments go to a channel chosen by hash(eventId) mod N, and an L7 load balancer uses consistent hashing on the event ID so its viewers sit on the same few realtime servers, which subscribe only to that channel, so each comment crosses the server network as few times as possible. The World Cup final is a single hot key, so it is marked hot and its viewers are spread over a group of thousands of servers (hashing viewer within the event), all subscribed to the event's channel. Each comment is persisted once, filtered, sampled down to a readable rate that adapts to velocity (1 to 2% or less at a goal), batched every few hundred milliseconds and fanned out through a relay tier so no node sends to more than a bounded number of children; recent-comment history and, for the very largest audiences, snapshots of the stream are served from a cache or CDN, and the fleet is pre-scaled before kick-off."
sources: [{"title": "Hello Interview: FB Live Comments", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/fb-live-comments", "note": "Fan-out to millions of viewers."}]
tags: ["Design"]
---

## The challenge

A World Cup final: hundreds of millions of viewers, and fans posting comments far faster than anyone can read. When a goal is scored, comment volume can jump tenfold within seconds. Everything in lesson 10.13 applies, but comments, unlike reactions, are **distinct messages** that can't simply be summed. You can send "2,070 hearts" as one number; you can't send "2,070 comments" as one number.

This lesson builds the standard live-comment design, shows why sharding pub/sub **by event** is the right default, and then shows why the biggest event of all breaks that default and what to do about it.

## The base design

Requirements: millions of concurrent live videos or events, **thousands of comments per second per video**, availability over consistency (a comment arriving a second late or slightly out of order is fine), and **under 200 ms** delivery for normal streams. The building blocks:

- Comments are posted over normal HTTP and **persisted** to a database, so late joiners can load recent history and nothing depends on the real-time path for durability.
- Viewers receive new comments over **SSE** (lesson 11.02), since reads vastly outnumber writes.
- **Realtime messaging servers** hold viewer connections, tens of thousands to a hundred thousand each.
- **Pub/sub** carries new comments from the comment service to the realtime servers (lesson 11.01).

## Sharding pub/sub by event

A naive pub/sub where every realtime server receives every comment for every video doesn't scale: each server would process the whole world's comments to deliver a tiny fraction of them. With 1,000 realtime servers and 100,000 comments per second worldwide, that's 100 million deliveries per second between servers, nearly all discarded. Shard it:

- Comments for a video are published to channel `hash(liveVideoId) % N`. A realtime server subscribes only to the channels for videos its viewers are watching.
- An **L7 load balancer** (one that understands HTTP and can route on the URL) routes viewers by **consistent hashing on liveVideoId**, so viewers of the same video land on the same few realtime servers. Then each server subscribes to few channels, and each comment is delivered to as few servers as possible.
- Redis pub/sub is used for low latency and dynamic subscriptions; its fire-and-forget delivery is acceptable because comments are persisted and clients re-sync after disconnects.

~~~dg-flow Comments sharded by event: a video's comments go to one channel, and only the servers holding that video's viewers subscribe.
Comment on final -(persist)-> Comments DB -(publish to hash of video id mod N)-> *Channel 17 -(subscribed servers only)-> Realtime servers for the final -(SSE)-> Viewers
~~~

Why co-location matters: without it, a video's 50,000 viewers would be scattered across all 1,000 servers, every server would subscribe to every popular video, and each comment would be delivered 1,000 times between servers. With consistent hashing, those 50,000 viewers sit on one or two servers, and each comment crosses the server network once or twice.

## When one event is the whole load

Sharding by event spreads **different** events across shards. The World Cup final is a **single** event, and on that day it may be most of the platform's traffic. Hash-based placement alone puts the final on one shard, one channel and one or two realtime servers, which can't hold 200 million connections. A single key that is hotter than any machine is the hot-key problem again (lesson 10.01), now for connections and fan-out.

~~~dg-compare Placing one event's viewers: hashing the event to one server works for normal streams and collapses for the final; mapping a hot event to a group of servers spreads it.
[Hash event to one server]
Final -(consistent hash)-> Server 17
+ Fewest subscriptions per comment
+ Perfect for 99% of streams
- One server cannot hold 200M viewers
- One channel becomes a hotspot
[*Hot event to a server group]
Final -(hash viewer within group)-> 2,000 servers
+ Load spread across the fleet
+ Same channel, many subscribers
- Needs a hot-event registry
- Fan-out must be tiered
~~~

What changes for a mega-event:

1. **Spread one event's viewers across many servers.** Keep a small registry of hot events. For a normal video, hash the video ID to one server; for a hot one, hash (video ID, viewer ID) into a **group** of servers sized to the audience: the final gets, say, 2,000 realtime servers, each subscribing to the event's channel. The comment is published once, and the pub/sub layer fans it out to 2,000 subscribers, each pushing to its 100,000 viewers.
2. **Tiered fan-out.** A single Redis channel feeding 2,000 subscribers at thousands of messages per second is itself a hotspot: one Redis node writing to 2,000 sockets for every message. Add a **relay tier**: the publisher sends to, say, 40 relays, each feeding 50 realtime servers. Each hop multiplies; no node fans out to everything.
3. **Sample the comments.** No human can read 10,000 comments per second; at more than about 10 per second the stream is an unreadable blur. For mega-streams, **sample**, lowering the share of comments delivered from 50% down to 1 to 2%, and far lower at peaks, as comment velocity rises. Always include the viewer's own comments (rendered locally) and perhaps comments from friends, verified accounts or highly rated commenters. All comments are still stored.
4. **Batch per tick.** Send comments in small batches (for example every 250 ms) rather than one message each, cutting per-message overhead at every hop.
5. **Snapshots via CDN.** For the largest streams, deliver periodic snapshots through a CDN: viewers fetch "the latest 50 comments" every second or two from an edge cache, which absorbs almost unlimited read load. It is pull instead of push, but at this scale, a one-second-old snapshot from a nearby edge is indistinguishable from live.

~~~dg-tree The fan-out tree for one batch of comments during the final: each node forwards to a bounded number of children, so no machine sends to everyone.
Event publisher | 4 batches per second
  Relay 1 | forwards to 50 servers
    Realtime server 1 | 100,000 viewers
    Realtime server 50 | 100,000 viewers
  Relay 2
    Realtime server 51
  *Relay 40 | 40 relays x 50 = 2,000 servers
~~~

~~~dg-flow Mega-event fan-out: sample and batch first, then fan out through a relay tier to thousands of realtime servers.
Comments, 50k per s -(sample to about 20 per s, batch per 250 ms)-> *Event publisher -(40 relays)-> Relay tier -(50 servers each)-> 2,000 realtime servers -> 200M viewers
~~~

<div class="ex">
<div class="lab">Worked example</div>

**sizing the final**

200 million viewers at 100,000 SSE connections per server = 2,000 realtime servers.

Raw comments: 50,000 per second. Sampled to about 20 per second (0.04%, the per-viewer target used below), batched every 250 ms into 4 messages per second of about 5 comments each, roughly 600 bytes per batch.

The publisher sends 4 messages per second to 40 relays: 160 sends per second. Each relay forwards to 50 servers: 200 sends per second. Each server writes 4 frames per second to each of its 100,000 viewers: 400,000 SSE writes per second and about 250 to 300 MB per second (2 to 2.5 Gbit/s) of outbound data per server, which is heavy but within a modern server with a fast network card. Every node's fan-out is bounded.

Per viewer: 600 bytes × 4 = 2.4 KB per second, about 20 kilobits per second, negligible next to the video stream. Delivering 1% (500 comments per second) instead would be 25 times more: about 60 KB per second per viewer and about 6 GB per second per realtime server, far beyond one network card. That is why the sampling rate has to fall as velocity rises.

</div>

### Adaptive sampling

The sampling rate should follow the comment velocity, not be fixed. A target such as "deliver about 20 comments per second to each viewer" works across the whole match.

~~~dg-timeline Comment velocity during the final and the sampling rate that keeps delivery at about 20 comments per second per viewer.
Kick-off | 8,000 per s, deliver 0.25%
30 min, quiet play | 2,000 per s, deliver 1%
*67 min, goal | 60,000 per s for 30 s, deliver 0.03%
Half-time | 1,000 per s, deliver 2%
*Final whistle | 90,000 per s, deliver about 0.02%
After the match | 500 per s, deliver 4%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**velocity-based sampling at a goal**

Before the goal the stream runs at 2,000 comments per second, delivering 1% (20 per second per viewer). In the five seconds after the goal it jumps to 60,000 per second. The sampler measures velocity over a sliding 2-second window (lesson 10.06) and recomputes the rate: 20 / 60,000 ≈ 0.033%. Without the change, each viewer would receive 600 comments a second, the batches would grow 30 times larger, and every relay and realtime server would see a 30-fold burst at the exact moment everyone is watching.

Which 20 are kept is a product choice: random (fair), weighted towards verified accounts and friends (more interesting), or filtered by a spam and toxicity score first (safer). Many systems do all three: filter, then a weighted sample.

</div>

## Other parts that need sharding

- **Write path.** 50,000 comments per second into one event. Partition the comment store by (event, time bucket) or a random suffix so writes spread across nodes; a single partition per event would be the hot row of lesson 10.01. Moderation and spam filtering run on a stream (Kafka partitioned by a hash of the comment ID, not the event ID, so the filter workers share the load) before publishing.
- **Rate limits.** Per user (for example one comment every few seconds during the final) and per event, enforced at the API, so a bot farm can't flood the stream.
- **History for late joiners.** Serve "last 50 comments" from a cache refreshed every second, not from the database per viewer. Ten million people opening the stream at kick-off would otherwise run ten million identical queries.
- **Reactions.** Aggregate as counts (lesson 10.13), not messages.
- **Pre-scaling.** Mega-events are scheduled. Warm the realtime fleet, the relays and the CDN before kick-off, and register the event as hot in advance rather than waiting for autoscaling to react to a 10-times surge.

## Common mistakes

- **Assuming hashing solves hotspots.** Hashing spreads many keys; it does nothing for one huge key.
- **Delivering every comment.** Beyond a few dozen per second it is unreadable, and the cost is real at every hop.
- **One pub/sub node fanning out to thousands of servers.** Add a relay tier.
- **Querying the database for history** on every join.
- **A fixed sampling rate.** It either wastes capacity in quiet periods or collapses at the goal.

## In an interview

Start with the default: "shard pub/sub by event, channel = hash(event) mod N, and co-locate an event's viewers with consistent hashing at the L7 load balancer so each comment reaches few servers". Then say the important part unprompted: "the final is one key, so it needs special handling: a hot-event registry that spreads its viewers over thousands of servers, a relay tier so fan-out is bounded per node, velocity-based sampling and batching, history and possibly the stream itself from a CDN, and pre-scaling because the date is known".

<div class="take">
<div class="lab">Key takeaway</div>

shard live-comment pub/sub by event (channel = hash(event) mod N) and co-locate an event's viewers on the same servers with consistent hashing at the L7 load balancer. A single huge event outgrows one shard, so spread its viewers over a group of servers, fan out through relay tiers, sample comments by velocity and batch them per tick, serve history and snapshots from a cache or CDN, and pre-scale for the scheduled peak.

</div>

---
id: "11.01"
course: "system-design-101"
module: "11"
slug: "realtime-database-and-websocket-scaling"
title: "Realtime database and WebSocket scaling"
keyIdeas: ["Realtime apps push updates to clients over persistent connections (WebSocket, SSE).", "Connection servers hold millions of open connections; they are stateful, so you need a way to route a message to the server holding the recipient's connection.", "Common design: a pub/sub layer (Redis pub/sub, Kafka, a custom channel service) where connection servers subscribe to channels for their connected users. Slack's channel servers do this, sharded by channel with consistent hashing.", "Handle reconnects: clients resume from the last seen message ID and fetch what they missed.", "Load balancers must support long-lived connections, and deploys must drain connections gradually."]
question: "A user's WebSocket lands on server 7. How does a message from server 2 reach them?"
answer: "Server 7 registers interest when the user connects: it subscribes to a pub/sub topic for that user or for their channels (or, as with Slack's channel servers, each channel is owned by a server found through consistent hashing, and gateway 7 subscribes to the owners of the user's channels). When the message is sent, server 2 first writes it to the database, then publishes it to the recipient's topic or hands it to the channel's owner. The broker or owner forwards it to every server subscribed, including server 7, which pushes it down the open socket. If the user is offline or the push is lost, nothing breaks: on reconnect the client, possibly on a different server, fetches every message newer than its last seen ID from the database."
sources: [{"title": "Hello Interview: Real-time updates", "url": "https://www.hellointerview.com/learn/system-design/patterns/realtime-updates", "note": "Polling to WebSockets to pub/sub."}, {"title": "Slack: Real-time messaging", "url": "https://slack.engineering/real-time-messaging/", "note": "How Slack routes messages to connected clients."}]
---

## Two hops

Chat, live dashboards, notifications, multiplayer cursors and presence dots all share one shape. Something happens on the server side, and a particular user's screen must change within a fraction of a second. Every real-time feature has two separate hops:

1. **Server to client**: how an update gets from a server down to a user's device. Polling, Server-Sent Events and WebSockets are the options (lesson 11.02).
2. **Source to server**: how the update gets from wherever it was created to **the particular server holding that user's connection**.

The first hop gets most of the attention, but the second is the scaling problem. A WebSocket is a long-lived TCP connection between one device and one specific server process. A plain HTTP request can be answered by any server behind the load balancer, because each request is independent. A pushed message can only be delivered by the one server that holds the socket.

With 100 connection servers and millions of connected users, a message created on server 2 (because the sender's HTTP request happened to land there) must find the one server, say server 7, that holds the recipient's socket. Server 2 doesn't know where the recipient is. This lesson is about how it finds out.

### What a connection server costs

It helps to know the numbers first. An idle WebSocket costs the server a TCP socket, a TLS session and some buffers: typically **10 to 50 KB of memory** depending on the language and buffer sizes. A well-tuned server (Go, Rust, Erlang, or Node with care) holds **50,000 to several hundred thousand** mostly idle connections. Ten million concurrently connected users therefore means something like 100 to 200 connection servers, and the routing problem is "which of 200 machines holds user X right now?", a question whose answer changes every time a phone switches networks.

## Option 1: pub/sub through a broker

Every connection server subscribes to topics for the users or channels it currently holds. Senders publish to a topic; the broker delivers to whichever servers subscribed. Neither side needs to know the other.

~~~dg-seq Pub/sub routing: server 7 subscribed on behalf of its connected user, so the broker delivers server 2's publish to it.
Bob > Server 7 : WebSocket connect
Server 7 > Broker : SUBSCRIBE user:bob
Alice > Server 2 : send "hi" to Bob
Server 2 > Database : store message 9001
Server 2 > Broker : PUBLISH user:bob "hi"
Broker -> Server 7 : "hi"
Server 7 -> Bob : push over socket
~~~

The topic can be per user (`user:bob`), per conversation (`dm:alice:bob`) or per channel (`channel:general`). Per-channel topics are much cheaper for group chat: a message to a 500-member channel is one publish, delivered once to each server that has at least one member online, instead of 500 publishes.

Redis pub/sub is a common choice: fast, simple and with dynamic subscriptions. But it is **fire-and-forget**: a message published while a subscriber is disconnected is simply gone; there is no buffer and no replay. So the durable copy of each message lives in a database, and clients re-sync on reconnect. Alternatives with persistence (Kafka, NATS JetStream, Redis Streams) let a subscriber resume from an offset, at the cost of more machinery. Kafka in particular is poor at millions of tiny dynamic topics, so it is usually used for the event backbone, not per-user routing.

## Option 2: consistent hashing of users or channels to servers

Instead of a broker, route by hash. Each user (or channel) belongs to a known server on a **consistent hash ring**, a scheme where keys and servers are hashed onto the same circle and each key belongs to the next server clockwise, so adding or removing a server moves only a small fraction of keys (lesson 04.07). A sender's server computes hash(recipient), looks up the owner, and forwards directly. There's no broker hop, and no subscription state, because placement **is** the routing table.

The catch: connection placement must follow the ring. Either the load balancer routes each user's WebSocket to their owner server (which means the load balancer must understand the hash), or a stateless edge tier accepts the connection and forwards to the owner. When servers join or leave, the affected users' connections must move. Discord, which runs on Erlang/Elixir, uses a similar idea: each guild (server community) is a process placed on a node by consistent hashing, and user sessions send events to the guild's process, which fans out to members' sessions.

~~~dg-compare Two ways for server 2 to reach the server holding the recipient: ask a broker, or compute the owner.
[Broker pub/sub]
Server 2 -(publish)-> Broker -> Server 7
+ Connections can land anywhere
+ Simple: subscribe on connect
- Extra hop and a broker to scale
- Redis pub/sub drops messages on blips
[Consistent hashing]
Server 2 -(hash user)-> Server 7
+ No broker, one direct hop
+ Owner holds state, e.g. presence
- Placement must follow the ring
- Rebalancing moves connections
[*Both, as at Slack]
Hash channels to owners, gateways subscribe
+ Scales to tens of millions online
- Most moving parts
~~~

## How Slack does it

Slack's real-time messaging architecture combines both ideas. Its components:

- **Channel Servers (CS)**: stateful and in-memory, each owning a subset of channels by **consistent hashing** of the channel ID. At peak, about **16 million channels per host**. Every kind of entity (user, team, file, huddle, channel) has a channel ID, so the same mechanism carries every kind of event.
- **Gateway Servers (GS)**: stateful servers holding users' WebSocket connections and their channel subscriptions, deployed in **multiple regions** so clients connect to one nearby.
- **Admin Servers (AS)**: stateless, sitting between the web app backend and the channel servers.
- **Presence Servers**: track who's online (the green dots); users are hashed to them.
- **Envoy**: the edge proxy and load balancer, terminating TLS.

When a client connects, the gateway fetches the user's channel list from the web app and **subscribes to the channel servers** owning those channels. A message then flows: client → web app (stores it) → admin server → (hash) channel server → every gateway subscribed to that channel → every connected client in that channel.

~~~dg-flow A Slack message: the web app stores it and hands it to an admin server, which finds the channel's owner by consistent hashing; that channel server fans out to every subscribed gateway.
Sender -(API)-> Webapp -> Admin server -(hash channel id)-> *Channel server -(to every subscribed gateway, all regions)-> Gateway servers -> Members' sockets
~~~

So the channel server plays the broker's role, but each channel has exactly one owner found by hashing, and gateways (which hold connections anywhere) subscribe to owners. Slack reports **tens of millions of connected clients** and delivery across the world within **500 ms**. Consistent hash ring managers replace an unhealthy channel server in **under 20 seconds**, during which delivery for its channels is slower.

## Things every design needs

- **A connection registry or subscriptions.** Something must know which server holds whom: subscriptions in a broker, ownership on a ring, or an explicit registry (`user:bob → gw-7`, in Redis with a TTL refreshed by heartbeats).
- **Reconnect and catch-up.** Connections drop constantly: mobile networks change, laptops sleep, servers deploy. The client reconnects, probably to a different server, and asks for everything since its last seen message ID. Real-time delivery is an **optimisation on top of a durable store**, never the only copy.
- **Heartbeats.** Ping frames every 20 to 60 seconds detect dead connections (a phone that lost signal doesn't send a TCP close) and stop proxies and NAT gateways from closing idle connections, many of which drop idle TCP flows after 60 to 350 seconds.
- **Load balancing and draining.** WebSockets are long-lived, so connections don't rebalance by themselves after a deploy or scale-out. Servers drain gradually (stop accepting, then close existing connections in batches over minutes), and clients reconnect with exponential backoff and **jitter** (random delay) so they don't all return at once.
- **Ordering and duplicates.** A message may arrive by push and again in the catch-up fetch. Clients deduplicate by message ID and order by server-assigned sequence.

~~~dg-state The client side of a real-time connection: every drop leads back through a catch-up fetch before the client trusts pushes again.
[*] -> Disconnected
Disconnected -> Connecting : start or backoff timer fires
Connecting -> Disconnected : failed, retry later
Connecting -> Catching up : socket open, subscribed
Catching up -> *Live : fetched messages since last ID
Live -> Disconnected : heartbeat missed
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a direct message, online and offline**

Bob is connected to gateway server 7, which subscribed to `user:bob` (or to the DM channel's channel server). Alice sends "hi" through server 2's API. Server 2 writes the message to the database with ID 9001, then publishes it. The broker or channel server pushes it to server 7, which writes it to Bob's socket. Bob's client stores 9001 as its last seen ID. End to end: a database write (5 ms), a publish (1 ms), a broker hop (1 ms), a socket write and network transit (20 to 100 ms).

If Bob was offline, nothing is pushed; the publish reaches no subscriber and disappears, which is fine. When he reconnects (to gateway 12 this time), his client says "give me everything after 8,950", fetches 9001 and the rest from the database, then switches to live pushes. A mobile push notification (APNs or FCM) can be sent in parallel to wake his phone.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**a deploy without a reconnect storm**

A gateway holds 80,000 connections and must restart. If it closes all of them at once, 80,000 clients reconnect within a second, each doing TLS, authentication, subscription and a catch-up query: 80,000 database queries in a burst, enough to hurt.

Instead, the server stops accepting new connections and closes its existing ones in batches of 1,000 every 3 seconds, over 4 minutes. Clients reconnect after a random delay of 0 to 5 seconds. The other gateways absorb about 330 new connections per second, smoothly.

</div>

~~~dg-timeline A gateway server failure: clients notice through heartbeats, reconnect elsewhere with jitter, and fill the gap from the database.
0 s | Gateway 7 crashes, 60,000 sockets die
0 to 30 s | Clients miss heartbeat, mark disconnected
30 to 40 s | *Reconnect to other gateways with random backoff
31 to 45 s | Gateways resubscribe to channels for new users
32 to 50 s | Clients fetch messages since last seen ID
50 s | *All clients live again, no message lost
~~~

## Common mistakes

- **Treating push as delivery.** If the database write happens after the publish, or not at all, a disconnected user loses the message. Store first, then publish.
- **One topic per user for group messages.** Publishing a 10,000-member channel message to 10,000 user topics is 10,000 publishes. Publish once per channel.
- **No jitter on reconnect.** Synchronised retries turn a small failure into a self-inflicted outage.
- **Sticky sessions as the routing plan.** Stickiness only keeps one client on one server; it doesn't tell server 2 where Bob is.
- **Forgetting the idle timeouts** of load balancers and NAT, which silently kill quiet connections.

## In an interview

Draw the two hops explicitly. Then say: "connection servers subscribe, on behalf of their connected users, to channel topics in a pub/sub layer (or channels are owned by servers found by consistent hashing, as at Slack); the sender's server stores the message and publishes; the broker forwards to subscribed servers; on reconnect, clients fetch from the database since their last seen ID". Mention sizing (50,000 to 100,000+ connections per server), heartbeats, and draining with jittered reconnects.

<div class="take">
<div class="lab">Key takeaway</div>

real-time delivery has two hops; the hard one is getting a message from the server where it was created to the server holding the recipient's socket. Use pub/sub (servers subscribe for their connected users or channels) or consistent hashing of channels to owner servers, or both as Slack does, and always back it with a durable store, heartbeats, and catch-up on reconnect.

</div>

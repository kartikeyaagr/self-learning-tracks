---
id: "11.02"
course: "system-design-101"
module: "11"
slug: "websockets-vs-sse-vs-long-polling"
title: "WebSockets vs SSE vs long polling"
keyIdeas: ["**Polling**: client asks every N seconds. Simple, wasteful, delayed.", "**Long polling**: server holds the request until there's data or a timeout. Near-realtime over plain HTTP.", "**Server-Sent Events**: one-way server → client stream over HTTP, with automatic reconnect and event IDs. Great for feeds, scores, notifications.", "**WebSockets**: full-duplex, low-overhead messages both ways. Needed for chat, games, collaborative editing.", "Live sports scores: SSE is enough, since data only flows to the client and it works well with HTTP infrastructure."]
question: "Live sports scores: WebSocket or SSE? Why?"
answer: "SSE. Scores flow one way, from server to fans, and only a few times a minute, so a two-way channel is unnecessary, and anything a fan sends can be an ordinary POST. SSE runs over ordinary HTTP, so it passes through existing load balancers, proxies and CDNs, multiplexes over HTTP/2 (avoiding the 6-connection limit of HTTP/1.1) and uses normal cookie auth; EventSource reconnects automatically after a mobile network drop and sends Last-Event-ID so the server resumes exactly where the fan left off. WebSockets would add a protocol upgrade, stateful two-way connections and hand-written reconnect logic for no benefit; they suit cases where clients also send frequent real-time messages, such as chat or games."
sources: [{"title": "MDN: WebSockets API", "url": "https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API", "note": "Full-duplex connections."}, {"title": "MDN: Server-sent events", "url": "https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events", "note": "One-way streaming over HTTP."}]
---

## The problem: HTTP can't speak first

Plain HTTP is request-response. The client asks, the server answers, and the exchange is over. The server has no way to say "something changed" unless the client happens to be asking at that moment. Every real-time feature (a chat message, a goal in a match, a price tick, a notification badge) needs a way around that.

There are four common techniques. They differ in who opens what, how long it stays open, which direction data flows, and how they behave with the proxies, load balancers and flaky mobile networks in between. Choosing well depends on one main question: does the client need to send frequent real-time messages too, or does it mostly listen?

~~~dg-matrix The four ways to get updates from server to browser, compared on the properties that decide most designs.
Technique | Server push | Client sends on same channel | Latency | Works through any HTTP proxy | Server cost when idle
Short polling | no, client pulls | yes, each poll is a request | up to the poll interval | yes | high, constant requests
Long polling | partial, emulated | partial, separate requests | low, plus reconnect gap | yes | medium, held requests
Server-Sent Events | yes | no, use normal POSTs | low | yes, if not buffered | low, one idle stream
WebSockets | yes | yes, both directions | lowest | partial, needs Upgrade support | low, one idle socket
~~~

## Short polling

The client asks "anything new?" every few seconds: `GET /scores?since=118`. The server answers immediately, usually with nothing.

It is simple and works everywhere, but it's wasteful and slow. Latency averages half the polling interval, and most requests are empty. Every poll carries full HTTP headers (often 500 to 1,000 bytes with cookies), a TLS record, and a trip through the load balancer, auth middleware and application.

<div class="ex">
<div class="lab">Worked example</div>

**what polling costs at scale**

1 million fans follow a match, polling every 5 seconds. That's 200,000 requests per second, all day, whether or not anything happens.

At about 800 bytes of request headers plus a small response, that's roughly 200 MB per second of traffic, and 200,000 authenticated requests per second hitting the application. A football match has perhaps 10 goals and a few hundred minor events in 2 hours, so over 99.9% of those requests return "nothing new".

And a goal still takes on average 2.5 seconds (and up to 5) to show up. Halving the latency doubles the cost.

</div>

## Long polling

The client asks, and the server **holds the request open** until there's data or a timeout (often 20 to 30 seconds), then responds. The client immediately sends a new request. When nothing is happening, each client costs one idle request every 30 seconds instead of six; when something happens, it arrives almost instantly.

~~~dg-seq Long polling: the request waits at the server until there is news, then the client immediately asks again.
Client > Server : GET /updates?since=118
Server > Server : hold up to 30 s
Server -> Client : 200: goal, id 119 (after 12 s)
Client > Server : GET /updates?since=119
Server > Server : hold, nothing happens
Server -> Client : 204 no content (after 30 s)
Client > Server : GET /updates?since=119
~~~

Its weaknesses: one message per response, so bursts cause a flurry of request cycles; a **gap** between the response and the next request in which events must be buffered server-side (the `since` parameter makes that safe); and held requests occupy server resources and can hit proxy timeouts. It was the backbone of early web chat (Facebook chat, Gmail) and remains a robust fallback when nothing else gets through a corporate proxy.

## Server-Sent Events

SSE is a **one-way** channel: one HTTP response that never ends, into which the server writes events whenever it likes. The client cannot send events back over it; it uses ordinary HTTP requests for that. The browser API is `EventSource`, and the server responds with `Content-Type: text/event-stream`. The format is plain UTF-8 text lines, with a blank line ending each event:

- `data:` the payload (several `data:` lines join with newlines)
- `event:` a named event type, so the client can register different handlers
- `id:` sets the last event ID
- `retry:` reconnection delay in milliseconds
- Lines starting with `:` are comments, useful as **keep-alive** pings.

The browser **reconnects automatically** if the connection drops, and sends the last ID it saw in a `Last-Event-ID` header, so the server can resume from where the client left off. That gives you reliable delivery semantics almost for free.

~~~dg-seq An SSE stream that survives a network change: the browser reconnects by itself and the Last-Event-ID header lets the server resume.
Browser > Server : GET /matches/42/stream (Accept: text/event-stream)
Server -> Browser : 200, text/event-stream
Server -> Browser : id: 118, event: goal, data: 2-1
Server -> Browser : ": ping" every 15 s
Browser > Browser : Wi-Fi to 4G, connection drops
Browser > Server : GET /matches/42/stream, Last-Event-ID: 118
Server -> Browser : id: 119, event: card, data: yellow, 71 min
~~~

The main warnings:

- **Connection limits on HTTP/1.1.** Browsers allow only about **6 connections per browser and domain**, across all tabs. Six tabs with an SSE stream each and the seventh tab's ordinary requests hang. Over **HTTP/2** the streams are multiplexed on one connection and the limit is negotiated by the server (commonly 100 or more streams; nginx defaults to 128), so serve SSE over HTTP/2.
- **Buffering proxies.** A proxy that buffers responses holds events until the buffer fills, destroying the real-time behaviour. For nginx, send `X-Accel-Buffering: no` or disable `proxy_buffering`; also disable response compression buffering.
- **Text only, no custom headers.** SSE carries UTF-8 text (send JSON; binary must be encoded). The `EventSource` API can't set an `Authorization` header, so authenticate with cookies or a short-lived token in the URL.

## WebSockets

WebSockets give a **two-way interactive session**: either side can send messages at any time without polling. The connection starts as an HTTP/1.1 request and is **upgraded** through a handshake:

1. The client sends `GET /chat` with `Upgrade: websocket`, `Connection: Upgrade`, `Sec-WebSocket-Version: 13` and a random `Sec-WebSocket-Key`.
2. The server replies `101 Switching Protocols` with `Sec-WebSocket-Accept`, a hash of the key combined with a fixed GUID, proving it understood the WebSocket protocol rather than being a confused cache.
3. From then on, the TCP connection carries WebSocket **frames**, not HTTP. Each frame has a 2 to 14 byte header; frames from the client are masked (XOR with a random key) so that intermediaries can't be tricked into caching attacker-chosen bytes.

Frame overhead is tiny compared with HTTP headers, so for frequent small messages (a typing indicator, a cursor position 30 times a second, a game input) WebSockets are by far the most efficient.

The costs:

- **Stateful connections** on the server, and the routing problem of lesson 11.01.
- **Infrastructure support.** Every proxy and load balancer in the path must support the Upgrade and long idle connections. Most do now, but some corporate proxies still break it, which is why libraries such as Socket.IO fall back to long polling.
- **No built-in reconnection or resume.** You write reconnect, backoff, heartbeats and "resume from message ID" yourself.
- **No backpressure in the browser API.** If messages arrive faster than the page can process them, memory or CPU can be exhausted. The newer `WebSocketStream` and WebTransport APIs address that.
- An open WebSocket can stop a page entering the browser's back/forward cache, so close it when the page is hidden for long.

## Choosing

~~~dg-tree A quick decision: who sends, how often, and how much infrastructure can you change?
Need server push?
  No, data changes rarely | short polling or plain fetch
  Yes
    Client also sends often, low latency? | chat, games, editors
      *WebSockets
    Mostly server to client? | scores, feeds, notifications
      *Server-Sent Events
      Hostile proxies or very old clients | long polling fallback
~~~

For **live sports scores**, choose **SSE**:

- Data flows one way: the server pushes score changes; the fan doesn't send anything in real time. Anything they do send (a prediction, a reaction) is an ordinary POST.
- It is plain HTTP, so it passes through existing load balancers, CDNs and proxies, works with HTTP/2 multiplexing, and uses your existing auth cookies.
- Automatic reconnection with event IDs comes free, which matters on mobile networks that drop every few minutes.
- Updates are infrequent (a few per minute), so long polling would also work; WebSockets' bidirectional channel and custom protocol would be complexity without benefit.

WebSockets win when the client also sends frequently and with low latency: chat with typing indicators, multiplayer games, collaborative editing, trading terminals.

~~~dg-compare Live scores against a collaborative editor: the direction and rate of client messages decides the transport.
[*Live scores: SSE]
Server -(score events)-> Fans
+ One-way, a few events per minute
+ Plain HTTP, auto-reconnect, resume by ID
- Fans' own actions go as POSTs
[Collaborative editor: WebSockets]
Messages both ways, every keystroke
+ Both sides send tens of messages per second
+ Tiny frame overhead
- Must build reconnect and resume yourself
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one SSE stream for a match**

The client runs `new EventSource("/matches/42/stream")`. The server sends `id: 118`, `event: goal`, `data: {"home":2,"away":1,"min":67}`, followed by a blank line. Every 15 seconds it sends a `:` comment line to keep proxies and mobile carriers from timing out the idle connection.

The phone switches from Wi-Fi to 4G and the connection drops. The browser waits the `retry` interval (say 3 seconds), reconnects and sends `Last-Event-ID: 118`, so the server sends anything after 118 from a short per-match event buffer in Redis. The fan sees nothing odd.

Server side, 1 million fans are 1 million idle HTTP/2 streams spread over, say, 20 servers holding 50,000 each. A goal is one publish, fanned out by each server to its 50,000 streams: a few hundred bytes each, done in well under a second.

</div>

## Common mistakes

- **WebSockets by default.** Many features are server-to-client only; SSE is simpler and plays better with HTTP infrastructure.
- **SSE over HTTP/1.1** with several tabs open, hitting the 6-connection limit.
- **Forgetting keep-alives.** Idle connections are silently closed by NATs and load balancers.
- **No resume.** Whatever the transport, a reconnect must fetch what was missed, by event or message ID.
- **Polling at 1 second "to be safe".** At scale, that is a self-inflicted load test.

## In an interview

Name the choice and justify it with direction and frequency: "scores flow one way a few times a minute, so SSE over HTTP/2: plain HTTP, auto-reconnect with Last-Event-ID, cheap idle connections; reactions go as POSTs." Then mention the second hop (lesson 11.01): how the score update reaches the server holding each stream.

<div class="take">
<div class="lab">Key takeaway</div>

short polling is simplest and wasteful; long polling emulates push one message at a time; SSE is a one-way push stream over plain HTTP with built-in reconnection and resume; WebSockets are a full two-way channel with tiny frames but more to build and operate. Use SSE when the server pushes and the client mostly listens, and WebSockets when both sides talk often.

</div>

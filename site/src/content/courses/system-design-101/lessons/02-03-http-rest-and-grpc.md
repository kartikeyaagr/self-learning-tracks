---
id: "02.03"
course: "system-design-101"
module: "02"
slug: "http-rest-and-grpc"
title: "HTTP, REST, and gRPC"
keyIdeas: ["HTTP/1.1 is text-based request/response. HTTP/2 adds multiplexing many requests over one connection and header compression. HTTP/3 runs over QUIC (UDP).", "REST over HTTP + JSON is universal, human-readable and cache-friendly. Good for public APIs and browsers.", "gRPC uses Protobuf schemas and HTTP/2. Smaller payloads, generated typed clients, streaming in both directions, strict contracts. Good for internal service-to-service calls.", "gRPC downsides: not native in browsers (needs gRPC-Web), binary payloads are harder to debug, and some proxies need extra setup.", "GraphQL lets clients ask for exactly the fields they need. Useful for varied front ends, but harder to cache and to protect from expensive queries."]
question: "Why do many companies use REST at the edge and gRPC between internal services?"
answer: "Public clients like browsers, mobile apps and third-party developers need something universal, cacheable by standard HTTP caches and CDNs, and easy to debug with everyday tools, which is REST over HTTP with JSON. Between internal services the company controls both ends, so it can use gRPC's compact protobuf encoding, generated typed clients, HTTP/2 multiplexing, deadlines and streaming to cut latency, CPU and bandwidth at very high call volumes and to catch contract mistakes at compile time."
sources: [{"title": "MDN: An overview of HTTP", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Overview", "note": "Requests, responses, methods, statelessness."}, {"title": "gRPC: Introduction", "url": "https://grpc.io/docs/what-is-grpc/introduction/", "note": "Protobuf contracts and HTTP/2 RPC."}, {"title": "Hello Interview: Networking Essentials", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/networking-essentials", "note": "REST vs gRPC vs GraphQL, L4 vs L7."}]
tools: ["gRPC"]
---

## HTTP in one page

**HTTP** (Hypertext Transfer Protocol) is a client-server protocol for fetching and changing resources. The client (a browser, a mobile app, or another service) always starts the exchange by sending a request; the server answers each request with exactly one response. A web page is assembled from many such exchanges: the HTML, then dozens of scripts, stylesheets, images and API calls.

A request has a **method** (GET, POST...), a **path**, a protocol version, **headers** and an optional **body**. A response has a version, a **status code**, headers and an optional body. Over HTTP/1.1 it is plain text you can read:

```
GET /v1/urls/abc123 HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJhbGciOi...
Accept: application/json

HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: private, max-age=60
ETag: "v7"

{"code":"abc123","long_url":"https://example.com/sale"}
```

Headers are where much of HTTP's power lives. They control **caching** (`Cache-Control`, `ETag`, `If-None-Match`), **authentication** (`Authorization`), **content negotiation** (`Accept`, `Content-Type`, `Content-Encoding: gzip`), and **cross-origin rules** for browsers (CORS headers).

HTTP is **stateless**: each request stands alone, and the server needn't remember anything between them. Sessions are layered on top with cookies or tokens sent on every request. Statelessness is what makes web servers easy to scale horizontally (lesson 01.08): any server can handle any request.

Because the meaning of methods, status codes and headers is standardised, **intermediaries** can work without understanding your application. Caches, CDNs, load balancers, auth gateways and corporate proxies all sit between clients and servers and act on standard headers. A CDN can cache your `GET /logo.png` for a day because of `Cache-Control: max-age=86400`, without knowing anything else about your site.

## How the versions differ

~~~dg-timeline The main HTTP versions. Each one attacked the cost of connections and the blocking between requests.
1996 | HTTP/1.0: new TCP connection per request
1997 | HTTP/1.1: keep-alive connections, pipelining (rarely usable)
2015 | HTTP/2: binary frames, multiplexed streams, header compression
2015 | gRPC released, built on HTTP/2
2022 | *HTTP/3 standardised: runs over QUIC on UDP
~~~

- **HTTP/1.0** opened a new TCP connection for every request. Each TCP handshake costs a round trip, and TLS adds one or two more, so fetching 50 resources from a server 50 ms away wasted seconds on setup alone.
- **HTTP/1.1** kept connections open for reuse (**keep-alive**), which removed most of the setup cost. But each connection still carries one request at a time: a slow response blocks the ones behind it on that connection (**head-of-line blocking** at the HTTP level). It also defined **pipelining** (sending several requests before responses arrive), but responses must come back in order, and buggy proxies made it unusable in practice. Browsers worked around this by opening about six connections per host.
- **HTTP/2** keeps the same methods, status codes and headers, but sends them as **binary frames**. Each request/response pair is a **stream** with an ID, and frames from many streams are interleaved on one connection, so a slow response no longer blocks others at the HTTP level. Headers are compressed with **HPACK**, which matters because the same large headers (cookies, auth tokens, user agents) repeat on every request.
- **HTTP/3** runs over **QUIC**, a transport built on UDP (lesson 02.04). HTTP/2 still suffers head-of-line blocking at the TCP level: one lost packet stalls every stream on the connection until it is retransmitted. QUIC gives each stream independent delivery, builds TLS 1.3 into the handshake, and lets a connection survive a phone switching from Wi-Fi to mobile data.

~~~dg-seq Over HTTP/2, many requests share one connection as interleaved streams, and responses come back in whatever order they finish.
Client > Server : HEADERS stream 1 (GET /feed)
Client > Server : HEADERS stream 3 (GET /avatar.png)
Client > Server : HEADERS stream 5 (GET /ads)
Server -> Client : DATA stream 3 (avatar, fast)
Server -> Client : DATA stream 1 part 1
Server -> Client : DATA stream 5 (ads)
Server -> Client : DATA stream 1 part 2, end
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what connection reuse saves**

A mobile app makes 20 API calls on start-up to a server 100 ms away (one round trip, RTT = 100 ms).

New connection per request, with TCP and TLS 1.2: about 1 RTT for TCP + 2 for TLS + 1 for the request = 4 RTT = 400 ms each. Even with six in parallel, about 4 rounds × 400 ms ≈ 1.6 s.

HTTP/2 with one reused connection: 3 RTT of setup once (300 ms), then all 20 requests go out together and come back in about 1 RTT: roughly 400 ms in total. With TLS 1.3 (1 RTT handshake) it is about 300 ms.

</div>

## REST

**REST** (representational state transfer) is an architectural style on top of HTTP. In practice it means: resources identified by URLs, standard methods (GET, POST, PUT, PATCH, DELETE) with their standard meanings, standard status codes, and usually JSON bodies (lesson 02.01 covers designing them).

Strengths:

- **Universal**: every language, browser, proxy, CDN and command-line tool already speaks it. A partner can integrate with `curl` in five minutes.
- **Cacheable**: GET responses can be cached by browsers and CDNs using standard headers.
- **Easy to debug**: requests and responses are readable text; you can inspect them in browser developer tools.
- **Loosely coupled**: clients can ignore fields they don't use, so servers can add fields freely.

Weaknesses:

- **Verbose payloads**: JSON repeats every field name in every object, and numbers are stored as text.
- **No enforced schema**: the contract lives in documentation (often an OpenAPI spec) rather than being checked by the compiler. A typo in a field name is a runtime bug.
- **Over-fetching and under-fetching**: a fixed endpoint returns more than a screen needs, or a screen needs five calls to assemble.
- **Request/response only**: streaming needs extra mechanisms such as server-sent events or WebSockets.

## gRPC

**gRPC** is an RPC (remote procedure call) framework, open-sourced by Google in 2015. You define services and messages in a `.proto` file using **Protocol Buffers** (protobuf), a compact binary serialisation format:

```
service Pricing {
  rpc GetQuote (QuoteRequest) returns (Quote);
  rpc WatchPrices (WatchRequest) returns (stream Price);
}
message QuoteRequest { string ride_id = 1; int32 distance_m = 2; }
message Quote { int64 price_cents = 1; string currency = 2; }
```

A compiler (`protoc`) generates **client stubs** and **server skeletons** in many languages from this file. Calling a remote service then looks like calling a local function: `quote = pricing.GetQuote(request, timeout=0.2)`. The generated code handles serialisation, connection management and errors.

gRPC runs on HTTP/2 and supports four call types:

- **Unary**: one request, one response (like a normal function call).
- **Server streaming**: one request, a stream of responses (subscribe to price changes).
- **Client streaming**: a stream of requests, one response (upload a batch of location points).
- **Bidirectional streaming**: both sides send streams independently (a chat, or live driver locations to a matcher).

It also has built-in **deadlines**: the caller says "I need an answer within 200 ms", and the deadline travels with the call to downstream services, so work for a caller who has given up can be cancelled.

### Why protobuf is small

Protobuf doesn't send field names. Each field is sent as a small **tag** (field number plus a type code, usually 1 byte), then the value. Integers use **varints**, which take 1 byte for values under 128 and 2 bytes for values under 16,384. Strings are a length followed by the bytes.

<div class="ex">
<div class="lab">Worked example</div>

**the same object in JSON and protobuf**

JSON: `{"id":12345,"name":"Ana","active":true}` is 39 bytes.

Protobuf, with id = 1, name = 2, active = 3: id is 1 tag byte + 2 varint bytes = 3; name is 1 tag + 1 length + 3 characters = 5; active is 1 tag + 1 byte = 2. Total: 10 bytes, about a quarter of the JSON.

Parsing is cheaper too: no text scanning, no number parsing from digits, no field-name string comparisons. Across billions of internal calls a day, smaller messages and cheaper parsing save real CPU and network cost.

</div>

~~~dg-bars Size of the same small record in different encodings, in bytes. Binary formats drop field names and text-encoded numbers.
JSON, pretty-printed | 52 | ~52 bytes
JSON, compact | 39 | 39 bytes
JSON, gzipped (one tiny object) | 50 | ~50 bytes, no gain
*Protobuf | 10 | 10 bytes
~~~

The `.proto` schema also gives you **type checking** and a clear contract between teams. Field numbers make it evolvable: add new fields with new numbers and old clients ignore them (lesson 01.11 covers the rules).

The downsides:

- **Browsers can't call gRPC directly**, because they don't expose the low-level HTTP/2 control gRPC needs. You need a proxy layer (gRPC-Web or a translating gateway).
- **Payloads aren't human-readable**; debugging needs tools like `grpcurl`.
- **HTTP caches and CDNs don't understand it**: every call is a POST to a method path.
- **Load balancing needs care.** gRPC keeps long-lived HTTP/2 connections and multiplexes all calls over them. A Layer 4 load balancer balances *connections*, so once a client is connected to one backend, all its calls go there. You need a Layer 7 balancer that balances individual calls, or client-side load balancing (lesson 02.06).

## GraphQL in brief

**GraphQL** lets the client send one query naming exactly the fields it wants, across related objects: "this user's name, their last 10 orders, and each order's first item's image". The server resolves it against a typed schema and returns exactly that shape. It fixes over-fetching and the many-round-trips problem for complex UIs, especially mobile apps on slow networks.

The costs: HTTP caching is harder (queries are usually POSTs with varied bodies), a single clever query can be very expensive (so servers need depth and cost limits), and naive resolvers cause the N+1 query problem (lesson 02.10 shows how DataLoader batches them). GraphQL is common as a **backend for frontend** layer that sits in front of REST or gRPC services.

~~~dg-layers What sits under a gRPC call. Each layer adds one capability; HTTP/3 would swap TCP for QUIC on UDP.
Generated stub | typed function call in your language
Protocol Buffers | compact binary messages from the .proto schema
*HTTP/2 | streams, multiplexing, header compression, flow control
TLS | encryption and server identity
TCP | reliable, ordered byte stream
~~~

## Why REST outside, gRPC inside

| | Edge (public clients) | Internal (service to service) |
| --- | --- | --- |
| Who calls | Browsers, mobile apps, third parties | Your own services |
| Who controls both ends | No one: clients vary and update slowly | You, in one organisation |
| Needs | Universal support, caching, easy debugging, stable public contract | Low latency, small payloads, strict typed contracts, streaming, deadlines |
| Good fit | REST (or GraphQL) | gRPC |

At the edge you don't control the clients. They're written in every language, run behind proxies, and need to be debuggable by outside developers. REST's universality and cacheability win. Inside, you control both ends, call volumes are huge, and the main risks are latency, CPU cost and teams breaking each other's contracts. gRPC's compact encoding, generated typed clients, multiplexing, streaming and deadlines win. An API gateway often translates between the two (lesson 02.05).

~~~dg-matrix REST, gRPC and GraphQL compared on the properties that usually decide the choice.
Property | REST | gRPC | GraphQL
Works from any browser | yes | no, needs a proxy | yes
Standard HTTP caching | yes | no | partial, with care
Compact payloads | no, JSON text | yes, protobuf | no, JSON text
Enforced typed schema | partial, OpenAPI optional | yes | yes
Streaming | partial, SSE or WebSockets | yes, 4 modes | partial, subscriptions
Easy to debug by hand | yes | partial, needs tools | yes
~~~

<div class="ex">
<div class="lab">Worked example</div>

**ride-hailing**

The rider app calls REST endpoints on the API gateway: `POST /v1/rides {pickup, dropoff}`. The gateway authenticates the user and translates the call into gRPC.

Inside, the ride service calls the pricing service with a unary gRPC call and a 150 ms deadline, and the matching service with another. The driver location service streams position updates to the matcher over a bidirectional gRPC stream: thousands of small protobuf messages per second on one HTTP/2 connection, with no per-message connection or header cost.

The rider app gets live driver positions through a WebSocket from the gateway, since browsers and most mobile HTTP stacks handle that more easily than raw gRPC.

</div>

## Common mistakes

- **Exposing gRPC directly to third parties** who expect REST and curl.
- **Assuming HTTP/2 solves everything**: TCP-level head-of-line blocking still hurts on lossy mobile networks.
- **Putting a Layer 4 load balancer in front of gRPC** and wondering why one backend is overloaded.
- **REST without pagination or versioning**, or gRPC with reused field numbers.

## In an interview

Default to REST for client-facing APIs. Mention gRPC for high-volume internal calls, especially with streaming or tight latency budgets, and note the gateway that translates between them. If the client needs flexible, nested data for many screens, mention GraphQL as a BFF layer.

<div class="take">
<div class="lab">Key takeaway</div>

REST wins at the edge because every client, proxy and cache understands it; gRPC wins inside because binary, schema-checked, multiplexed calls with deadlines and streaming are cheaper and safer at high volume. HTTP/2 multiplexes streams on one connection, and HTTP/3 removes TCP's head-of-line blocking.

</div>

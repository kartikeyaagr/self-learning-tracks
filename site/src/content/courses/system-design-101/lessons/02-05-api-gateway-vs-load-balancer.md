---
id: "02.05"
course: "system-design-101"
module: "02"
slug: "api-gateway-vs-load-balancer"
title: "API gateway vs load balancer"
keyIdeas: ["A **load balancer** spreads traffic across identical backends. L4 works on TCP/UDP connections; L7 understands HTTP and can route by path or header.", "An **API gateway** is an L7 entry point with API-specific features: authentication, rate limiting, request transformation, routing to many different services, API keys, aggregation.", "They often sit together: a gateway routes `/orders` to the orders service, and a load balancer spreads requests across that service's instances.", "Keep business logic out of the gateway. It should handle cross-cutting concerns only, or it becomes a bottleneck everyone must change."]
question: "List 3 things an API gateway does that a plain L4 load balancer doesn't."
answer: "A plain Layer 4 load balancer sees only IP addresses and ports, so it cannot: (1) authenticate users by verifying tokens or API keys, (2) route requests by URL path or header to different services, or (3) apply per-user or per-key rate limits and quotas. Gateways also validate requests, translate REST and JSON into gRPC, combine responses from several services into one, and add consistent logging, metrics and tracing."
sources: [{"title": "microservices.io: API Gateway", "url": "https://microservices.io/patterns/apigateway.html", "note": "What a gateway does beyond routing."}, {"title": "F5 (NGINX): What is a load balancer?", "url": "https://www.f5.com/glossary/load-balancer", "note": "Load balancer basics and algorithms."}, {"title": "Hello Interview: API Gateway deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/api-gateway", "note": "Interview view."}]
---

## Two boxes that both "route requests"

Architecture diagrams often show a load balancer and an API gateway in the same spot, just behind the client, so it is easy to treat them as the same thing. Both receive requests and forward them somewhere else. But they solve different problems, and confusing them leads to designs where nobody handles authentication, or where a "load balancer" is expected to enforce per-user quotas it cannot even see.

A **load balancer** spreads traffic for **one service** across many identical instances. It answers: "which copy of this service handles this request?" It cares about health, capacity and even distribution.

An **API gateway** is the single entry point for clients into **many different services**. It answers: "which service handles this request, and what must happen before and after?" It cares about the API surface: routes, clients, keys, quotas, formats.

~~~dg-compare A load balancer spreads traffic across copies of one service; a gateway is the front door that knows the API and routes to many services.
[Load balancer]
Traffic -> LB -(round robin)-> Copies 1, 2, 3 of one service
+ Health checks, failover
+ Very fast, can be Layer 4
- Knows nothing about users or API keys
[API gateway]
Client -> Gateway: auth, rate limit -(/orders)-> Orders service
+ Auth, quotas, validation, translation
+ One public surface for many services
- Extra hop, more latency
- Critical component to keep up
~~~

## What a load balancer does

Load balancing spreads traffic across a group of servers so that none is overloaded, and steers around servers that fail. Load balancers can be dedicated hardware appliances, software (HAProxy, NGINX, Envoy) or managed cloud services (AWS ALB and NLB, Google Cloud Load Balancing). They work at one of two levels:

- **Layer 4** (transport): routes by IP address and port without reading the request. It forwards TCP connections or UDP packets to a backend. It is very fast (millions of packets per second per node) and works for any protocol, but it can't see URLs, headers or users.
- **Layer 7** (application): terminates the HTTP connection, reads the request, and can route by path, host, header or cookie. It can also terminate TLS, add headers, and retry idempotent requests on another backend. It costs more CPU and adds a little latency.

Its core features: **health checks** (stop sending to backends that fail), **algorithms** for picking a backend (round robin, least connections, hashing), **TLS termination**, and **connection management**. Lesson 02.06 covers these in depth.

## What an API gateway does

Without a gateway, every client would need to know dozens of service addresses, make many calls per screen, speak each service's protocol, and authenticate separately with each. Every service would reimplement token checks, rate limits and logging. The gateway gives clients one front door and centralises those cross-cutting concerns:

- **Routing**: `/users/*` to the user service, `/orders/*` to the order service, `/v2/search` to the new search service. Routing by API path is the gateway's core job.
- **Authentication and authorisation**: verify the token (signature, expiry, scopes) once at the edge, so services don't each reimplement it. The gateway usually forwards the verified identity to services in a trusted header.
- **Rate limiting and quotas** per client, per user or per API key: "free tier: 1,000 calls a day; paid: 100 per second".
- **Request validation**: reject malformed requests (bad JSON, missing fields, oversize bodies) before they reach your services.
- **Protocol translation**: accept REST and JSON from the client, call gRPC internally (lesson 02.03).
- **Composition**: one client call that fans out to several services and merges the results, saving round trips on slow mobile networks.
- **Response transformation and caching**: strip internal fields, reshape responses for a client version, cache public GETs.
- **Observability**: consistent access logs, metrics per route and per client, and a trace ID added to every request.
- **API lifecycle**: versioning, API key issuance, developer portals and usage billing, especially for public APIs.

Products include Kong, Tyk, AWS API Gateway, Apigee, Azure API Management, and Envoy-based gateways.

~~~dg-seq One request passing through a gateway. Each step can reject the request before any backend service does work.
Client > Gateway : GET /v1/orders/981 + Bearer token
Gateway > Gateway : validate JWT signature, expiry, scope
Gateway > Gateway : rate limit: user 42 at 37 of 100 per min
Gateway > Gateway : route /v1/orders/* to orders service
Gateway > Orders LB : gRPC GetOrder(981), user=42, trace=ab12
Orders LB > Orders pod 3 : forward
Orders pod 3 -> Gateway : Order message
Gateway -> Client : 200 JSON, X-RateLimit-Remaining: 63
~~~

### How rate limiting works at the gateway

The most common algorithm is the **token bucket**. Each client has a bucket that holds up to B tokens and refills at R tokens per second. Each request takes one token; if the bucket is empty, the request gets **429 Too Many Requests** with a `Retry-After` header. B controls how big a burst is allowed; R controls the sustained rate. With many gateway instances, the counts must be shared (usually in Redis) or split per instance, otherwise a client could get N times its limit by spreading calls across N gateways.

<div class="ex">
<div class="lab">Worked example</div>

**a token bucket in action**

A partner's API key has a bucket of B = 20 tokens refilling at R = 5 per second.

At 12:00:00 the partner fires 30 requests at once. The first 20 succeed (bucket now empty) and the other 10 get 429 with `Retry-After: 1`.

Over the next second the bucket refills 5 tokens. If the partner then sends steadily at 5 per second, every request succeeds. If it sends 8 per second, 3 of every 8 are rejected. The partner's average can never exceed 5 per second, while short bursts of up to 20 are absorbed smoothly.

</div>

## Backends for frontends

A single gateway serving web, iOS, Android and partner APIs tends to grow a tangle of client-specific logic. A common variation is **Backends for Frontends (BFF)**: one gateway per client type. The mobile BFF returns small, pre-composed payloads tuned for slow networks; the web BFF returns richer data; the partner API is stable and versioned slowly. Each BFF is often owned by the team that builds that client, so they can change it without negotiating with everyone else.

~~~dg-tree Backends for frontends: one edge layer per client type, each shaped for its client, all calling the same internal services.
Edge
  Web BFF | rich pages, server-side rendering
  *Mobile BFF | small composed payloads, one call per screen
  Partner API gateway | stable v1 contract, API keys, quotas
Internal services
  Users
  Orders
  Catalogue
  Payments
~~~

## How they fit together

In practice you use both, in layers. The gateway is itself a horizontally scaled service, so there is a load balancer in front of the gateway fleet. Behind the gateway, each backend service runs many instances, so there is a load balancer (or client-side balancing, or a service mesh sidecar) in front of each service too.

~~~dg-layers The usual layering from the internet inward. Gateways handle API concerns; load balancers handle distribution at each tier.
DNS and CDN | resolve api.example.com, serve cached static content
Edge load balancer | Layer 4 or 7, TLS termination, spreads across gateways
*API gateway fleet | auth, rate limits, validation, routing, translation
Service load balancers | one per service, health checks, algorithms
Service instances | stateless copies of each service
~~~

The gateway's main costs:

- **An extra network hop**: typically 1 to 5 ms with auth and routing, more if it calls an external identity provider on every request (cache token verification keys instead).
- **A critical component**: if the gateway is down, everything is down. It must be redundant across zones, with capacity for peak traffic plus a failed zone.
- **A risk of becoming a monolith**: if business logic creeps into the gateway ("just add this discount rule here"), every team ends up deploying through it. Keep it to cross-cutting concerns.

The lines between the two have blurred: Layer 7 load balancers can do path routing and some auth, and many gateways have built-in load balancing. A **service mesh** (Istio, Linkerd) moves load balancing, retries, mutual TLS and metrics into a sidecar proxy next to every service, handling internal (east-west) traffic, while the gateway handles external (north-south) traffic.

| | Load balancer | API gateway |
| --- | --- | --- |
| Distributes across | Identical copies of one service | Different services |
| Understands | Connections (L4) or HTTP (L7) | Your API: routes, clients, keys, quotas |
| Typical features | Health checks, algorithms, TLS termination | Auth, rate limits, validation, translation, composition |
| Added latency | Microseconds (L4) to about 1 ms (L7) | About 1 to 5 ms |
| Examples | AWS NLB/ALB, HAProxy, NGINX | Kong, AWS API Gateway, Apigee |

~~~dg-matrix Which box can do which job. A Layer 4 balancer can't do anything that requires reading the request.
Capability | L4 load balancer | L7 load balancer | API gateway
Spread load, health checks | yes | yes | yes
TLS termination | partial, passthrough or terminate | yes | yes
Route by URL path | no | yes | yes
Verify user tokens | no | partial, some products | yes
Per-user or per-key rate limits | no | partial, basic | yes
REST to gRPC translation | no | no | yes
Compose several services | no | no | yes
~~~

## In interviews

Draw the gateway at the edge, say it handles routing, authentication and rate limiting, put load balancers in front of each service tier, and move on. Don't spend minutes on it unless asked. If asked to go deeper, discuss where the rate limit counters live, how the gateway is made highly available, and what happens to latency.

<div class="ex">
<div class="lab">Worked example</div>

**food delivery app**

The mobile app sends every call to api.example.com. DNS points that at a regional edge load balancer, which terminates TLS and spreads requests across 12 gateway instances in three zones.

The gateway checks the JWT (signature verified locally with a cached public key, about 0.1 ms), applies a 100 requests per minute limit per user using counters in Redis (about 0.5 ms), and routes `/restaurants` to the catalogue service and `/orders` to the order service, translating to gRPC.

For the home screen, the mobile BFF fans out to catalogue, promotions and recent orders in parallel and returns one response, so the phone makes one request instead of three over a 150 ms mobile link.

Behind each service, another load balancer spreads requests across its instances using least connections. If one zone fails, the edge balancer stops sending to its gateways within seconds of failed health checks.

</div>

## Common mistakes

- **Expecting an L4 load balancer to authenticate or rate-limit per user.** It can't see users.
- **Putting business logic in the gateway.**
- **A single gateway instance**, which is a single point of failure.
- **Per-instance rate limits** that let clients multiply their quota by the number of gateways.
- **Calling the identity provider on every request** instead of verifying signed tokens locally.

<div class="take">
<div class="lab">Key takeaway</div>

a load balancer spreads load across copies of one service, using health checks and an algorithm; a gateway is the front door that knows your API and handles routing, auth, rate limits, validation and translation. Real systems use both, in layers, and keep business logic out of the gateway.

</div>

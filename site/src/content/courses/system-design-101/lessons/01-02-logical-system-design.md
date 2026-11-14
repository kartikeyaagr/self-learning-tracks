---
id: "01.02"
course: "system-design-101"
module: "01"
slug: "logical-system-design"
title: "Logical system design"
keyIdeas: ["A logical design shows components and data flow without naming products: client, API layer, services, data stores, queues, caches.", "Start with the simplest working path for each functional requirement, end to end. Then add pieces only where a requirement forces it.", "Draw at one level of detail at a time. The C4 model calls these levels context, containers and components.", "Label arrows with what flows over them (HTTP request, event, SQL query) and whether it's sync or async.", "Keep vendor names out until the end. \"Key-value store\" says what you need; \"Redis\" is an implementation choice you justify later.", "A good logical design lets someone else trace a request from the user to storage and back."]
question: "Draw boxes and arrows for a URL shortener (client, API, service, DB) without naming any vendor."
answer: "Client sends requests to an API service behind a load balancer. On create, the service takes a unique ID from an ID generator, base-62 encodes it, and writes (code → URL) to a database. On redirect, it reads the code from a cache, falls back to the database on a miss, and returns an HTTP redirect. Arrows: client to API (HTTP, sync), API to cache and database (sync reads and writes), API to ID generator (sync)."
sources: [{"title": "The C4 model", "url": "https://c4model.com/", "note": "Drawing architecture at context, container and component levels."}, {"title": "System Design Primer: high level and core components", "url": "https://github.com/donnemartin/system-design-primer#step-2-create-a-high-level-design", "note": "Steps 2 and 3 of the approach."}]
---

## From requirements to boxes

Once the requirements are agreed, the next job is a **high-level design**: the main components and how data flows between them. Think of it as the map you will navigate for the rest of the session. It should be simple enough to fit on one whiteboard, complete enough that every functional requirement has a path through it, and agreed with the interviewer (or your team) before you go deeper.

Jumping straight to boxes is still a mistake. Two small steps first make the drawing almost mechanical.

### Step 1: core entities

List the **nouns** the system stores and passes around. For a Twitter-like product: User, Tweet, Follow, and maybe Media. For a ride-hailing app: Rider, Driver, Ride, Location, Fare. Don't design full schemas yet; just name the entities and the one or two fields that matter for the design (a Tweet has an author and a timestamp; a Follow links two users).

Entities matter because they tell you what has to be stored, and how big it is. A Tweet is small text; Media is large binary data. That alone predicts two different kinds of storage.

### Step 2: the API

Write **one endpoint per functional requirement**. This forces you to think about inputs and outputs, and it becomes the contract every component must fulfil.

| Requirement | Endpoint | Returns |
| --- | --- | --- |
| Post a tweet | `POST /v1/tweets {text, mediaId?}` | the new tweet's ID |
| Follow a user | `POST /v1/follows {followeeId}` | 204 No Content |
| Read the feed | `GET /v1/feed?cursor=...` | 50 tweets and a next cursor |

Two habits worth building now:

- **The current user comes from the auth token, never from the request body.** If `POST /v1/tweets` accepted `{authorId}`, anyone could post as anyone.
- **Paginate every list** with a cursor. "Return the whole feed" doesn't survive a user with 10,000 posts in it.

With entities and endpoints written down, the design becomes a sequence of small questions: for each endpoint, which components does a request touch, and what state changes?

## Draw one request at a time

Take each endpoint and trace it. For a URL shortener's redirect:

~~~dg-seq Tracing one endpoint end to end: a redirect touches the load balancer, the service, the cache and, on a miss, the database.
Browser > Load balancer : GET /abc123
Load balancer > Link service : forward
Link service > Cache : get abc123
Cache -> Link service : miss
Link service > Database : SELECT url WHERE code = abc123
Database -> Link service : https://example.com/long/path
Link service > Cache : set abc123 (TTL 1 day)
Link service -> Browser : 301 Location: https://example.com/long/path
~~~

Then the create path. Then any background work. When every endpoint has a path, the high-level design is done, and every box on the board is justified by at least one requirement. If a box isn't on any path, delete it.

## Logical before physical

A **logical** design names roles, not products: client, load balancer, API service, feed service, relational database, key-value cache, message queue, object store. A **physical** design picks the products: NGINX, Postgres, Redis, Kafka, S3.

Name roles first. Naming a product too early ("we'll use Kafka") invites questions you can't yet justify ("why not SQS?") and pulls the conversation into product trivia. Choose products at the end, when a requirement makes the choice obvious: "we need replayable, ordered event streams consumed by three teams, so Kafka" is a justified choice; "Kafka because it's popular" is not.

~~~dg-compare The same design at two levels. Agree the logical roles first; pick products only when a requirement forces the choice.
[*Logical]
Client -(HTTPS)-> API service -(key lookup)-> Link store
+ Easy to discuss and change
+ Focuses on what each part must do
- Hides cost and operational details
[Physical]
Mobile app -(HTTPS)-> Go service on Kubernetes -(GetItem)-> DynamoDB + Redis
+ Concrete enough to build and cost
- Invites product debates too early
~~~

### Good habits while drawing

- **Build up one endpoint at a time** instead of drawing everything at once.
- **Label every arrow** with what flows over it, and whether it is **synchronous** (request and wait for a response) or **asynchronous** (fire an event onto a queue and move on). That one distinction decides a lot about latency and failure handling.
- **Write key fields next to each data store**, not full schemas: "links: code (PK), url, created_at".
- **Mark where state lives.** Stateless services can be copied freely; stateful boxes (databases, caches) are where the hard problems are.
- **Defer optimisations.** When you think "we'll need a cache here", say it, note it, and move on. Add it once the simple path works and you can say which requirement needs it.

## The C4 model: zoom levels for diagrams

Diagrams fail when they mix levels: a box for "the whole payments platform" next to a box for "the retry helper function". The **C4 model** gives a vocabulary for drawing at a consistent level of detail. It defines four zoom levels, like a map that goes from country to street:

~~~dg-tree The C4 model's zoom levels. Each level zooms into one box of the level above.
System context: your system as one box
  Containers: apps and data stores
    API service
      Components: controllers, feed builder, auth client
        Code: classes and functions
    *Feed worker
    Postgres database
    Redis cache
~~~

- **System context**: your system as a single box, with the people who use it and the external systems it talks to (payment provider, email service). Good for non-technical audiences.
- **Container**: zoom into your system. A container is anything that runs or stores data separately: a web app, a single-page app, a mobile app, a backend service, a database, an S3 bucket. The diagram shows how responsibilities are split, the main technology choices, and how containers communicate. Every team should have one.
- **Component**: zoom into one container to show its major internal parts (modules, controllers, a repository layer).
- **Code**: classes and functions. Rarely worth drawing by hand; IDEs generate these.

C4 deliberately leaves **deployment details** (clusters, load balancers, replication, failover, regions) off the container diagram, because they differ between production and staging. They go on separate deployment diagrams. In an interview, your high-level design is roughly a container diagram with a few deployment notes (a load balancer, replicas) added once they matter.

## Then go deep where it matters

After the high-level design comes a **deep dive** on two or three components, chosen because they are the hardest or most important part of this specific system: feed generation for Twitter, the ID generator for a URL shortener, matching for ride-hailing. Then come scaling concerns (load balancing, caching, sharding, replication) and failure modes.

The logical design is what makes that possible. You can only discuss bottlenecks once everyone can trace a request end to end, and you can only choose where to go deep once you can see all the parts.

## Worked example: URL shortener, from requirements to boxes

<div class="ex">
<div class="lab">Worked example</div>

**Entities**

Link: code (7 characters, primary key), long_url, created_at, expires_at. That's under 1 KB per row.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**API**

POST /v1/links {url, alias?, expiresAt?} → 201 {code}. GET /{code} → 301 or 302 redirect, or 404 if unknown or expired.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**Create path**

Client → load balancer → link service. The service gets a unique number from an ID generator (so two creates never collide), encodes it in base 62 (0-9, a-z, A-Z) to get a 7-character code, stores (code → long URL) in the database, and returns the short link.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**Redirect path**

Client → load balancer → link service → cache. On a hit, return the redirect immediately. On a miss, read from the database, fill the cache, return the redirect. Reads outnumber writes 100 to 1, so the cache absorbs most of the load.

</div>

~~~dg-flow The URL shortener's high-level design: two paths, five roles, every box justified by a requirement.
[Create] Client -(POST /links)-> Link service -(next id)-> ID generator
[Create] Link service -(insert code, url)-> *Database
[Redirect] Client -(GET /abc)-> Link service -(get)-> Cache -(miss)-> *Database
~~~

## Common mistakes

- **Boxes without paths.** A "search service" no requirement uses.
- **Unlabelled arrows.** Nobody knows what flows over them or whether the caller waits.
- **Product names as design.** "Kafka, Redis, Mongo" isn't an architecture.
- **Mixing zoom levels.** A whole subsystem and a helper function as equal boxes.
- **Drawing the scaled version first.** Start with the simplest design that meets the requirements; scale it when a number demands it.

<div class="take">
<div class="lab">Key takeaway</div>

list entities and one API endpoint per requirement, then trace each endpoint through generic roles (logical design). Label arrows, mark where state lives, keep one zoom level, and pick products only when a requirement makes the choice obvious.

</div>

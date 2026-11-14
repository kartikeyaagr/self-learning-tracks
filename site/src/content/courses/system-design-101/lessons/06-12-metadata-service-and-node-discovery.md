---
id: "06.12"
course: "system-design-101"
module: "06"
slug: "metadata-service-and-node-discovery"
title: "Metadata service and node discovery"
keyIdeas: ["Clusters need a source of truth for metadata: which nodes are alive, which node owns which partition, current config, who's leader.", "Store it in a small, strongly consistent, highly available service (ZooKeeper, etcd, Consul) backed by consensus, so all nodes agree.", "Features used: ephemeral nodes that vanish when a session dies (liveness), watches to get notified of changes, sequential nodes for leader election and locks.", "Service discovery: services register their address; clients look them up or a load balancer watches the registry.", "Keep the metadata service off the data path. Clients cache metadata and only consult the service on changes."]
question: "Why store cluster metadata in a consensus-backed service rather than a normal database?"
answer: "Because every node must see the same answer to questions like \"who is the leader\" or \"who owns this shard\", even during server failures and failovers, and a wrong answer causes split brain and divergent data. A consensus-backed service commits each change to a majority before acknowledging it, so it never loses or reorders acknowledged updates when a server fails, unlike a single database or one with asynchronous replicas. It also provides sessions with ephemeral nodes or leases that tie membership to liveness automatically, watches that push change notifications instead of polling, and monotonically increasing version numbers (zxid, etcd revision) usable as fencing tokens. The cost is small capacity and slower writes, which is fine for metadata."
sources: [{"title": "ZooKeeper: Overview", "url": "https://zookeeper.apache.org/doc/current/zookeeperOver.html", "note": "Coordination and metadata storage."}, {"title": "microservices.io: Service registry", "url": "https://microservices.io/patterns/service-registry.html", "note": "How services find each other."}, {"title": "Hello Interview: ZooKeeper deep dive", "url": "https://www.hellointerview.com/learn/system-design/deep-dives/zookeeper", "note": "Coordination for interviews."}]
---

## The problem

Every distributed system carries a small set of **shared facts** that all its parts must agree on:

- **Membership**: which nodes are currently in the cluster.
- **Leadership**: who is leader of each shard or each job.
- **Placement**: which shard (range of keys) lives on which node. This is often called the **shard map** or routing table.
- **Configuration**: feature flags, limits, timeouts, the current schema version.
- **Discovery**: where each running instance of each service can be reached (IP and port).

These facts are **metadata**: data about the system, not the users' data. They are small (kilobytes to a few megabytes), they change rarely (seconds to hours between changes), and they are read constantly. But they must be **correct**, because disagreement is dangerous. If two storage nodes each believe they own shard 7, both accept writes for it and the data diverges. If half the routers think node N4 is alive and half think it is dead, requests go to a node that is no longer serving. A wrong shard map can be worse than no shard map.

So metadata lives in a special kind of store: a **coordination service** backed by consensus.

## Coordination services: ZooKeeper

ZooKeeper is the classic example, built specifically to hold this kind of data.

### The data model

Data is a tree of **znodes**, addressed by paths like a file system: `/services/payments/instance-3`. Each znode holds a small amount of data (the limit is about 1 MB, but typical values are bytes to kilobytes) and has a **version number** that increases on every change. Three special features make it a coordination tool rather than just a small database:

- **Ephemeral znodes** exist only as long as the **session** of the client that created them. The client keeps the session alive with heartbeats. If the client crashes or loses its connection for longer than the session timeout (often 2 to 30 s), the session expires and all its ephemeral znodes vanish. This makes membership automatic: a node is "in the cluster" exactly as long as it is alive and connected.
- **Watches** notify a client when a znode or its children change, so clients do not have to poll. A watch fires once; the client re-reads and sets a new watch.
- **Sequential znodes** get a monotonically increasing counter appended to their name by the server (`lock-0000000017`). This gives a total order to anything that creates them, which is the building block for locks, queues and elections.

~~~dg-tree A typical ZooKeeper tree for one cluster. Ephemeral nodes vanish when their owner's session dies; sequential nodes get increasing suffixes.
/
  /config
    payments | timeouts, flags; services watch this
  /services
    /api
      *i-7 | ephemeral: 10.0.3.7:8080
      i-9 | ephemeral: 10.0.4.2:8080
  /election
    n_0000000017 | ephemeral sequential: current leader
    n_0000000018 | watches n_17
  /shards
    7 | owner N2, version 41
~~~

### How it runs

ZooKeeper runs as an **ensemble** of servers, usually 3 or 5. Writes go through a leader and are committed once a majority has them, using ZooKeeper's **ZAB** protocol (similar to Raft, lesson 06.06). Every committed change gets a **zxid**, a transaction ID that increases across the whole ensemble.

Reads are served by whichever server the client is connected to, straight out of RAM, which makes them fast but possibly slightly behind the leader. A client that needs the very latest value calls `sync` first, which makes its server catch up with the leader. ZooKeeper stays available while a majority is up, keeps all data in memory (with a write-ahead log and snapshots on disk), and performs best for read-heavy workloads, around 10 reads per write or more.

Its guarantees:

- **Sequential consistency**: updates from one client are applied in the order sent.
- **Atomicity**: an update either fully succeeds or fully fails.
- **Single system image**: a client sees the same view whichever server it connects to, and never goes backwards in time even if it reconnects to another server.
- **Durability**: once applied, an update persists until overwritten.
- **Timeliness**: a client's view is up to date within a bounded time.

### Classic recipes

A handful of patterns built from these primitives cover most uses.

- **Leader election**: each candidate creates an ephemeral sequential node under `/election`. The one with the lowest number is leader. Each other candidate watches **only the node just before it**, not the leader. When the leader dies, its node vanishes and only the next candidate is notified; it checks that it is now the lowest and takes over. Watching the leader directly would wake every candidate at once (a herd effect) for no reason.
- **Locks**: the same pattern under `/locks/resource`. The zxid or the node's version doubles as a **fencing token** (lesson 06.08).
- **Service discovery**: each instance registers an ephemeral node under `/services/name`; clients list the directory and watch it.
- **Configuration**: a znode holds the config; every service watches it and reloads on change.

~~~dg-seq ZooKeeper leader election: each candidate watches only its predecessor, so a leader's death wakes exactly one node.
Node A > ZooKeeper : create /election/n_ (ephemeral sequential)
ZooKeeper -> Node A : n_17, lowest, so A leads
Node B > ZooKeeper : create, gets n_18, watch n_17
Node C > ZooKeeper : create, gets n_19, watch n_18
Node A > Node A : crashes, session expires
ZooKeeper -> Node B : watch fired: n_17 deleted
Node B > Node B : n_18 now lowest, become leader
~~~

### Alternatives

- **etcd**: a key-value store with Raft consensus, a flat key space with prefixes rather than a tree, **leases** instead of sessions, watches over key ranges, and a global **revision** number on every change. It is the store behind Kubernetes, which keeps every object (pods, services, deployments) in it.
- **Consul**: Raft for its strongly consistent catalogue and key-value store, plus SWIM gossip (lesson 06.11) for membership and failure detection among potentially thousands of agents. It adds built-in health checks and DNS-based discovery.

## Service registries

In a microservice system, instances change constantly: autoscaling adds and removes them, containers get new IP addresses on every restart, deployments replace them. Hard-coding addresses is impossible, so a **service registry** keeps a database of services and their live instances.

How instances get in:

- **Self-registration**: each instance registers itself on startup, sends heartbeats (or answers health checks) while running, and deregisters on clean shutdown. If it crashes, missed heartbeats remove it.
- **Third-party registration**: the platform registers instances on their behalf. Kubernetes does this implicitly: the control plane tracks which pods are ready and updates the endpoints for each service.

How clients find them:

- **Client-side discovery**: the client queries the registry (and caches the result), then picks an instance itself, applying its own load balancing. Netflix Eureka with its client libraries works this way.
- **Server-side discovery**: the client calls a fixed address (a load balancer or router), which queries the registry and forwards the request. Kubernetes Services and cloud load balancers work this way.

~~~dg-compare Client-side versus server-side discovery. Both read the same registry; they differ in who picks the instance.
[Client-side]
Client -(query registry)-> Instance list -(client picks)-> Instance
+ No extra hop
+ Client can balance smartly
- Discovery logic in every client
- Every language needs a library
[*Server-side]
Client -> Load balancer -(queries registry)-> Instance
+ Clients stay simple
+ One place to change routing
- Extra hop and component
~~~

Examples of registries: Netflix Eureka, ZooKeeper, Consul, etcd and Kubernetes' own endpoints. The drawbacks: the registry is another critical component that must be highly available, and clients cache its data, which goes stale if it fails. Eureka deliberately chose availability: when many instances stop heartbeating at once (which looks more like a network problem than mass death), it enters a **self-preservation** mode and stops expiring them, preferring a slightly stale list to an empty one.

## Why not a normal database?

You could store the shard map in a PostgreSQL table. Here is why you usually should not.

| Need | Consensus-backed store | Typical database |
| --- | --- | --- |
| No conflicting answers through failover | Yes: every change is on a majority before it is visible | Async replica promoted on failover can lose or reorder recent changes |
| Membership tied to liveness | Yes: sessions, ephemeral nodes, leases | No: needs your own heartbeat and cleanup job |
| Change notification | Yes: watches | Polling, or a separate change stream |
| Increasing ID per change | Yes: zxid, etcd revision | Possible with a sequence, but not tied to failover safety |
| Survives a minority of servers failing | Yes, automatically | Depends on failover setup, often minutes |

- **Strong consistency under failure.** A consensus-backed store replicates each change to a majority before acknowledging it, so it never shows two nodes conflicting answers about the leader or a shard's owner, even across failovers. A single database is a single point of failure; a replicated one with asynchronous replicas can lose acknowledged updates or show them out of order when a replica is promoted (lesson 06.03).
- **Sessions and ephemeral data.** Membership is tied to liveness automatically. With a normal table you need heartbeat columns, a reaper job and careful clock handling.
- **Watches.** Change notifications reach every interested client within milliseconds, instead of polling every few seconds.
- **Ordering and fencing.** Every change gets a monotonically increasing ID (zxid, etcd revision), usable directly as a fencing token.

The trade-off: these stores are small (all data in memory, typically a few gigabytes at most; etcd's default limit is 2 GB) and relatively slow for writes (each write is a consensus round, often thousands per second rather than hundreds of thousands). So they hold **metadata, never application data**. Putting user records in ZooKeeper is a classic mistake.

<div class="ex">
<div class="lab">Worked example</div>

**shard map for a key-value store**

A storage cluster keeps its shard-to-node map in etcd under `/shards/{id}`. Each storage node holds a 10 s lease and registers under `/nodes/{name}`. A controller watches `/nodes`.

Node N4 crashes at t = 0. Its lease expires at about t = 10 s, and etcd deletes `/nodes/N4`. The controller's watch fires. In one etcd transaction it reassigns N4's 40 shards to their replicas and bumps each shard's revision.

Routers watch `/shards` and update their routing tables within milliseconds. Requests are stamped with the shard-map revision the router used; storage nodes reject requests carrying an older revision than they know, so a router with a stale view gets an error and refreshes rather than writing to the wrong owner.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**why polling a database falls short**

The same cluster uses a PostgreSQL table for the shard map, with 500 routers polling every 2 s: 250 queries per second just to check for changes, and changes take up to 2 s to be noticed.

Worse, during a database failover to an async replica, the last shard reassignment (written 50 ms before the crash) is lost. Half the routers already read it and route shard 12 to N7; the new database says N2. Both nodes accept writes for shard 12 until someone notices. With etcd, the reassignment would either be committed on a majority (and survive) or not acknowledged at all.

</div>

## Common mistakes

- **Storing application data** in the coordination service.
- **Treating ZooKeeper reads as linearizable** without `sync`.
- **Watching the leader node** in an election recipe, causing a herd on every failover.
- **Ignoring session expiry** in clients, so a client continues acting as leader after its ephemeral node has gone.
- **Clients that cannot survive a registry outage.** Cache the last known list and keep using it.

## In an interview

When your design has shards, leaders or dynamic membership, say where that metadata lives: "a 3- or 5-node etcd cluster holds the shard map and node leases; routers watch it; every request carries the map revision". Then add one sentence on why: consistent through failures, liveness-tied membership and change notifications.

<div class="take">
<div class="lab">Key takeaway</div>

put small, critical cluster metadata (membership, leaders, shard maps, configuration, service endpoints) in a consensus-backed store such as ZooKeeper, etcd or Consul. It stays consistent through failures, ties membership to liveness with sessions or leases, notifies watchers of changes, and gives increasing versions usable as fencing tokens. Keep application data out of it.

</div>

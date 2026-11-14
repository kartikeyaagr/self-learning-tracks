---
id: "04.04"
course: "system-design-101"
module: "04"
slug: "graph-database-decision-boundary"
title: "Graph database decision boundary"
keyIdeas: ["Graph databases (Neo4j, Neptune) store nodes and edges directly, so traversing relationships costs proportional to the edges followed, not the table size.", "Natural fits: \"friends of friends who like X\", recommendation paths, fraud rings, network topology, permission inheritance.", "In SQL, each extra hop is another self-join, which gets slow and awkward for variable-depth queries (recursive CTEs help, up to a point).", "Many \"social graph\" products still use sharded SQL or key-value stores for simple one-hop queries (followers, following) because those are just lists.", "Use a graph database when multi-hop queries are central, not just because data has relationships."]
question: "Which query is painful in SQL but natural in a graph database?"
answer: "Multi-hop and variable-length traversals, such as \"friends of friends of friends who like jazz\", the shortest path between two users, or finding every account connected to a known fraudster through any chain of shared cards or devices within 4 hops. In SQL each hop is another self-join on the edges table with fast-growing intermediate results, and a variable number of hops needs an awkward, hard-to-optimise recursive CTE. A graph database stores each node's relationships directly (index-free adjacency), so a query like MATCH (a)-[:FOLLOWS*1..3]->(b) follows pointers hop by hop with the hop count as a simple parameter, and offers path-finding built in."
sources: [{"title": "Neo4j: Graph database concepts", "url": "https://neo4j.com/docs/getting-started/graph-database/", "note": "Nodes, relationships, traversals."}]
---

## The property graph model

A **graph database** stores data as a network instead of as tables. The most common model, the **property graph**, has three ingredients:

- **Nodes** are entities: a person, a product, a city, a bank card. Each node can carry one or more **labels** such as `Person` or `Card`, and any number of key-value **properties** (`name: "Ana"`, `country: "IN"`).
- **Relationships** (also called edges) connect two nodes. Each has a **type** (`FOLLOWS`, `BOUGHT`, `USED_CARD`), a **direction**, and optional properties of its own (`since: 2021`, `amount: 40`).
- **Properties** live on both nodes and relationships.

The key design choice is that **relationships are stored directly, as first-class data**. In Neo4j, each node record holds a pointer to its first relationship, and relationship records are chained together, so moving from a node to its neighbours is a matter of following pointers. This is called **index-free adjacency**: the cost of one hop depends on how many neighbours that node has, not on how big the whole database is. A relational join, by contrast, looks up matching rows in an index whose depth grows (slowly) with the size of the table, and the planner must decide how to join at every step.

The practical consequence is that a graph database can traverse millions of relationships per second on one machine, and the cost of a query depends on the part of the graph it touches, not on the total graph size.

~~~dg-tree A small property graph seen from Alice: nodes with labels and properties, and typed relationships with their own properties.
Alice | Person, country IN
  FOLLOWS since 2021 | to Bob
    Bob | Person
      FOLLOWS | to Carol
        *Carol | Person, likes jazz
          LIKES | to Post 9
  USED_CARD | to Card 4411
    Card 4411 | also used by Dave
~~~

## Querying a graph

Graph databases have their own query languages. **Cypher** (Neo4j, and the basis of the ISO standard GQL) describes patterns with ASCII-art arrows. Gremlin is a step-by-step traversal language used by several other systems.

```
MATCH (me:Person {id: 1})-[:FOLLOWS*2..3]->(fof:Person)
WHERE fof.likes = 'jazz' AND NOT (me)-[:FOLLOWS]->(fof)
RETURN DISTINCT fof.name LIMIT 20
```

This reads: start at person 1, follow 2 to 3 `FOLLOWS` relationships, keep people who like jazz and whom I do not already follow. The `*2..3` is a **variable-length path**: the number of hops is part of the pattern, not hard-coded into the query's structure.

## Where SQL struggles

In SQL, a relationship is a foreign key, and following it is a **join**. One or two hops are perfectly fine with good indexes; this is what relational databases do all day. The trouble starts with queries that have **many hops** or an **unknown number of hops**:

- "Friends of friends of friends who like jazz."
- "Shortest path between two people" (degrees of separation).
- "Is account A connected to account B through any chain of shared devices, cards or addresses?" (fraud rings).
- "Every part that eventually depends on this component" (deep bill-of-materials) or "everyone who reports up to this VP" (org charts).

With a `follows(follower_id, followee_id)` table, three hops is three self-joins:

```
SELECT DISTINCT f3.followee_id
FROM follows f1
JOIN follows f2 ON f2.follower_id = f1.followee_id
JOIN follows f3 ON f3.follower_id = f2.followee_id
WHERE f1.follower_id = 1;
```

That is readable for a fixed three hops. But "anywhere from 1 to 6 hops", or "until you reach account B", needs a **recursive common table expression** (`WITH RECURSIVE`), which is awkward to write, hard to stop early, hard for the planner to estimate, and prone to revisiting the same nodes along different paths.

~~~dg-compare The same three-hop question in SQL and in a graph query language.
[SQL self-joins]
follows f1 -(join)-> follows f2 -(join)-> follows f3
- One join per hop, fixed in the query
- Variable hops need WITH RECURSIVE
- Planner struggles to estimate rows
+ Works in your existing database
[*Graph traversal]
(me)-[:FOLLOWS*1..3]->(x)
+ Hop count is a parameter
+ Follows stored pointers per node
+ Shortest path is built in
- A second database to run
~~~

### The explosion of intermediate results

The deeper problem is not syntax; it is **fan-out**. If each person follows 200 others, one hop reaches 200 people, two hops reach up to 40,000 (200 × 200), and three hops up to 8 million, before removing duplicates. Every database, graph or relational, must deal with that growth. The difference is in how efficiently each hop is taken and whether the engine can prune, stop early or search from both ends at once (**bidirectional search**, which meets in the middle and turns 200^4 into 2 × 200^2 for a four-hop path).

~~~dg-bars How many paths a friends-of-friends query may touch when each person follows about 200 others.
1 hop | 200 | 200
2 hops | 40000 | 40,000
3 hops | 8000000 | 8 million
*4 hops | 1600000000 | 1.6 billion
~~~

This is also why "6 degrees of separation" queries are expensive in any system: within 4 or 5 hops, a well-connected social graph reaches most of its members.

## Why interviews rarely need one

Reaching for a graph database is a common interview mistake. The word "social" in a prompt does not mean "graph database". Even companies built on social graphs keep their core relationships in relational or key-value storage: Facebook stores its social graph in sharded MySQL, behind a graph-shaped caching and API layer called TAO; Twitter built its follow graph service (FlockDB) on MySQL too. Most product features need exactly one hop:

- "Who do I follow?" and "who follows me?": an indexed edges table, `follows(follower_id, followee_id)`, with an index on each column (or the table stored twice, sharded once by each column).
- Follower and following **counts** are stored denormalised on the user row rather than counted on each request.
- "Do I follow this person?": a primary-key lookup on `(follower_id, followee_id)`.

One-hop queries on an indexed table take a millisecond and scale by sharding on the user ID (lesson 04.05). A graph database adds a new system to run and, as the next section explains, is much harder to shard.

## The decision boundary

Use a graph database when:

- The core value of the feature is **multi-hop traversal or path finding**, and the number of hops is variable.
- **Relationships themselves carry rich data** and are queried in flexible, exploratory ways.
- Typical examples: fraud detection over shared identifiers, network and IT dependency analysis ("what breaks if this switch fails?"), knowledge graphs, access-control graphs ("can this user reach this resource through any group?"), and recommendation exploration.

Do not use one when:

- Queries are mostly **one or two hops** with known shapes.
- You need **huge write throughput** or simple horizontal scaling. Partitioning a highly connected graph across machines is fundamentally hard: whatever cut you make, many edges cross machines, and every traversal across a cut becomes a network hop. Most graph databases scale reads with replicas but keep the full graph on each machine.
- The data is mostly **aggregations** ("total sales per region"), which graph engines are not built for.

~~~dg-matrix Common relationship questions and whether a graph database is the right tool.
Question | Hops | Graph DB worth it?
Who do I follow? | 1 | no, indexed edges table
Mutual friends with X | 2, fixed | no, two indexed lookups and intersect
People you may know | 2 to 3 | partial, often batch-computed
Fraud ring through shared devices | variable, up to 6 | yes
Shortest path between two users | variable | yes
What depends on this service? | variable, deep | yes
~~~

A middle path that many teams use: keep the system of record in Postgres or a key-value store, and **export** the relationships into a graph database (or a graph processing framework run in batch) for the specific analytical questions that need deep traversal. The graph is then a derived, read-mostly copy that can be rebuilt.

<div class="ex">
<div class="lab">Worked example</div>

**fraud ring detection**

A payments company models accounts, cards, devices, phone numbers and IP addresses as nodes, with relationships like `(:Account)-[:USED]->(:Card)` and `(:Account)-[:LOGGED_IN_FROM]->(:Device)`. About 50 million accounts and 400 million relationships.

Question: "which accounts are within 4 hops of this known fraudster through shared cards or devices?" In Cypher: `MATCH (f:Account {id: $id})-[:USED|LOGGED_IN_FROM*1..4]-(a:Account) RETURN DISTINCT a`. Because most cards and devices link only a few accounts, the traversal touches a few thousand nodes and returns in tens of milliseconds.

In SQL, it is four rounds of self-joins across several link tables (account to card, card to account, account to device, device to account), with the combinations of link types multiplying the query size. A recursive CTE can do it but is slow and hard to bound.

The main transaction ledger stays in Postgres. The graph is a separate store fed by account and login events, used by the fraud team's scoring service.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the follow graph that does not need a graph database**

A social app has 100 million users with an average of 300 follows each: 30 billion edges. Every query in the product is one hop: list followers, list following, check follow status, show counts.

Store edges twice, as `following(user_id, followee_id, created_at)` sharded by `user_id`, and `followers(user_id, follower_id, created_at)` sharded by `user_id`. Each list is a single-shard range read; each check is a primary-key lookup. At about 40 bytes per row and two copies, that is about 2.4 TB, spread over 32 shards at 75 GB each.

"People you may know" (friends of friends not yet followed) is computed nightly in a batch job and stored per user, so no live multi-hop query is needed.

</div>

## Common mistakes

- **Choosing a graph database because the data "is a graph".** All data with relationships is a graph; the question is whether you query it with deep traversals.
- **Ignoring fan-out.** Deep traversals on high-degree nodes (celebrities) explode in any engine; cap the depth and the per-node expansion.
- **Planning to shard a graph database like a key-value store.**
- **Moving the system of record into a graph database** for one analytical feature.

## In an interview

If the prompt involves relationships, ask how many hops the core queries need. One or two: an indexed edges table, sharded by user, with denormalised counts. Variable and deep: consider a graph database as a separate, derived store, and mention how you keep it in sync and how you bound traversal cost.

<div class="take">
<div class="lab">Key takeaway</div>

graph databases store relationships as first-class data so that variable, multi-hop traversals and path queries are natural and fast. Most product features only need one or two hops, which an indexed edges table in SQL or a key-value store handles well and scales more easily.

</div>

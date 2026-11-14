---
id: "03.09"
course: "system-design-101"
module: "03"
slug: "database-ticket-servers"
title: "Database ticket servers"
keyIdeas: ["A ticket server is a dedicated database whose only job is handing out unique IDs, using auto-increment.", "Flickr used two MySQL ticket servers: one generated odd numbers, the other even (`auto_increment_increment = 2`, different offsets). Either can fail without stopping ID generation.", "Simple and gives compact, roughly ordered numeric IDs.", "Downsides: an extra network hop per ID, and the servers are a central dependency. IDs from the two servers aren't strictly time-ordered.", "Alternatives: Snowflake-style IDs generated locally, or fetch ranges of IDs in blocks to cut round trips."]
question: "How do two ticket servers avoid handing out the same ID?"
answer: "They split the number space so it can never overlap. Both servers set auto_increment_increment = 2, but server A uses auto_increment_offset = 1 and issues only odd numbers (1, 3, 5...) while server B uses offset 2 and issues only even numbers (2, 4, 6...). Each server uses REPLACE INTO on its one-row table plus LAST_INSERT_ID() to get the next number, so neither needs to know what the other has issued and there is no replication or coordination between them. Clients round-robin between the two and fail over to the other if one is down. The counters drift apart, so IDs are unique and roughly, not strictly, increasing. With N servers you use increment N and offsets 1 to N."
sources: [{"title": "Flickr: Ticket Servers", "url": "https://code.flickr.net/2010/02/08/ticket-servers-distributed-unique-primary-keys-on-the-cheap/", "note": "Unique IDs from two MySQL servers."}]
---

## Why IDs get hard after sharding

On a single database, IDs are a solved problem. An auto-increment column (`AUTO_INCREMENT` in MySQL, a sequence or `IDENTITY` column in Postgres) hands out 1, 2, 3 and so on, and the database guarantees no two rows get the same number. The counter lives in one place, so uniqueness is trivial.

Once data is **sharded** (split across many databases, lesson 03.10 and Module 4), each database has its own counter. Shard 1 and shard 2 both happily create a photo with ID 1,000. That breaks anything that assumes an ID identifies one thing: URLs like `/photos/1000`, caches keyed by ID, references from other tables, and above all moving a row from one shard to another, where it would collide with a different row.

Flickr hit exactly this in the 2000s. Its data was sharded across many MySQL databases, each set up as a master-master pair (two servers that both accept writes and replicate to each other). Users' data sometimes had to move between shards, so IDs had to be unique **globally**, not just per shard. Lesson 04.08 looks at this problem in general; this lesson looks at one simple, durable solution.

## The requirements

Flickr's IDs needed to be:

- **Globally unique** across every shard, forever.
- **Compact**: 64 bits, so indexes on ID columns stay small and fit in memory.
- **Roughly increasing**, so new rows go at the end of B-tree indexes (lesson 03.04) and sorting by ID is close to sorting by time.
- **Highly available**: if ID generation stops, uploads stop.

## Options Flickr rejected

- **GUIDs/UUIDs**: unique without any coordination, but 128 bits is twice the size, and random values scatter inserts across the whole B-tree, causing page splits and poor cache use (lesson 04.09). Flickr's performance depended on indexes fitting in memory. Sequential IDs also made debugging and reporting easier.
- **Consistent hashing with GUID keys**, as in Amazon's Dynamo: suits write-optimised key-value stores, not MySQL tuned for fast random reads by primary key.
- **One central database with a normal auto-increment table**, where each ID is a new row: at 60+ photos per second, plus comments, favourites, tags and groups, that table would grow by millions of useless rows a day.

## The ticket server

A **ticket server** is a tiny, dedicated MySQL database whose only job is to hand out numbers. Its table looks like this:

```
CREATE TABLE Tickets64 (
  id   BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  stub CHAR(1) NOT NULL DEFAULT '',
  PRIMARY KEY (id),
  UNIQUE KEY stub (stub)
);
```

It always holds exactly **one row**. To get a new ID, a client runs two statements on the same connection:

1. `REPLACE INTO Tickets64 (stub) VALUES ('a');` Because `stub` has a unique key and the value `'a'` already exists, `REPLACE` deletes the old row and inserts a new one. The insert takes the next auto-increment value.
2. `SELECT LAST_INSERT_ID();` returns the ID generated **by this connection**, even if other clients are generating IDs at the same moment. MySQL tracks it per connection, so there is no race.

The table never grows past one row, and MySQL's auto-increment mechanism guarantees that each call gets a new number. Separate tables give separate sequences (one for photos, one for accounts, one for groups), or several tables can share one sequence.

~~~dg-seq Getting one ID from a ticket server, then using it to insert a photo on any shard.
App > Ticket DB : REPLACE INTO Tickets64 (stub) VALUES ('a')
Ticket DB > Ticket DB : delete row, insert new row, id 4,816,001
App > Ticket DB : SELECT LAST_INSERT_ID()
Ticket DB -> App : 4,816,001
App > Photo shard 7 : INSERT photo id 4,816,001
Photo shard 7 -> App : OK
~~~

## Two servers without coordination

One ticket server is a **single point of failure**: if it is down, nothing can be created. The obvious fix, two servers replicating to each other, is slow and fragile at this write rate and risks both handing out the same number during a replication delay.

Flickr's trick avoids coordination completely. It runs two ticket servers and **splits the number space** between them using two MySQL settings:

- `auto_increment_increment`: how much the counter goes up each time.
- `auto_increment_offset`: the first value.

| Server | increment | offset | Sequence |
| --- | --- | --- | --- |
| A | 2 | 1 | 1, 3, 5, 7, 9... |
| B | 2 | 2 | 2, 4, 6, 8, 10... |

Server A only ever produces odd numbers and server B only even numbers, so they can **never** produce the same number, no matter what happens to the network between them. They do not talk to each other at all. Clients round-robin between the two and skip whichever is down.

~~~dg-timeline Two ticket servers serving requests independently. They never coordinate, and their counters drift apart without any harm.
[Server A, odd] 10:00:01 | issues 1,001
[Server B, even] 10:00:01 | issues 1,002
[Server A, odd] 10:00:02 | issues 1,003
[Server B, even] 10:00:03 | *goes down for maintenance
[Server A, odd] 10:00:05 | issues 1,005
[Server A, odd] 10:00:06 | issues 1,007
[Server B, even] 10:00:07 | back, issues 1,004
~~~

### The price: roughly ordered, not strictly ordered

The two counters drift apart. If server B is down for an hour, server A races ahead; when B returns, it issues IDs lower than ones A gave out an hour ago. Flickr reported having a few hundred thousand more odd IDs than even ones. That is fine for their needs: IDs are unique and **roughly** increasing, which is enough for index locality and approximate time sorting. But you cannot rely on "a higher ID was created later", and you cannot use the IDs to detect gaps or count rows.

This setup ran in production from January 2006, in keeping with Flickr's principle of doing "the dumbest possible thing that will work". It is a good model for system design in general: a few lines of configuration on a technology the team already understood, instead of a new distributed system.

## Making it faster: allocate in blocks

A network round trip per ID costs about 0.5 to 1 ms and puts every insert on the ticket server's critical path. The common improvement is **block allocation** (sometimes called hi/lo or segment allocation): a client asks for a range instead of one number, then hands out IDs locally until the range runs out.

With a sequence that steps by 1,000, one call reserves 1,000 IDs. The database load drops by a factor of 1,000, and the client can fetch the next block in the background when the current one is, say, 80% used, so no request ever waits on the network.

~~~dg-seq Block allocation: one round trip reserves 1,000 IDs, which the app server then hands out locally.
App server > Ticket DB : reserve next block
Ticket DB -> App server : block 52,000 to 52,999
App server > App server : hand out 52,000, 52,001, ...
App server > Ticket DB : at 80% used, prefetch next block
Ticket DB -> App server : block 61,000 to 61,999
~~~

The trade-offs:

- If an app server crashes, the unused part of its block is **lost**. With 64-bit IDs that waste does not matter, but it leaves gaps.
- IDs are now even less ordered: server 1 is issuing from 52,000 while server 2 issues from 61,000.

## Failure scenarios

- **One ticket server dies.** Clients use the other. Capacity halves, but nothing breaks.
- **A ticket server's disk is restored from an old backup.** Its counter jumps backwards, and it starts re-issuing IDs that are already in use. This is the dangerous case. The safe recovery is to set the counter well above the highest ID it could ever have issued (for example, the current maximum ID in the data plus a large margin) before letting it serve.
- **You need a third server.** The increment is baked into the scheme. Changing from step 2 to step 3 means picking a new start point above every ID already issued and reconfiguring all servers at once. This is the scheme's main inflexibility, and why many designs fix a generous step (say 16) from day one and run only as many servers as they need.

## Where it fits among ID schemes

~~~dg-matrix Common ways to generate unique IDs across shards, and what each one gives you.
Scheme | No central service | 64-bit | Time-ordered | Survives clock problems
Single auto-increment | no | yes | yes, strictly | yes
Ticket servers | no | yes | partial, roughly | yes
Block allocation | no | yes | partial, coarse | yes
UUID v4 | yes | no, 128-bit | no | yes
Snowflake style | yes | yes | yes, roughly | no, needs care
~~~

Snowflake-style IDs (time + machine ID + sequence, lessons 04.10 and 04.11) remove the network call entirely but depend on clocks behaving. Ticket servers depend on no clocks at all, which is a real advantage. The same idea also works outside MySQL: in Postgres, `CREATE SEQUENCE photo_ids INCREMENT BY 2 START WITH 1` on one server and `START WITH 2` on another gives the same odd/even split, and Redis's atomic `INCRBY` can serve as a ticket counter if it is persisted carefully.

<div class="ex">
<div class="lab">Worked example</div>

**ID ranges with three servers**

With 3 ticket servers, set `auto_increment_increment = 3` and offsets 1, 2 and 3. Server 1 issues 1, 4, 7, 10; server 2 issues 2, 5, 8, 11; server 3 issues 3, 6, 9, 12. For any ID, `id mod 3` tells you which server issued it, which is handy for debugging.

Adding a fourth server later requires a new increment of 4. Suppose every ID issued so far is below 9.2 billion. You configure all four servers with increment 4 and offsets 1 to 4, move each counter up to 10 billion (MySQL then issues the next value above the counter that matches its offset), and switch them over together. The old servers must stop before the new scheme starts, or the residues could overlap. Planning a step of 8 or 16 from the start avoids this migration.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**is one ticket server pair enough?**

Peak traffic is 20,000 new objects per second across the whole site. Without batching, that is 20,000 `REPLACE` + `SELECT` pairs per second, split over two servers: 10,000 each. Each `REPLACE` is a delete and an insert on a one-row table, which a modern MySQL server can do at tens of thousands per second, but with durable commits it is limited by log fsyncs.

With block allocation of 1,000 IDs, the load drops to 20 requests per second in total. Each server sees 10 requests per second and is essentially idle. A 64-bit space is 1.8 × 10^19 values; at 20,000 per second you would run out in about 29 million years, so wasting unused blocks is irrelevant.

</div>

## Common mistakes

- **Assuming IDs are strictly ordered** across servers and using them as timestamps.
- **Restoring a ticket server from backup** without bumping its counter above every issued ID.
- **Choosing an increment of exactly the current server count**, which makes adding servers a migration.
- **Calling the ticket server on every insert** instead of allocating blocks.

## In an interview

If asked how to generate unique IDs across shards, the ticket server is a strong, simple answer to give first: name the odd/even offset trick, block allocation to remove the per-insert round trip, and the trade-off (roughly ordered, a small service to run). Then compare it with Snowflake-style IDs, which avoid the network call but depend on clocks.

<div class="take">
<div class="lab">Key takeaway</div>

a ticket server is a one-row MySQL table that hands out auto-increment IDs. Two servers with the same step and different offsets (odd and even) can never collide, need no coordination, and survive either one failing. Allocate IDs in blocks to remove the per-insert round trip, and accept that IDs are only roughly ordered.

</div>

---
id: "15.05"
course: "system-design-101"
module: "15"
slug: "a-collaborative-editor"
title: "A collaborative editor"
keyIdeas: ["Collaborative editing lets many users edit the same document simultaneously and see each other's changes live.", "**Operational transformation (OT)**: a central server orders operations and transforms concurrent ones against each other (e.g. shift an insert position). Google Docs uses OT.", "**CRDTs**: data structures where concurrent operations commute, so replicas converge without a central server. Each character gets a unique ID ordered between its neighbours.", "Two users typing at the same spot: OT transforms one op by the other; a CRDT uses unique IDs (with a tiebreak like site ID) so both inserts land in a consistent order.", "Architecture: WebSocket to a document server (one per document), operation log persisted, periodic snapshots, presence/cursor broadcasting. Figma uses a server-authoritative design with last-writer-wins per property."]
question: "Two users type at the same position at once. How do OT and CRDTs each resolve it?"
answer: "Both users apply their own insert locally, then exchange operations. With OT, a central server puts the two inserts in a single order; the later one is transformed against the earlier, and because the positions are equal a fixed tie-break (for example, already-committed operation or lower user ID first) decides that it shifts one place right. Each client transforms the incoming operation against its own pending one using the same rule, so both end with the same text. With a CRDT, each insert refers to the unique ID of the character it follows, not to a position; both inserts follow the same character, so a deterministic rule on their own new IDs (counter, then site ID) decides which comes first. Every replica reaches the same order whatever order the operations arrive in, without needing a server."
sources: [{"title": "Hello Interview: Google Docs breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/google-docs", "note": "OT vs CRDT in an interview."}, {"title": "Figma: How multiplayer technology works", "url": "https://www.figma.com/blog/how-figmas-multiplayer-technology-works/", "note": "A real server-authoritative design."}, {"title": "crdt.tech", "url": "https://crdt.tech/", "note": "CRDT resources hub."}]
tags: ["Design"]
---

## The problem

Two people edit the same document at the same time. A collaborative editor must feel instant, so each keystroke is applied to the local copy **immediately**, before any server has seen it, and then sent to everyone else. That is what makes the problem hard: by the time Alice's edit reaches Bob, Bob's document has already changed underneath it.

Consider the document `abcdef`. Alice types "x" at position 5 (between `e` and `f`, counting from 0). At the same instant Bob deletes the character at position 3 (`d`). Each applies their own change locally first:

- Alice now has `abcdexf`.
- Bob now has `abcef`.

Then they exchange operations. Alice applies "delete at 3" and gets `abcexf`. Bob applies "insert x at 5" literally, to `abcef`, and gets `abcefx`: the `x` lands one place too far right, because his delete shifted everything after position 3 left by one. The two copies now differ, and they will drift further apart with every edit. **Convergence**, every replica ending in the same state once all operations are applied, is the one property a collaborative editor cannot give up.

Two families of algorithms solve this: **operational transformation (OT)** and **conflict-free replicated data types (CRDTs)**. The surrounding system for a Google Docs-like editor needs: create documents, many users editing at once, changes visible to others within a few hundred milliseconds, other users' cursors and presence, and history.

## Operational transformation

OT keeps operations as simple position-based edits ("insert 'x' at 5", "delete at 3"), but **transforms** each incoming operation against the concurrent operations that were already applied, adjusting its position so it still means what its author intended.

The transform function takes two concurrent operations A and B (both made against the same document version) and produces A′, a version of A that can be applied after B. For insert and delete on plain text, the rules are short:

- An insert at position p, transformed against a delete at position q < p, moves to p − 1.
- An insert at p, transformed against an insert at q < p, moves to p + 1.
- A delete at p, transformed against an insert at q ≤ p, moves to p + 1.
- Two deletes of the same character: the second becomes a no-op.
- Two inserts at the **same** position: a tie-break rule decides which goes first (for example, lower site or user ID first), and both sides must apply the same rule.

In the example, Bob receives "insert x at 5" and transforms it against his own "delete at 3". Since 3 < 5, the insert becomes "insert x at 4", and Bob gets `abcexf`, the same as Alice.

### Why OT uses a central server

Transforming two operations is easy. Transforming arbitrary interleavings among many peers, each with their own pending operations, is notoriously hard: some published peer-to-peer OT algorithms were later shown to diverge in rare cases. The practical answer, used by Google Docs, is a **central server** that decides a single order of operations. Each operation carries the document **revision** it was based on. The server transforms an incoming operation against everything committed since that revision, assigns it the next revision, and broadcasts it.

~~~dg-seq OT with a central server: Bob's delete is committed first, so Alice's concurrent insert is transformed from position 5 to 4.
Alice > Server : insert "x" at 5 (based on rev 10)
Bob > Server : delete at 3 (based on rev 10)
Server > Server : commit Bob's delete as rev 11
Server > Server : transform Alice's insert past rev 11: 5 becomes 4, commit rev 12
Server -> Alice : rev 11: delete at 3
Server -> Bob : ack rev 11, then rev 12: insert "x" at 4
Alice > Alice : already applied own insert; transform delete, result abcexf
Bob > Bob : apply insert at 4, result abcexf
~~~

Each client also has to transform: Alice's own insert is already applied locally, so the incoming delete must be transformed against it before she applies it. To keep this manageable, a client sends **one operation at a time** and buffers further edits until the server acknowledges the previous one.

~~~dg-state An OT client's synchronisation states: at most one operation is in flight, and later edits are buffered and composed.
[*] -> Synchronised
Synchronised -> Awaiting ack : local edit, send it
Awaiting ack -> Synchronised : server acks
Awaiting ack -> Awaiting with buffer : more local edits, compose into buffer
Awaiting with buffer -> *Awaiting ack : ack arrives, send buffer
~~~

Figma considered OT and chose not to use it: the transform logic is complex and hard to get right, with a combinatorial explosion of cases as you add operation types (move, resize, reparent, set property). OT suits long text documents with a small set of operations; a design tool has hundreds.

## CRDTs

A **conflict-free replicated data type** is a data structure where every replica can be changed independently, even offline, and replicas always **merge into the same state** when they exchange updates, in any order, with no conflict-resolution code in the application. CRDTs do not need a central server, so they also work peer to peer and **offline-first**.

The trick is to design operations that **commute**: applying A then B gives the same result as B then A. Simple examples: a grow-only counter where each replica increments its own slot and the value is the sum; a set where adds and removes are tagged with unique IDs.

For text, a sequence CRDT avoids positions entirely. Each character gets a **unique, permanent ID**, for example a pair (counter, site ID), where the site ID identifies the user or device. An insert says "put 'x' **after the character with ID (17, Alice)**". Deleting only marks a character as deleted (a **tombstone**), so its ID stays as a stable anchor for anything inserted next to it. Because operations refer to IDs, not shifting positions, they can be applied in any order and still give the same result.

### Same position, both approaches

The lesson's question is the hardest case: two users insert at the same place at the same moment.

<div class="ex">
<div class="lab">Worked example</div>

**two users type at the same position, resolved by OT**

The document is `cat` at revision 7. Alice types "s" at position 3 (end). Bob types "!" at position 3 at the same moment. Both are based on rev 7.

Locally Alice sees `cats`; Bob sees `cat!`.

The server receives Alice's first and commits it as rev 8. Bob's insert also targets position 3; equal positions need the tie-break. The rule says the operation already committed (or the lower user ID) goes first, so Bob's insert shifts to position 4 and is committed as rev 9.

Alice receives rev 9: "insert ! at 4", giving `cats!`. Bob receives rev 8: "insert s at 3"; he transforms it against his own pending insert at 3. With the same tie-break, Alice's "s" goes before his "!", so it stays at 3 and his own character moves to 4: `cats!`. Both converge.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the same edit, resolved by a CRDT**

The characters of `cat` have IDs c = (1, A), a = (2, A), t = (3, A). Alice inserts "s" after (3, A) with the new ID (4, Alice). Bob inserts "!" after (3, A) with the new ID (4, Bob).

Both inserts reference the same anchor. A deterministic rule orders siblings, for example: higher counter first, then higher site ID first. The counters tie at 4; comparing site IDs, "Bob" > "Alice", so "!" comes immediately after `t` and "s" after it.

Every replica, whether it receives Alice's insert first or Bob's, applies the same rule to the same IDs and gets `cat!s`. No server was needed. The result may be `cat!s` rather than `cats!`, but it is the same everywhere, which is what matters; the users see it within a second and fix it if they care.

</div>

~~~dg-tree A sequence CRDT stored as a tree of character IDs: concurrent inserts after the same anchor become siblings ordered by a fixed rule.
start
  c (1, A)
    a (2, A)
      t (3, A)
        *! (4, Bob) | first
        s (4, Alice) | second
~~~

### The cost of CRDTs

Every character carries an ID (often 8 to 16 bytes before compression) and deleted characters remain as tombstones, so memory and document size grow. A 100,000-character document with heavy editing history might hold several times that many entries. Collapsing old tombstones needs every replica to agree that nobody will reference them again, which is hard without a coordinator. Libraries such as Yjs and Automerge use compact run-length encodings (consecutive characters typed by one user share one ID range), bringing overhead down to a small multiple of the plain text size.

There is also an **interleaving anomaly**: when two users concurrently type whole words at the same spot, some CRDT algorithms can interleave their characters ("hweolrllod" instead of "hello world"). Modern algorithms are designed to avoid this by keeping each user's run together.

~~~dg-compare OT versus CRDTs for collaborative text: OT moves complexity into transforms and a server; CRDTs move it into metadata.
[OT]
local edit -> server orders -(transform)-> broadcast
+ small operations, compact documents
+ proven at Google Docs scale
- needs a central server in practice
- transform logic is subtle per op type
[*CRDT]
local edit -(ID-based op)-> any peer -> merge in any order
+ works offline and peer to peer
+ convergence guaranteed by construction
- per-character IDs and tombstones
- document grows with history
~~~

| | OT | CRDT |
| --- | --- | --- |
| What travels | position-based edits plus a base revision | edits referencing unique IDs |
| Needs a central server | in practice yes, to order operations | no; peer to peer and offline work |
| Main cost | complex transform logic | metadata per character, tombstones |
| Used by | Google Docs | Figma (CRDT-inspired), Yjs- and Automerge-based editors |

## What Figma actually does

Figma took a useful middle ground: CRDT ideas, but with a **server as the authority**, which removes most CRDT overhead.

- **One server process per document.** Everyone editing that file connects to it over WebSockets. On open, a client downloads the whole file; afterwards, changes flow both ways.
- **Data model**: a tree of objects, each with an ID and properties: `Map<ObjectID, Map<Property, Value>>`.
- **Last writer wins per property.** The server keeps the latest value received for each property of each object. Edits to different properties, or different objects, never conflict. Two edits to the same property: whichever reached the server last wins. The server's arrival order decides, so no clocks are needed.
- **Text is a single value** in this model, so concurrent text edits in one text box produce one user's version, never a merge. Figma accepts this for a design tool; a text editor could not.
- **No flicker**: clients apply their own changes immediately and **ignore server updates to properties they have unacknowledged changes for**, because their pending value is the best guess of the final one.
- **IDs include a client ID**, so a client (even offline) can create objects without asking the server.
- **Parent links as properties.** Moving an object sets its parent property; the server rejects moves that would create a cycle. Child order uses **fractional indexing**: the position is a fraction between its neighbours (between 0.5 and 0.75 → 0.625), so inserting never renumbers other items.
- **Offline**: on reconnect, download a fresh copy, reapply local edits on top, and continue.

## The surrounding system

- **WebSocket servers**, with all editors of one document routed to the same server process (consistent hashing on document ID, lesson 04.07), so one place orders the operations. A popular document with 200 editors is still one process; a server hosts thousands of quieter documents.
- An **operation log** per document, appended for every committed change, with periodic **snapshots** (say every 1,000 operations) so loading does not replay millions of operations. The log also gives version history and undo.
- **Cursors and presence** are separate, ephemeral messages (who is here, where their cursor is). They are broadcast, throttled to a few per second, and never stored.
- **Failover**: if a document's server dies, clients reconnect, are routed to a new server that loads the latest snapshot plus the log tail, and resend their unacknowledged operations, which carry IDs so the server can drop duplicates.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the document servers**

2 million documents are open at peak, with 3 editors on average, each producing about 2 operations per second while typing (keystrokes are batched every 100 ms or so). Assume a quarter are actively typing at any moment: 2M × 3 × 0.25 × 2 = 3 million operations per second.

Each operation is about 100 bytes in the log: 300 MB per second, about 26 TB a day before snapshots allow old log segments to be compacted.

If one server process handles 30,000 operations per second (transform, append, fan out to a few sockets), you need about 100 servers, plus headroom for failover.

</div>

## Common mistakes

- **Sending raw positions without transformation** and assuming edits rarely overlap.
- **Peer-to-peer OT** without a proven algorithm; subtle divergence shows up months later.
- **Last-writer-wins on whole text fields** in a text editor, silently losing a user's paragraph.
- **Storing presence updates in the operation log**, bloating history with cursor moves.
- **No snapshots**, so opening an old busy document replays its entire history.

## In an interview

Start with the divergence example in two lines, then explain OT (transform plus central ordering) and CRDTs (unique IDs, tombstones, commutative merge), and show how each resolves two inserts at the same spot. Choose OT with a server for a classic online editor or a CRDT if offline and peer-to-peer matter, and describe the session server, the operation log with snapshots, and separate presence.

<div class="take">
<div class="lab">Key takeaway</div>

concurrent edits shift each other's positions. OT fixes this by transforming operations against each other, with a central server ordering them (Google Docs) and a tie-break for equal positions. CRDTs give each element a unique ID so edits merge in any order, even offline, at the cost of metadata and tombstones. Figma takes CRDT ideas but uses a server as the authority with last-writer-wins per property.

</div>

---
id: "06.01"
course: "system-design-101"
module: "06"
slug: "distributed-systems-foundations"
title: "Distributed systems foundations"
keyIdeas: ["A distributed system is several computers cooperating over a network that can drop, delay, duplicate or reorder messages.", "Nodes can crash, pause (GC, VM migration) or run slow, and other nodes can't tell these apart from a network problem. Timeouts are guesses.", "Clocks on different machines disagree, so you can't rely on timestamps to order events.", "Partial failure is normal: some parts work while others don't. A single request can fail at any of many steps, and the caller may not know if the action happened.", "Core tools: replication, partitioning, consensus, idempotency, timeouts and retries."]
question: "Why can't a node tell the difference between a crashed peer and a slow network?"
answer: "A node learns about a peer only through messages. If no reply arrives, the peer may have crashed, it may be alive but paused or overloaded (a long garbage collection pause, say), the request or the reply may have been lost or delayed, or the network may be partitioned. All of these look identical from outside: silence. Because real networks have no guaranteed upper bound on message delay or processing time, any timeout you choose can expire on a healthy but slow peer, so failure detection can only ever be a suspicion, and decisions taken on a timeout must be safe if the peer is still alive."
sources: [{"title": "Kleppmann: Distributed Systems lecture notes", "url": "https://www.cl.cam.ac.uk/teaching/2122/ConcDisSys/dist-sys-notes.pdf", "note": "Free course notes. Read sections 1-2."}, {"title": "Kleppmann: Distributed Systems lecture videos", "url": "https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB", "note": "The same course on video."}, {"title": "AWS Builders' Library: Challenges with distributed systems", "url": "https://builder.aws.com/content/3F08f7GPFiZMCgXD8gny6OjxR0Z/challenges-with-distributed-systems", "note": "8 failure modes of a single request."}]
---

## Why distributed is different

On one machine, a function call is all or nothing. It returns, or it throws, or the whole program crashes and takes the caller with it. Either way you know what happened. A **distributed system** is a set of programs on separate machines that cooperate by sending messages over a network, and that certainty disappears. Across a network, the caller and the callee no longer share fate: one can crash while the other carries on, and the network between them can lose, delay, duplicate or reorder messages on its own.

A single request and reply between a client and a server takes eight separate steps:

1. The client puts the request on the network.
2. The network delivers it to the server.
3. The server validates it.
4. The server updates its state (writes to its database, say).
5. The server puts the reply on the network.
6. The network delivers the reply.
7. The client validates the reply.
8. The client updates its own state.

Any one of these steps can fail while the others succeed. If step 2 fails, nothing happened. If step 6 fails, the server did the work but the client never hears about it. The nastiest outcome is **UNKNOWN**: the client waited, gave up, and cannot tell whether the server did the work or not. That is why idempotency (lesson 02.09) matters so much: if you cannot know whether an operation ran, you need to be able to run it again safely.

~~~dg-seq The eight steps of one remote call. Any one can fail while the others succeed, and failures after step 4 leave the client unsure.
Client > Network : 1. put request on network
Network > Server : 2. deliver request
Server > Server : 3. validate, 4. update state
Server > Network : 5. put reply on network
Network -> Client : 6. deliver reply
Client > Client : 7. validate, 8. update state
~~~

### Why testing gets so much harder

Each remote call can end in several distinct outcomes: failed to send, a retryable error, a fatal error, success, or unknown. Code that makes three remote calls in a row has dozens of combinations of outcomes to handle. A function that needed a handful of test cases on one machine can need dozens or hundreds once its steps are remote.

Distributed bugs also hide. They need a rare combination (a timeout on this call, while that node is restarting, during a deploy) and can stay latent for months. When they do trigger, they tend to **spread** along the network rather than stay local. One real example: an amazon.com catalogue server's disk filled up, so it started returning empty responses very quickly. The load balancer, which judged health by response time, saw a server answering fast and sent it more traffic. A single broken machine drew in a large share of requests and took down a big part of the site. A failure that would have been contained on one machine became a system-wide outage because of how the parts interacted.

## The models we reason with

To reason about anything in this module, you first state your assumptions. A **system model** is an explicit list of what can go wrong with the network, the nodes and timing. Algorithms are proven correct under a particular model, and they can break when reality violates it.

### Network models

- **Reliable links**: every message sent is eventually delivered, though maybe reordered. Rarely true on its own.
- **Fair-loss links**: messages can be lost, duplicated or reordered, but if you keep retrying, one eventually gets through. This is the usual assumption, and retries plus de-duplication turn it into a reliable link (TCP does exactly this).
- **Arbitrary links**: an attacker can drop, change or inject messages. Encryption and authentication (TLS) turn this back into fair-loss.

### Node models

- **Crash-stop**: a node that crashes never comes back.
- **Crash-recovery**: a node can crash and restart. It loses its memory but keeps whatever it wrote to disk. This is the realistic model for servers.
- **Byzantine**: a node can do anything, including lying, sending conflicting messages or acting maliciously.

### Timing models

- **Synchronous**: there are known upper bounds on message delay, processing time and clock drift. If a reply has not arrived within the bound, the node is definitely dead.
- **Partially synchronous**: the system behaves synchronously most of the time, but sometimes the bounds are exceeded, for periods of unknown length.
- **Asynchronous**: no timing bounds at all. A message can take any amount of time. No clocks you can trust.

~~~dg-matrix The common system models. Real datacenter systems are best described by the last row.
Model | Messages lost? | Nodes restart? | Nodes lie? | Delay bounded?
Synchronous, crash-stop | no | no | no | yes, always
Asynchronous, crash-stop | partial, delayed | no | no | no
Byzantine | yes | yes | yes | maybe
Partially sync, crash-recovery | yes | yes | no | partial, usually
~~~

Real systems are best modelled as **partially synchronous with crash-recovery nodes and fair-loss links**. Networks are usually fast, but a switch can buffer packets for seconds, a TCP retransmission can take 200 ms or more, and a virtual machine can be paused by its host. An algorithm that silently assumes synchrony ("a reply always comes within 100 ms") can break when a garbage collection pause or network hiccup violates the assumption, which is exactly when you need it to work.

## Classic impossibilities

Two thought experiments show the limits of what any protocol can achieve.

### The two generals

Two armies camp on either side of a valley. They win only if they attack at the same time. They can only communicate by messengers who cross the valley and might be captured. General A sends "attack at dawn". But A does not know the message arrived, so B sends back an acknowledgement. Now B does not know the acknowledgement arrived, so A acknowledges the acknowledgement. And so on. **No finite number of messages makes both generals certain** that they agree, because the last message sent can always be lost, and its sender cannot know.

The practical lesson: you cannot get certainty of agreement over an unreliable channel. So design for outcomes you can **reconcile** afterwards. An online shop that is unsure whether a charge went through can check later and refund a duplicate. Do not design something that requires both sides to be certain at the same moment.

### The Byzantine generals

Now some generals are traitors who send different messages to different peers. Agreement is possible only if **fewer than one third** of the participants are faulty: you need at least **3f + 1** nodes to tolerate f liars. Most datacenter systems assume crash faults, not Byzantine ones, because the machines are run by one organisation. Byzantine fault tolerance matters for blockchains and some aerospace systems, where participants cannot be trusted or hardware may corrupt data.

## Crashed or slow? The central problem

To act on a failure, a node first has to **detect** it. A **failure detector** usually works with timeouts: send a message (or expect a regular heartbeat), and if nothing arrives within T seconds, **suspect** the node.

But silence has many causes, and from the outside they all look identical:

- The node **crashed**.
- The node is alive but **slow**: a 5 second garbage collection pause, CPU starvation, a disk stall, overload.
- The **request was lost** or delayed in the network.
- The **reply was lost** or delayed in the network.
- The network is **partitioned**: the node is fine and talking to others, just not to you.

~~~dg-compare Four very different situations produce the same observation: no reply within the timeout.
[Crashed]
Request -> Node dead -> No reply
- Never answers
[Slow]
Request -> GC pause 5 s -> Reply arrives late
- Answers after you gave up
[Message lost]
Request -> Packet dropped -> No reply
- Node never saw it, or did the work
[*From outside]
Request -> ? -> Silence
- All four look identical
~~~

A node only learns about a peer through the messages it receives. It has no way to look inside the other machine. In an asynchronous network, there is no upper bound on how long a message can take, so whatever timeout you pick, a healthy but slow peer can exceed it. This is not a weakness of a particular implementation. It is a fundamental property, and it is behind a famous result (often called **FLP**, after its authors' initials): in a purely asynchronous system where even one node may crash, no deterministic algorithm can guarantee that nodes always reach agreement. Real consensus systems (lesson 06.06) get around it by relying on timing assumptions for **progress** while keeping **safety** independent of timing.

### Perfect and eventually perfect detectors

A **perfect** failure detector never suspects a live node and eventually suspects every crashed one. It only exists in a synchronous system. In practice you get an **eventually perfect** detector: it may wrongly suspect a slow node for a while, but eventually it stops making mistakes. Practical detectors like the **phi accrual** detector (used by Cassandra and Akka) do not give a yes or no answer. They output a suspicion level based on how late the heartbeat is compared with the recent history of heartbeat intervals, and you choose the threshold.

### The timeout trade-off

Timeouts trade **fast detection** against **false alarms**. A short timeout notices a crash quickly but misjudges slow nodes, triggering unnecessary failovers that themselves cause load and disruption. A long timeout avoids false alarms but leaves the system waiting on a dead node.

<div class="ex">
<div class="lab">Worked example</div>

**picking a heartbeat timeout**

A node sends heartbeats every 100 ms. Measured over a week, 99.9% of heartbeats arrive within 120 ms of the previous one, but garbage collection pauses occasionally delay one by up to 2 s, and once a month a network blip delays them by 4 s.

Timeout 300 ms: crashes detected in about 0.3 s, but every long GC pause triggers a false failover: several per day.

Timeout 5 s: no false alarms seen in the data, but a real crash leaves writes stalled for 5 s.

A common answer is around 1 to 2 s plus tuning garbage collection to shorten pauses, accepting a rare false alarm, and making sure that a false alarm is safe (see below). Many systems use something like this: ZooKeeper session timeouts are usually several seconds, and Raft election timeouts are commonly 150 to 300 ms on fast local networks and longer across regions.

</div>

### Designing around uncertainty

Because you cannot tell crashed from slow, every decision made after a timeout must be **safe even if the suspected node is still alive**:

- A node declared dead may still be running and acting on old beliefs. If it was the leader, you may now have two leaders (lesson 06.07). Fencing tokens (06.09) protect storage from the old one.
- A request that timed out may still have been executed. Retries must be idempotent.
- A lock holder that paused past its lease may wake up and keep writing (06.08).
- Timeouts should be set from measured latency distributions, not guessed.

~~~dg-timeline A "dead" node that was only paused. The rest of the cluster moved on, and the old node wakes up believing nothing changed.
0 s | Node A is leader, sending heartbeats
0.2 s | *A enters a 6 s garbage collection pause
2.2 s | Peers stop hearing A, timeout fires
2.5 s | Peers elect B as new leader
6.2 s | *A wakes up, still thinks it is leader, tries to write
6.3 s | Storage rejects A's stale term or token
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the unknown payment**

The order service calls the payment service with a 2 s timeout. At 2 s there is no answer. The possibilities: the request never arrived; the payment service crashed before charging; it charged the card and crashed before replying; or the reply is still in flight and will arrive at 2.3 s.

The order service cannot tell which. If it gives up, the customer may have been charged for an order that does not exist. If it retries naively, the customer may be charged twice.

The fix: the order service generates an **idempotency key** once per order and sends it with every attempt. It retries with the same key. The payment service stores the key with the result of the first charge, so a repeat returns the stored result instead of charging again. Whatever happened, the customer is charged at most once, and the order service eventually learns the outcome.

</div>

## Common mistakes

- **Treating a timeout as "the operation failed".** It means "I do not know".
- **Assuming a node declared dead is dead.** It may come back and act.
- **Guessing timeouts** instead of setting them from measured latency percentiles.
- **Relying on synchronised clocks or bounded delays** for correctness.
- **Testing only the happy path and clean crashes**, never pauses, partitions or lost replies.

## In an interview

Whenever your design has one component detect another's failure (failover, leader election, health checks), say the key sentence: "a timeout cannot distinguish a crashed node from a slow one, so the action after the timeout must be safe if the node is still alive". Then name the mechanism that makes it safe: idempotency keys, terms, fencing tokens.

<div class="take">
<div class="lab">Key takeaway</div>

in a distributed system any step of a call can fail independently, and a timeout cannot distinguish a dead node from a slow one or a lost message. Design so that every decision made on a timeout stays safe even when the suspected node is alive.

</div>

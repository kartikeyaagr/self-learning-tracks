---
id: "04.10"
course: "system-design-101"
module: "04"
slug: "snowflake-id-design"
title: "Snowflake ID design"
keyIdeas: ["Twitter's Snowflake ID is 64 bits: 1 unused sign bit, 41 bits of milliseconds since a custom epoch (~69 years), 10 bits machine ID (1,024 workers), 12 bits sequence (4,096 IDs per ms per worker).", "Each worker generates IDs locally with no coordination, so it's fast and highly available.", "IDs sort roughly by time, which helps indexing and lets you extract the creation time from the ID.", "Machine IDs must be unique, often assigned via ZooKeeper or config.", "If the sequence overflows within a millisecond, the worker waits for the next millisecond."]
question: "Break down the 64 bits of a Snowflake ID. How many IDs per ms per machine?"
answer: "1 sign bit, always 0 so the ID is positive; 41 bits of milliseconds since a custom epoch (2^41 ms is about 69 years); 10 bits of machine ID (1,024 generators, often split into 5 bits of data centre and 5 bits of worker); and 12 bits of sequence that counts IDs within the same millisecond on that machine. So each machine can generate 2^12 = 4,096 IDs per millisecond, about 4 million per second; if it needs more within one millisecond it waits for the next millisecond and resets the sequence. The ID is built as (timestamp minus epoch) shifted left 22, OR machine shifted left 12, OR sequence."
sources: [{"title": "Twitter: Announcing Snowflake", "url": "https://blog.x.com/engineering/en_us/a/2010/announcing-snowflake", "note": "The original post."}, {"title": "twitter-archive/snowflake", "url": "https://github.com/twitter-archive/snowflake/tree/snowflake-2010", "note": "Source and bit layout."}]
tags: ["Code"]
---

## Twitter's design

**Snowflake** is the ID generator Twitter built in 2010 when it moved tweets off a single MySQL database (lesson 04.08). Each ID is made of three parts: a **timestamp**, a **worker number** and a **sequence number**. Worker numbers are assigned once, when a generator starts, through ZooKeeper (a coordination service) or a configuration file. After that, generating an ID needs **no coordination at all**: it is a few lines of arithmetic on local state. That is what gives Snowflake its throughput and availability.

Twitter's requirements were specific:

- At least **10,000 IDs per second per process**, with a response time of about 2 ms or better including the network hop to the generator service.
- **No coordination** between generating machines.
- IDs **k-sorted**: roughly time-ordered, with any disorder bounded (Twitter promised within 1 second and aimed for tens of milliseconds).
- **Sortable without loading the objects**, so clients and services can order tweets by ID alone.
- **Under 64 bits**, so the IDs fit in the signed 64-bit integers used everywhere in Twitter's code and in many client languages.

## The 64 bits

| Bits | Field | Meaning |
| --- | --- | --- |
| 1 | Sign | Always 0, so the ID is a positive signed integer |
| 41 | Timestamp | Milliseconds since a custom epoch |
| 10 | Machine | Configured worker ID; later designs often split it into 5 bits of data centre and 5 bits of worker |
| 12 | Sequence | Counter within the same millisecond on this machine |

That is 63 bits of content. The top bit stays 0 so the value is a positive number in languages that only have signed 64-bit integers (Java's `long`, for example).

~~~dg-tree The bits of a Snowflake ID, from most significant to least. Time comes first so that IDs sort by creation time.
Snowflake ID, 64 bits
  Sign, 1 bit | always 0
  *Timestamp, 41 bits | ms since custom epoch
  Machine, 10 bits | up to 1,024 generators
    Data centre, 5 bits | optional split, up to 32
    Worker, 5 bits | up to 32 per data centre
  Sequence, 12 bits | 0 to 4,095 within one ms
~~~

### What those sizes give you

- **41 bits of milliseconds**: 2^41 ms is about 2.2 × 10^12 ms, or **69.7 years**. Counting from 1970 would waste 40 of those years, so Snowflake uses a **custom epoch**, a start date close to the system's launch (Twitter's is in November 2010). Twitter's IDs will run out around 2080.
- **10 machine bits**: 2^10 = **1,024** generators running at once.
- **12 sequence bits**: 2^12 = **4,096 IDs per millisecond per machine**, which is about 4.1 million per second per machine.

Across all 1,024 machines, the theoretical ceiling is about 4.2 billion IDs per second, far beyond what Twitter, or almost anyone, needs. The real limit is how fast one generator process can run its loop, which is millions per second.

If a single machine needs a 4,097th ID within the same millisecond, it **waits** (spins) until the clock reaches the next millisecond, then resets the sequence to 0. At 4 million IDs per second per machine, that is rare.

## Generating an ID

The generator keeps two pieces of state: `last_ts`, the timestamp of the last ID it issued, and `seq`, the sequence used within that millisecond.

```
def next_id():
    now = current_millis()
    if now < last_ts:               # clock went backwards (lesson 04.11)
        now = wait_until(last_ts)
    if now == last_ts:
        seq = (seq + 1) & 0xFFF     # 12-bit wrap
        if seq == 0:                # 4,096 used this ms
            now = wait_until(last_ts + 1)
    else:
        seq = 0
    last_ts = now
    return ((now - EPOCH) << 22) | (machine_id << 12) | seq
```

The final line is the whole encoding: shift the time left by 22 bits (10 + 12) to put it above the machine and sequence fields, shift the machine ID left by 12 to put it above the sequence, and combine with bitwise OR. The function must be called under a lock (or from a single thread) so that two callers never read the same `seq`.

~~~dg-state The generator's decision on each call, based on the current time compared with the last timestamp it used.
[*] -> Read clock
Read clock -> Same ms : now equals last_ts
Read clock -> New ms : now after last_ts
Read clock -> Clock went back : now before last_ts
Same ms -> *Issue ID : seq + 1 fits in 12 bits
Same ms -> Wait for next ms : seq overflowed
Wait for next ms -> New ms : clock ticks
New ms -> *Issue ID : seq reset to 0
Clock went back -> Wait for next ms : small jump
~~~

### Assigning machine IDs

The whole scheme rests on **no two running generators ever having the same machine ID**. If they did, both would produce identical IDs whenever their clocks and sequences coincided, which at high rates is constantly. Common ways to assign them:

- **ZooKeeper or etcd at startup**: each generator claims a free number by creating a lock node, holds it while running, and releases it on exit. If a process dies, its claim expires after a session timeout before the number can be reused.
- **Configuration**: a fixed number per host, managed by deployment tooling. Simple, but risky when containers are rescheduled.
- **Derived from the environment**: for example the pod's ordinal in a Kubernetes StatefulSet, or the low bits of a private IP address (which risks collisions if addresses are reused).

~~~dg-seq Machine IDs are coordinated once at startup; after that, every ID is generated locally.
Generator > ZooKeeper : claim a free worker ID
ZooKeeper -> Generator : you are worker 37
Generator > Generator : issue IDs locally, millions per second
App > Generator : next ID
Generator -> App : 562,036,736,000,151,557
Generator > ZooKeeper : heartbeat keeps claim alive
~~~

## Properties

- **Roughly sorted ("k-sorted")**: within one machine, IDs are strictly increasing. Across machines, IDs from the same millisecond interleave arbitrarily, and small clock differences between machines mean ordering is only approximate. Twitter aimed for a disorder bound well under one second.
- **Time is recoverable**: shift the ID right by 22 bits and add the epoch to get the creation time to the millisecond. Useful for debugging ("when was this created?") and for time-range queries by ID ("all tweets after ID X is all tweets after time T").
- **No central bottleneck**, but machine IDs must be unique and clocks must not go backwards (lesson 04.11).
- **Compact**: 8 bytes, the same as a `BIGINT` auto-increment key, so indexes stay small and inserts land at the end of the B-tree (lesson 04.09).
- **Guessable**: anyone can read the creation time from an ID, and IDs are dense enough to estimate volumes. That is acceptable for tweets, and a reason to use an opaque public ID for some other data.

## Tuning the layout

The split of bits is a design choice, not a law. You trade **lifetime** (timestamp bits), **number of generators** (machine bits) and **per-generator throughput** (sequence bits) against each other within 63 bits.

~~~dg-matrix Some well-known variants of the Snowflake layout and what each trades.
Scheme | Time bits and unit | Machine bits | Sequence bits | Notes
Twitter Snowflake | 41, 1 ms | 10 | 12 | 69 years, 4,096 per ms per machine
Discord | 42, 1 ms | 10, worker + process | 12 | epoch 2015, same idea
Instagram | 41, 1 ms | 13, logical shard | 10 | ID reveals its shard
Sonyflake | 39, 10 ms | 16 | 8 | 174 years, 65,536 machines, 25,600 per s each
~~~

- Fewer machine bits and more sequence bits suit a **few very busy** generators.
- More machine bits suit **many small** generators, such as one per container in a large fleet.
- Coarser time units (Sonyflake's 10 ms) buy a longer lifetime but reduce precision and per-machine burst rate.
- Instagram replaced the machine field with a 13-bit **logical shard ID**, so every ID says which shard holds its row (lesson 04.08).

<div class="ex">
<div class="lab">Worked example</div>

**decoding an ID**

ID = 562,036,736,000,151,557.

Timestamp: shift right by 22 bits. 562,036,736,000,151,557 ÷ 4,194,304 = 134,000,000,000 (rounded down). So the ID was created 134,000,000,000 ms after the custom epoch, which is about 4.25 years after it.

Machine: shift right by 12 and keep the low 10 bits. 562,036,736,000,151,557 − (134,000,000,000 × 4,194,304) = 151,557. Then 151,557 ÷ 4,096 = 37 remainder 5. So the machine ID is 37.

Sequence: the low 12 bits, the remainder: 5. This was machine 37's sixth ID in that millisecond (the sequence starts at 0).

</div>

<div class="ex">
<div class="lab">Worked example</div>

**will the layout last, and is it fast enough?**

A system plans 300 generator instances (well under 1,024) and peak traffic of 500,000 new IDs per second in total. Spread evenly, each instance needs about 1,700 per second, or under 2 per millisecond against a capacity of 4,096 per millisecond. Even a single instance taking all the traffic would only need 500 per millisecond.

With a custom epoch of 1 January 2026, the 41-bit timestamp runs out in late 2095. If the team expects the system to need more than 69 years, or more than 1,024 generators (say 5,000 containers), they could take 2 bits from the sequence (leaving 1,024 per ms) and give them to the machine field (4,096 generators): a layout decided once, at design time.

</div>

## Common mistakes

- **Duplicate machine IDs**, the fastest way to produce duplicate IDs.
- **Using the 1970 Unix epoch**, wasting 56 years of a 69-year range.
- **No lock around the sequence** in a multi-threaded generator.
- **Treating IDs as strictly ordered across machines.**
- **Storing IDs as JavaScript numbers.** JavaScript's numbers are exact only up to 2^53, so 64-bit IDs lose precision; APIs send them as strings (Twitter added an `id_str` field for exactly this reason).

## In an interview

Draw the 64 bits, give the field sizes and say what each buys: 41 bits of milliseconds is about 69 years from a custom epoch, 10 bits is 1,024 generators, 12 bits is 4,096 IDs per millisecond per generator. Mention that machine IDs are assigned once at startup and that the generator waits if the sequence overflows or the clock goes back.

<div class="take">
<div class="lab">Key takeaway</div>

Snowflake packs a sign bit, 41 bits of milliseconds since a custom epoch, 10 bits of machine ID and 12 bits of per-millisecond sequence into a 64-bit integer: about 69 years of range, 1,024 generators, 4,096 IDs per millisecond per machine, and no coordination per ID once machine IDs are assigned.

</div>

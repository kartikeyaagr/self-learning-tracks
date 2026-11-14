---
id: "04.11"
course: "system-design-101"
module: "04"
slug: "clock-skew-and-id-ordering"
title: "Clock skew and ID ordering"
keyIdeas: ["Machine clocks drift and get adjusted by NTP, sometimes backward. Clocks on different machines can differ by milliseconds or more.", "Time-based IDs from different machines are only roughly ordered; you can't use them to decide which of two close events happened first.", "If a Snowflake generator sees the clock go backward, it must not reuse timestamps. Options: refuse to generate until time catches up, or keep using the last timestamp and increment the sequence.", "For true ordering across machines, use logical clocks or a single sequencer. Google Spanner uses TrueTime with bounded uncertainty and waits it out."]
question: "A machine's clock jumps backward 5 ms. What should a Snowflake generator do?"
answer: "It must not issue IDs with the earlier timestamps, because it may already have used those (timestamp, sequence) pairs, and reusing them would create duplicate IDs. For a small 5 ms jump it should either wait until the clock passes the last timestamp it used (Twitter's original approach, a delay of about 5 ms), or keep using the last timestamp and keep incrementing the sequence, waiting only if the 4,096 sequence values for that millisecond run out. It should record a metric and log the event, and refuse to generate (failing health checks and alerting) if a backward jump is large, such as more than 100 ms to a second. It should also persist its last timestamp so a restart after a backward step cannot reuse old timestamps."
sources: [{"title": "Kleppmann: Distributed Systems lecture notes", "url": "https://www.cl.cam.ac.uk/teaching/2122/ConcDisSys/dist-sys-notes.pdf", "note": "Section 3: physical clocks and skew."}]
---

## Computer clocks are not reliable

A Snowflake ID (lesson 04.10) puts the current time in its top 41 bits and relies on one promise: **a generator never issues the same timestamp and sequence twice**. That promise depends on the clock never running backwards. Unfortunately, computer clocks are far less reliable than most people assume.

Every computer keeps time with a **quartz crystal** oscillator, which ticks at a nominal frequency. Real crystals are slightly off and their error changes with temperature. The error is called **drift** and is measured in **parts per million (ppm)**. One ppm is one microsecond per second, which is about **86 ms per day**. Typical server clocks are within about 50 ppm, so an unsynchronised clock can wander by **4 seconds a day**. Two machines left alone for a week can disagree by most of a minute.

### NTP and how it corrects the clock

**NTP** (Network Time Protocol) keeps clocks close to a reference. A client sends a request to a time server, records four timestamps (when it sent, when the server received, when the server replied, when it received the reply), and estimates both the network round trip and its own **offset** from the server's time. The estimate assumes the delay is the same in both directions. When it is not (a congested uplink, a different route back), the error goes straight into the offset. In good conditions in one data centre, NTP keeps machines within a millisecond or a few; over the internet, tens of milliseconds; under bad conditions, far worse. Cloud providers now offer more precise time services, some within microseconds, but no clock is ever exactly right.

How the NTP daemon applies a correction depends on how far off the clock is:

- Offset under about **128 ms**: **slew** the clock. It runs slightly fast or slow (by at most 500 ppm, so about half a millisecond per second) until it has caught up. Time keeps moving forwards, just at a slightly different rate.
- Offset between about 128 ms and 1,000 seconds: **step** the clock, jumping it straight to the right time. If the clock was ahead, **this moves it backwards**.
- Offset over 1,000 seconds: assume something is badly wrong and give up, leaving it to an operator.

Other things also step clocks: an administrator running `date -s`, a virtual machine being paused and resumed or live-migrated to another host, a container starting with a wrong clock and syncing a minute later.

~~~dg-bars How far a clock can be off in different situations, in milliseconds.
NTP in one data centre | 1 | about 1 ms
NTP over the internet | 30 | tens of ms
Typical step-back event | 50 | tens to hundreds of ms
*Unsynced for a day at 50 ppm | 4320 | about 4.3 s
~~~

### Leap seconds

To keep clocks aligned with the Earth's rotation, a **leap second** has occasionally been added, making some days 86,401 seconds long. Unix time cannot represent the extra second, so systems typically **repeat** a second, which means time goes backwards by one second. The leap second of 30 June 2012 triggered a Linux kernel bug that sent many Java and MySQL processes into 100% CPU loops and caused outages at several large websites. Many companies now **smear** the extra second, slowing their clocks very slightly over many hours so time never repeats. Leap seconds are due to be abolished by 2035, but backward steps from other causes remain.

## Two kinds of clock

Operating systems offer two different clocks, and confusing them causes real bugs.

- A **time-of-day clock** (wall clock), such as Java's `System.currentTimeMillis()`, Unix `gettimeofday()` or `CLOCK_REALTIME`, returns the date and time. It is comparable across machines (roughly), but it can **jump forwards or backwards** when NTP steps it.
- A **monotonic clock**, such as Java's `System.nanoTime()` or `CLOCK_MONOTONIC`, only ever moves forwards. NTP may slew its rate but never steps it. Its absolute value is meaningless (often the time since boot) and cannot be compared between machines.

~~~dg-compare The two clocks every operating system provides, and what each is safe for.
[Time-of-day clock]
currentTimeMillis, CLOCK_REALTIME
+ Comparable across machines, roughly
+ Gives a calendar date
- Can jump backwards on NTP step
- Unsafe for measuring durations
[Monotonic clock]
nanoTime, CLOCK_MONOTONIC
+ Never goes backwards
+ Right for timeouts and elapsed time
- Value means nothing on another machine
- Resets on reboot
~~~

Rule of thumb: **measure durations with the monotonic clock**, and use the wall clock only when you need an actual date or a value other machines will interpret. A timeout computed from the wall clock can fire instantly or never if the clock steps.

A Snowflake generator needs the **wall clock**, because its timestamps must be comparable across machines and must survive restarts. So it must handle backward jumps explicitly.

## What a Snowflake generator should do when the clock goes back

Suppose the last ID used timestamp 1,000 ms with sequence 17, and the clock now reads 995 ms. If the generator simply carried on, it would issue IDs with timestamp 995 and sequence 0, 1, 2.... Five milliseconds ago it may already have issued `(995, 0)`, `(995, 1)` and so on, so these would be **duplicate IDs**: two different tweets, or two payments, with the same primary key. Depending on the system, that means a failed insert, an overwritten row, or a cache returning the wrong object.

The options:

- **Refuse and wait.** Twitter's original implementation refused to generate IDs until the clock had passed the last timestamp it used, returning an error in the meantime. For a 5 ms jump, waiting 5 ms is trivial and callers barely notice.
- **Keep using the last timestamp.** Treat time as not having moved: keep issuing IDs with timestamp 1,000 and sequence 18, 19, 20.... When the sequence reaches 4,095, wait for the real clock to pass 1,000. This never blocks for small jumps at normal rates, IDs keep increasing, and it is safe. Its only cost is that a few IDs carry a timestamp a few milliseconds in the "future".
- **Large jumps: stop and alert.** If the clock went back by more than some threshold (100 ms, or a second), something is wrong with time sync on that machine. Waiting a minute would block all writes through this generator, so it is better to fail health checks, take the instance out of service and page someone.
- **Persist the last timestamp across restarts**, or at least refuse to start until the clock is past the time of the previous shutdown. Otherwise a generator could crash, restart after a backward step, and reuse timestamps from before the crash. An alternative is to take a fresh worker ID on every restart, so old IDs cannot collide with new ones.

~~~dg-seq A generator facing a 5 ms backwards step keeps issuing from its last timestamp instead of reusing old ones.
Caller > Generator : next ID
Generator > Clock : read time
Clock -> Generator : 1,000 ms
Generator -> Caller : ID (1,000, seq 17)
Clock > Clock : NTP steps back 5 ms
Caller > Generator : next ID
Generator > Clock : read time
Clock -> Generator : 995 ms, behind last
Generator -> Caller : ID (1,000, seq 18), metric logged
~~~

Prevention helps too. Configure NTP (or chrony) to **slew rather than step** whenever possible, start generators only after the clock is synchronised, and monitor clock offset on every machine.

~~~dg-matrix Strategies for a backwards clock, compared on safety and impact.
Strategy | No duplicates | Blocks callers | Handles large jumps
Ignore it | no | no | no
Refuse until caught up | yes | yes, for the jump length | no, blocks for long
Keep last timestamp | yes | only if sequence runs out | partial
Alert and stop above threshold | yes | yes, instance removed | yes
~~~

## Why IDs only give rough order

Even with perfect generators, two machines' clocks differ by some milliseconds. An event on machine A can get a smaller timestamp than an event on machine B that happened **after** it, even one **caused** by it: a user posts a tweet (machine A, clock 3 ms fast) and a reply arrives 2 ms later through machine B (accurate clock). The reply's ID is smaller than the tweet's.

So physical timestamps cannot reliably order events across machines. Distributed systems use the **happens-before** relation instead: A happens before B if they are on the same machine in that order, or A sent a message that B received, or through a chain of such steps. **Lamport clocks** capture it with counters: each process increments its counter for each event, attaches it to every message, and on receiving a message sets its counter to the larger of its own and the received value, plus one. If A caused B, A's counter is guaranteed to be smaller (Module 6 covers this and vector clocks). Google's Spanner takes a different route: its TrueTime API returns an interval with a guaranteed error bound, and the database waits out the uncertainty before committing.

Snowflake IDs are fine for "roughly by time". If you need **causal** order, you need logical clocks or a single sequencer, such as one partition of a log.

~~~dg-timeline A reply whose ID sorts before the tweet it replies to, because machine A's clock runs 3 ms fast.
real 12:00:00.000 | Tweet created on machine A, ID time .003
real 12:00:00.001 | *Reply created on machine B, ID time .001
real 12:00:00.002 | Sorting by ID shows the reply first
~~~

<div class="ex">
<div class="lab">Worked example</div>

**handling a 5 ms backward jump**

The last ID used timestamp 1,000,250 ms with sequence 17. NTP steps the clock back, and it now reads 1,000,245.

The generator keeps issuing IDs with timestamp 1,000,250 and sequence 18, 19, 20.... At the current rate of 300 IDs per millisecond, the 4,096 sequence values last about 13 ms, longer than the 5 ms the clock needs to catch up, so no caller ever waits.

When the real clock passes 1,000,250, the generator returns to normal. A metric records a 5 ms backward jump. Alerts fire if any jump exceeds 100 ms, and the instance fails its health check if one exceeds 1 second.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**the restart that would have caused duplicates**

A generator issues IDs up to timestamp T = 10:00:00.900, then crashes. The machine's clock was 2 seconds fast; on reboot NTP steps it back, and the generator restarts at 10:00:00.200 by the corrected clock.

Without protection, the next 700 ms of IDs reuse timestamps from 10:00:00.200 to 10:00:00.900, which this machine ID already used. At 2,000 IDs per second, about 1,400 IDs could duplicate earlier ones.

With protection, the generator writes a high-water mark to local disk once a second, always one second ahead of its clock, so the last value saved before the crash is 10:00:01.900 or earlier but never below any timestamp it used. On restart it reads that mark and refuses to issue IDs until the clock passes it: about 1.7 seconds of downtime for this instance, while the load balancer sends traffic elsewhere. No duplicates.

</div>

## Common mistakes

- **Assuming the wall clock only moves forwards.**
- **Measuring timeouts or durations with the wall clock.**
- **Ignoring restarts**, where the generator forgets its last timestamp.
- **Using timestamps to order events across machines**, such as "last write wins" with wall-clock times, which silently drops writes when clocks disagree.
- **No monitoring of clock offset**, so a drifting machine goes unnoticed until it causes duplicates or ordering bugs.

## In an interview

When you propose Snowflake-style IDs, raise clocks before you are asked: NTP can step the clock back; the generator must never reuse a timestamp, so it keeps using its last timestamp (or waits) for small jumps, alerts and stops for large ones, and persists its last timestamp across restarts. Add that IDs are only roughly ordered across machines, and causal order needs logical clocks.

<div class="take">
<div class="lab">Key takeaway</div>

wall clocks drift and can jump backwards (NTP steps, leap seconds, VM pauses, restarts). A Snowflake generator must never reuse a timestamp: for small jumps, keep issuing from the last timestamp or wait for the clock to catch up; for large jumps, stop and alert; and remember the last timestamp across restarts.

</div>

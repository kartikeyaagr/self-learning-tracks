---
id: "11.06"
course: "system-design-101"
module: "11"
slug: "a-newly-unread-indicator"
title: "A newly-unread indicator"
keyIdeas: ["An unread indicator shows whether a channel or conversation has new messages since you last read it.", "Store per user per channel a last-read message ID (or timestamp). Store per channel the latest message ID.", "\"Has unread\" = latest message ID > last-read ID, a cheap comparison if both are in a fast store (Redis or in-memory service).", "Mention counts need more state: increment a per-user counter when someone mentions them, reset on read.", "Discord's Read States service tracks exactly this for every user and channel, and moved from Go to Rust to avoid GC latency spikes at that scale."]
question: "Store last-read message ID per user per channel. How do you compute \"has unread\" cheaply?"
answer: "Keep the newest message ID on each channel and, per user per channel, the last read message ID, updated monotonically when the user catches up. Because message IDs increase over time, \"has unread\" is a single comparison, channel.last_message_id > last_read_id, with no messages scanned and no per-message read flags; dense per-channel sequence numbers turn an exact unread count into one subtraction too. Serve both values from RAM (a cache of read states partitioned by user and a cache of channel heads), batch-load them for all of a user's channels on connect, keep a separate mention counter, and persist read states lazily with coalesced writes, as Discord does with an LRU cache flushed to Cassandra 30 seconds after an update or on eviction."
sources: [{"title": "Discord: Why Discord is switching from Go to Rust", "url": "https://discord.com/blog/why-discord-is-switching-from-go-to-rust", "note": "About the Read States service that tracks unread state."}]
tags: ["Design"]
tools: ["Redis"]
---

## The feature

Chat apps bold a channel's name when it has messages you haven't read, show a badge with the number of times you were mentioned, and draw a "new messages" line where you stopped reading. It looks like a small UI detail. At scale it is one of the hottest paths in the whole system, because it is touched every time anyone sends a message, every time anyone reads one, and every time anyone connects.

The question proposes the standard model: store, per user per channel, the **ID of the last message the user has read**. Then:

`has unread = channel.last_message_id > read_state.last_read_message_id`

If message IDs increase over time (Snowflake-style IDs, lesson 04.10, or per-channel sequence numbers), this is a single integer comparison. No message is scanned, and no per-message "read" flag is stored.

~~~dg-flow Unread is one comparison: the channel's newest message ID against the user's last-read ID.
Channel last_message_id = 1,057 -(compare)-> *User last_read = 1,049 -> 1,057 > 1,049, so bold
~~~

## Why not a read flag per message?

The obvious alternative is a `message_reads(user_id, message_id)` table: mark each message read for each user. It fails on arithmetic. A message in a 10,000-member server channel would need 10,000 read rows. A busy community with 100 messages a minute would generate a million read rows a minute, for one channel. And "has unread" becomes "does any message in this channel lack a read row for me?", which is a scan.

The pointer model works because reading in a chat is **sequential**: when you open a channel and scroll to the bottom, you've read everything up to the newest message. One number captures it. (Email is different: people read messages out of order, which is why email systems do keep a per-message read flag, and why their mailboxes are per user rather than shared.)

~~~dg-compare Two ways to track what a user has read in a shared channel: a flag per message per user, or one pointer per user per channel.
[Read flag per message]
One row per user per message
- 10,000 rows per message in a big channel
- "Has unread" needs a scan
+ Handles out-of-order reading
[*Last-read pointer]
One row per user per channel
+ Has unread is one comparison
+ Tiny: about 30 bytes per row
+ One write when the user catches up
- Assumes sequential reading
~~~

## Why it's cheap, and what makes it expensive

The comparison is cheap. Getting the two numbers to the comparison, for every channel, for every user, every time, is where the cost lives.

- **Per channel**: store `last_message_id` on the channel, updated on every new message. Reading it is a key lookup, usually from a cache of channel heads.
- **Per user per channel**: a **read state** `{last_read_id, mention_count}`. When the user opens the channel or scrolls to the bottom, set `last_read_id` to the newest visible ID and reset `mention_count` to 0. Updates are monotonic: never move the pointer backwards, so a stale update from another device can't un-read messages (`SET last_read_id = max(last_read_id, :new)`).
- **On connect**: to draw the channel list, the client needs both numbers for each of the user's channels. With hundreds of channels, this must be one batch lookup served from RAM, not hundreds of database queries.
- **Mentions**: a mention increments the mentioned user's `mention_count` for that channel. That's a write fan-out, but only to the people mentioned. `@everyone` in a 100,000-member channel means 100,000 counter writes, which is why that feature is restricted by permissions and often rate limited.
- **Multiple devices**: when you read on your phone, your laptop's bold names should clear. The read-state update is published to all of the user's connected sessions (lesson 11.01).

### Exact unread counts

"Has unread" is cheaper than "12 new messages". An exact count needs the number of messages with ID > last_read_id, which with Snowflake IDs is a range count over the channel's message index. Two cheaper routes:

- **Dense per-channel sequence numbers.** If each channel numbers its messages 1, 2, 3, ..., then unread count = `channel.last_seq − last_read_seq`, again one subtraction. Deleted messages make it slightly high, which is usually acceptable. The price is a per-channel counter that every new message must increment (a hot row for a busy channel; lesson 10.01 applies).
- **Cap the display.** Many apps show "99+" or just a dot, so they only need to count up to 100, a bounded range scan.

~~~dg-matrix Cost of each unread feature, per channel shown in the sidebar.
Feature | Data needed | Work per check | Write cost
Has unread (bold) | last_message_id, last_read_id | one comparison | one pointer write on read
Mention badge | mention_count | read one counter | one increment per mention
Exact count, sequence numbers | last_seq, last_read_seq | one subtraction | sequence per message
Exact count, Snowflake IDs | message index | range count | none extra
Count capped at 99+ | message index | bounded scan, at most 100 | none extra
~~~

## Discord's Read States service

Discord runs this design at very large scale in a dedicated **Read States** service, which tracks which channels and messages users have read. Its characteristics:

- One read state **per user per channel**, holding the last-read message ID and counters (such as the number of @mentions) that are updated atomically and often reset to 0.
- **Billions** of read states in total.
- It is on the hot path: accessed **every time a user connects, every time a message is sent, and every time a message is read**.
- Each instance holds an **LRU cache** of read states (least recently used entries are evicted first), with millions of users and tens of millions of read states per cache.
- **Hundreds of thousands of cache updates per second** and **tens of thousands of database writes per second**.
- Persistence: **Cassandra**. A read state is written to the database when evicted from the cache, and a write is scheduled **30 seconds** after each update, batching many updates into one write.

~~~dg-flow Discord's Read States: updates hit an in-memory LRU cache; Cassandra receives at most one write per read state per 30 seconds, plus one on eviction.
Message sent or read -(update)-> *LRU cache of read states -(30 s after update, or on eviction)-> Cassandra
~~~

The 30-second delayed write is the same batching idea as lesson 10.01. A user actively reading a busy channel might update its read state every few seconds; mentions arrive in bursts. Coalescing all updates in a 30-second window into one Cassandra write cuts database writes by an order of magnitude. The cost: if an instance crashes, up to 30 seconds of read-state changes are lost, and a few channels look unread again. For this feature that is an acceptable trade.

Users are partitioned across instances (by user ID), so all of one user's read states live in one instance's cache and the connect-time batch lookup is a single call to a single instance.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the read-state cache**

Suppose 20 million users are online, in an average of 80 channels each: 1.6 billion potential read states, but most belong to channels the user rarely opens. Say 25% are hot: 400 million.

A read state is about 8 bytes (channel ID) + 8 (last read ID) + 4 (mention count) + flags and LRU pointers, roughly 64 bytes in memory: 400 million × 64 bytes ≈ 26 GB, spread over a few dozen instances at around 1 GB each, comfortably in memory.

Without the 30-second coalescing, 300,000 updates per second would mean 300,000 Cassandra writes per second; with it, tens of thousands.

</div>

## The Go-to-Rust lesson

Discord's first version of Read States was written in Go and suffered **latency and CPU spikes about every 2 minutes**. Go's garbage collector forces a collection at least every 2 minutes, and each collection had to scan the huge LRU cache, tens of millions of objects, to find what was still referenced. The cache was the whole point of the service, so it couldn't be made smaller.

Discord rewrote the service in Rust, which has no garbage collector: memory is freed deterministically as soon as a read state is evicted. The Rust version beat Go on latency, CPU and memory, with average response times in **microseconds**, and it used a `BTreeMap` in the LRU for memory efficiency. Tail latency became flat because nothing paused to scan the heap.

~~~dg-timeline The pattern Discord saw with the Go version: steady latency punctuated by a spike each time the forced garbage collection scanned the cache.
0:00 | Normal: sub-millisecond responses
2:00 | *Forced GC scans tens of millions of cached entries: latency and CPU spike
2:05 | Back to normal
4:00 | *Forced GC again: another spike
6:00 | *And again, every 2 minutes
After Rust rewrite | No collector pauses: flat microsecond latency
~~~

The general lesson: very large, long-lived in-memory caches and tracing garbage collectors are an awkward combination. Options other than a rewrite include moving the cache off-heap, splitting it into many smaller processes, or using an external store like Redis, each with its own cost.

## Pitfalls

- **Non-monotonic updates.** Two devices sending read updates out of order can move the pointer backwards. Always take the max.
- **IDs that aren't ordered.** Random UUIDs can't be compared. Use time-ordered IDs or per-channel sequences.
- **Per-channel queries on connect.** 300 channels × 2 queries is 600 round trips before the sidebar draws. Batch, and serve from RAM.
- **Persisting every update synchronously.** Read positions change constantly; coalesce writes.
- **@everyone without limits.** One message can create 100,000 counter writes.

<div class="ex">
<div class="lab">Worked example</div>

**opening the app**

A user in 300 channels connects. The gateway asks Read States for the user's 300 read states (one call to the instance that owns the user) and gets the channels' `last_message_id` values from the channel cache in one multi-get.

41 channels have last_message_id greater than last_read_id and are drawn in bold; 3 have mention_count > 0 and show a badge. The user opens one: one update sets its last_read_id to 1,057 and mention_count to 0, the update is pushed to the user's other devices, and Cassandra receives the write within 30 seconds.

</div>

## In an interview

Propose the pointer model, write the comparison, and then talk about the hot path: "read states are touched on every connect, send and read, so they live in an in-memory cache partitioned by user, loaded in one batch on connect, persisted lazily with coalesced writes". Mention monotonic updates, mention counters, and sequence numbers if exact counts are needed.

<div class="take">
<div class="lab">Key takeaway</div>

store one read state per user per channel (last read message ID plus a mention counter). With increasing message IDs, "has unread" is just channel.last_message_id > last_read_id, and with dense per-channel sequence numbers the unread count is one subtraction. At Discord's scale, read states sit on every hot path and live in an in-memory LRU cache with batched writes to Cassandra.

</div>

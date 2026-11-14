---
id: "16.06"
course: "system-design-101"
module: "16"
slug: "chat-and-messaging"
title: "Chat and messaging"
keyIdeas: ["Chat: clients hold WebSocket connections to gateway servers. A message goes to a chat service, which stores it and routes it to recipients' gateways via pub/sub.", "Message IDs must be unique and ordered within a conversation (per-conversation sequence or Snowflake IDs).", "Offline users: messages are stored; when the user reconnects on any device, the client asks for everything after its last seen message ID. Push notifications alert offline devices.", "Each device tracks its own sync position, so 2 devices both catch up correctly.", "Storage: a wide-column store partitioned by conversation and time bucket suits huge write volume (Discord moved Cassandra → ScyllaDB).", "Presence and typing indicators are ephemeral and don't need durable storage."]
question: "How does a message reach a user who is offline and later comes online on 2 devices?"
answer: "The sender's server first saves the message and creates an undelivered inbox entry for each of the recipient's registered devices, then acknowledges the sender. Because the recipient is offline, the live pub/sub publish reaches nobody, so a push notification goes out through APNs or FCM. When the first device comes online it connects to any chat server, fetches its pending inbox entries, receives the message and acknowledges it, which deletes only that device's entry. The second device does the same whenever it connects, because its own entry is still there. If an ack is lost the message is resent and the client deduplicates by ID. Alternatively, when history is kept forever, each device stores the last message ID it has seen and fetches everything newer on connect."
sources: [{"title": "ByteByteGo: Design a Chat System", "url": "https://bytebytego.com/courses/system-design-interview/design-a-chat-system", "note": "WebSockets, message IDs, presence."}, {"title": "Hello Interview: WhatsApp breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/whatsapp", "note": "Delivery to offline users."}, {"title": "Discord: How Discord stores trillions of messages", "url": "https://discord.com/blog/how-discord-stores-trillions-of-messages", "note": "Real storage evolution."}]
tags: ["Design"]
---

## What we are building

A chat system in the style of WhatsApp, Messenger or Slack: people send text messages to one person or to a group, see who is online, use several devices at once, and get a push notification when they are not looking. Messages must never be lost, must arrive quickly, and should appear in the same order for everyone in a conversation.

The defining feature, compared with most web systems, is that the server has to **push** data to clients the moment it arrives. That changes the connection model, makes some servers stateful, and raises the question this lesson centres on: how does a message reach someone who is offline and later comes back on two devices?

### Functional requirements

- One-to-one chats and group chats of up to about **100 members** (large channels are a different problem, touched on at the end).
- **Several devices per account** (phone, laptop, tablet), each receiving every message.
- Online **presence** (online, last seen).
- **Push notifications** to devices that are not connected.
- Text up to 100,000 characters; media through attachments.
- History: either **kept forever** (Messenger, Slack, Discord) or **deleted from servers once delivered** (WhatsApp's model). We will design for both.

### Non-functional requirements

- Delivery to online users in under **500 ms**.
- **Guaranteed delivery**: once the sender sees "sent", the message will reach every recipient device.
- Offline messages kept for up to **30 days** in the delete-after-delivery model.
- Consistent ordering within a conversation.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing a WhatsApp-like system**

200 million daily active users each send about 20 messages a day: 4 billion messages a day, about 46,000 per second on average and perhaps 150,000 per second at peak.

At about 200 bytes per text message including metadata, that is 800 GB a day of new message data. Kept forever, about 300 TB a year before replication.

Connections: if a third of users are connected at peak with 1.5 devices each, that is 100 million open connections. A tuned chat server holds 1 to 2 million idle WebSocket connections (each costs tens of kilobytes of kernel and application memory), so you need on the order of 100 chat servers, more with headroom for deploys and failures.

</div>

## Connections: why WebSockets

How does the server tell a client "you have a new message"? Lesson 11.02 compares the options:

- **Polling** every few seconds wastes requests (most responses are empty) and adds up to the poll interval in latency.
- **Long polling** holds each request open until there is data, which works but needs a new request after every message and is awkward when the next request lands on a different server.
- **WebSockets** give one persistent, two-way connection per device over TCP. The server writes a frame whenever a message arrives; the client writes frames to send.

Use WebSockets. That makes the chat servers **stateful**: a given device is connected to one specific server, and only that server can write to it. Everything else (login, profiles, contacts, media upload) stays as ordinary stateless HTTP services. At login, a **service discovery** component (ZooKeeper, etcd or a simple registry) hands the client the address of a healthy chat server near it with spare capacity.

A **session registry** records which server each device is on: `user 42, device phone -> chat-17`. Heartbeats keep entries fresh, and an entry expires if the server dies.

## Routing between servers

The sender is connected to server 3 and the recipient to server 17. How does server 3 get the message to server 17?

- **Look up and forward**: server 3 asks the session registry where the recipient's devices are and calls server 17 directly.
- **Pub/sub**: each chat server subscribes to a channel per connected user (for example in Redis Pub/Sub). Server 3 publishes to `user:bob`, and whichever server holds Bob's connections receives it.

Redis Pub/Sub is **at most once**: if no one is subscribed, or a subscriber is briefly disconnected, the message is simply dropped. That is acceptable only because durability comes from somewhere else: the message is **written to the database first**, and pub/sub is just the fast path that wakes up the right server. If the fast path fails, the device still finds the message when it next syncs.

~~~dg-flow Sending to an online recipient: persist first, then use pub/sub as the fast path.
[Send] Alice -(WebSocket)-> Chat server 3 -(write)-> *Message store
[Route] Chat server 3 -(publish user:bob)-> Pub/sub -(subscriber)-> Chat server 17
[Deliver] Chat server 17 -(WebSocket frame)-> Bob's phone -(ack)-> Chat server 17
~~~

## Delivering to an offline user on two devices

This is the core question. The key structure is the **inbox**: a per-device list of messages waiting to be delivered to that device. The server keeps a **Clients** table listing each user's registered devices (say up to 3 or 4 per user).

1. Alice sends "hi" to Bob. Her chat server stores the message in the **Message** table and creates an **Inbox** row for each of Bob's devices: one for his phone, one for his laptop. Both writes happen before anything else.
2. It acknowledges Alice: her app shows one tick, "sent".
3. It publishes to Bob's channel. Bob is offline, so nobody is subscribed. Instead it sends a push notification through APNs or FCM, which wakes the phone app or at least shows a banner.
4. Later, Bob's **phone** connects to server 17. It asks for its undelivered inbox entries, receives "hi", and **acknowledges** it. The server deletes the phone's inbox row and can tell Alice "delivered" (two ticks).
5. Bob's **laptop** connects hours later to server 9. Its own inbox row is still there, so it receives "hi" too, then acknowledges.
6. Once every device has acknowledged, the delete-after-delivery model removes the message from the server altogether. Anything still unacknowledged after 30 days expires.

~~~dg-seq An offline user's two devices each pull their own copy from their inbox.
Alice > Chat srv 3 : send "hi" to Bob
Chat srv 3 > DB : save message, inbox rows for phone + laptop
Chat srv 3 -> Alice : sent
Chat srv 3 > Push : Bob offline: push notification
Phone > Chat srv 17 : connect, fetch inbox
Chat srv 17 -> Phone : "hi"
Phone > Chat srv 17 : ack, delete phone inbox row
Laptop > Chat srv 9 : connect later, fetch inbox, ack
~~~

### Why acknowledgements matter

Notice the server deletes an inbox row only after the device **acknowledges**, not after the server writes to the socket. A write to a TCP socket can succeed locally while the phone is already in a tunnel; the bytes never arrive. If the server deleted on send, that message would be lost. With acknowledge-then-delete, the worst case is a duplicate: the phone received the message, but the ack was lost, so the server sends it again on the next connect. The client deduplicates by message ID. This is the same at-least-once plus idempotent receiver pattern as in lesson 16.01.

~~~dg-state One inbox entry for one device. It disappears only after that device acknowledges.
[*] -> Pending
Pending -> Sent to device : device connected
Sent to device -> Pending : no ack, retry
Sent to device -> *Acked : device acks
Pending -> Expired : 30 days pass
~~~

### The cursor alternative

When history is kept forever, per-device inbox rows are wasteful: you are storing the message anyway. Instead, each device remembers the highest message ID it has seen, called its **cursor** (`cur_max_message_id`). On connect, it sends its cursor, and the server returns every message for this user's conversations with a higher ID. Each device syncs independently: a phone that was online all day has a recent cursor; a laptop closed for a week has an old one and fetches a week of messages in pages.

~~~dg-compare Two ways a device catches up after being offline.
[Inbox per device]
Message -(row per device)-> Fetch, ack, delete
+ Server forgets delivered messages
+ Device needs no history
- One write per device per message
[Cursor per device]
Message stored forever -> Fetch newer than cursor
+ No per-device rows
+ New device can load history
- Server keeps everything
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the cost of inbox rows**

4 billion messages a day, with recipients averaging 1.5 devices, and groups adding fan-out (a message to a 20-person group creates up to 30 inbox rows).

If one-to-one chats are 80% of messages and groups average 10 members, total inbox writes are roughly 4B x (0.8 x 1.5 + 0.2 x 15) = 4B x 4.2, about 17 billion rows a day, about 200,000 per second. Each row is small (user, device, message ID: about 40 bytes) and short-lived, so a write-optimised store with TTLs (Cassandra, or a log-structured key-value store) handles it.

The cursor model avoids those 17 billion writes but must store all 4 billion messages forever and serve "everything after ID X" reads efficiently, which pushes you towards conversation-partitioned storage.

</div>

## Storing messages

Chat history is huge, append-heavy, and almost always read as "the most recent messages in this conversation". That fits a **wide-column or key-value store** (HBase for Messenger, Cassandra and later ScyllaDB for Discord) better than a single relational database.

- Key messages by `(channel_id, message_id)`, with `channel_id` as the partition key and `message_id` as the clustering key, so one conversation's messages are stored together, sorted.
- `message_id` must be **unique and sortable** within the channel. Timestamps alone collide (two messages in the same millisecond). A Snowflake-style ID (lesson 04.10), which embeds a timestamp plus a machine ID and a sequence number, or a per-channel sequence both work.
- **Discord** partitions messages by **channel plus a time bucket** (a window of days), with Snowflake IDs, so a busy channel does not grow into one enormous partition (lesson 18.03 covers why). Its move from 177 Cassandra nodes to 72 ScyllaDB nodes cut p99 read latency from 40 to 125 ms down to about 15 ms.

### Ordering

Strict global order across all conversations is not needed and would be very expensive. What users need is that **everyone in a conversation sees the same order**. The usual approach: the server assigns the message ID when it receives the message (not the client's clock, which may be wrong), and clients sort by that ID. Occasionally a message sent a fraction of a second later appears above one sent earlier, because it reached the server first. Everyone sees the same order, which is what matters. For strict per-conversation order, a per-channel sequence assigned by a single owner of each channel works, at the cost of making that owner a bottleneck for very busy channels.

## Groups

- **Small groups**: copy each message into every member's device inboxes (WeChat fans out like this for groups up to 500). Each client checks only its own inbox, so reads are simple.
- **Large groups or channels**: copying to every inbox is too expensive (a 100,000-member channel would create 100,000 rows per message). Store each message once in a per-channel log and let members read from it with cursors. A common threshold is to switch from per-user to per-channel pub/sub above about 25 members, so the sender's server publishes once to `channel:123` and every server with a member connected receives it once.

## Presence

On connect, a user is online. Clients send **heartbeats** (say every 5 seconds), and the presence service marks a user offline only if none arrives for about 30 seconds, so a train through a tunnel does not make them flicker offline and online. Changes are published to the user's contacts. In a 100,000-member group that would be 100,000 events per change, so large groups fetch presence on demand (only for members visible on screen) instead of subscribing to every change.

~~~dg-timeline Presence with heartbeats. A short network drop does not mark the user offline; a long one does.
0 s | Heartbeat received, online
5 s | Heartbeat received
10 s | *Phone enters tunnel, heartbeats stop
25 s | Still online (under 30 s grace)
40 s | No heartbeat for 30 s: marked offline
45 s | Reconnects, online again, inbox sync
~~~

## Media and push

- **Media** (photos, voice notes, video) goes to blob storage through presigned upload URLs (lesson 13.01). The message carries only a reference and perhaps a thumbnail. With end-to-end encryption, the media is encrypted on the device and the key travels inside the encrypted message.
- **Push notifications** go through APNs and FCM via a notification service (lesson 16.01). With end-to-end encryption, the push often carries only "you have a new message", and the app decrypts after fetching.

## Failure handling

- **A chat server dies**: its 1 to 2 million connections drop. Clients reconnect with exponential backoff and jitter (otherwise a million reconnects arrive in the same second), get a new server from discovery, and sync their inbox or cursor. No message is lost because nothing was only in the dead server's memory.
- **Pub/sub drops a message**: the device does not receive it live, but finds it on the next sync. Clients also sync periodically, or when they notice a gap in message IDs.
- **Duplicate delivery**: clients deduplicate by message ID.
- **Sender retries**: the client attaches a client-generated message ID so a resend after a timeout does not create two messages.

## Common mistakes

- Delivering before persisting, so a crash loses the message.
- Deleting an inbox row when the server sends, not when the device acknowledges.
- Treating pub/sub as durable.
- Using client timestamps for ordering.
- One inbox per user instead of per device, so the first device to connect "uses up" the message.
- Pushing presence changes to every member of huge groups.

## In an interview

Start with the connection model (WebSockets, stateful chat servers, registry) because it frames everything. Then the offline two-device question: persist, per-device inbox or cursor, push, fetch and acknowledge on reconnect. Mention ordering and the group fan-out threshold. Ask: is history kept forever, is end-to-end encryption required (it moves search and many features to the client), and how large can groups get?

<div class="take">
<div class="lab">Key takeaway</div>

each device holds a WebSocket to a stateful chat server, and servers reach each other through pub/sub channels used only as a fast path. Durability comes from writing the message and per-device inbox entries (or relying on a per-device cursor) before delivering. Offline devices are woken by a push notification, then fetch and acknowledge on reconnect, each device independently, and clients deduplicate by message ID.

</div>

---
id: "02.04"
course: "system-design-101"
module: "02"
slug: "tcp-vs-udp"
title: "TCP vs UDP"
keyIdeas: ["**TCP**: connection-oriented, reliable, ordered delivery with retransmission, flow control and congestion control. Costs a handshake and head-of-line blocking.", "**UDP**: connectionless, no delivery or order guarantee, minimal overhead. The app handles loss if it cares.", "Use TCP for anything that must arrive intact: web pages, APIs, file transfers, database connections.", "Use UDP when late data is useless: voice/video calls, games, live metrics. A retransmitted video frame arrives too late to show.", "QUIC (HTTP/3) builds reliability on top of UDP to avoid TCP's head-of-line blocking and get faster connection setup."]
question: "Why do video calls and games use UDP while file downloads use TCP?"
answer: "In a call or game, data is only useful if it is fresh. TCP would hold back every later packet until a lost one is retransmitted, causing freezes and forcing extra buffering delay, whereas UDP delivers whatever arrives and the app conceals small gaps or simply uses the next update. A file download needs every byte in the right order and doesn't care about a few milliseconds of delay, so TCP's handshake, retransmission, ordering and congestion control are exactly right."
sources: [{"title": "Hello Interview: Networking Essentials", "url": "https://www.hellointerview.com/learn/system-design/core-concepts/networking-essentials", "note": "TCP vs UDP section."}, {"title": "Cloudflare Learning: What is UDP?", "url": "https://www.cloudflare.com/learning/ddos/glossary/user-datagram-protocol-udp/", "note": "UDP vs TCP in plain terms."}]
---

## The transport layer's job

**IP** (Internet Protocol) moves packets between machines, but it promises nothing. Packets can be lost when a router's queue overflows, duplicated, corrupted, or delivered out of order because they took different paths. Each packet is limited in size too, typically about 1,500 bytes on Ethernet, so anything bigger must be split up.

The **transport layer** sits on top of IP and decides how much of that mess the application sees. It also adds **ports**, so one machine can run many services: IP gets the packet to the machine, the port gets it to the right program. The two classic transport protocols are **TCP** and **UDP**, and a newer one, **QUIC**, sits between them.

~~~dg-layers Where TCP and UDP sit. Applications pick a transport; everything below is shared.
Application | HTTP, DNS, video calls, games, database protocols
*Transport | TCP (reliable stream) or UDP (datagrams), ports
Internet (IP) | addresses, routing, best-effort packets
Link | Ethernet, Wi-Fi, mobile radio: frames of ~1,500 bytes
~~~

## TCP: reliable, ordered streams

**TCP** (Transmission Control Protocol) gives the application the illusion of a reliable, ordered **byte stream**, like a pipe between two programs. Underneath, it works hard to maintain that illusion:

- **Handshake**: before data flows, client and server exchange **SYN**, **SYN-ACK**, **ACK**, agreeing on starting sequence numbers. That costs one round trip before the first byte of data. TLS encryption adds one more round trip with TLS 1.3, or two with TLS 1.2.
- **Sequence numbers and ordering**: every byte is numbered, so the receiver can put segments back in order and discard duplicates.
- **Acknowledgements and retransmission**: the receiver acknowledges what it has received. If the sender gets no acknowledgement within a timeout, or sees three duplicate acknowledgements (a sign that one segment went missing while later ones arrived), it sends the missing segment again.
- **Flow control**: the receiver advertises how much buffer space it has (the **receive window**), so a fast sender cannot overwhelm a slow receiver.
- **Congestion control**: the sender also limits itself to protect the network. It starts slowly (**slow start**, doubling its sending window each round trip), and when it detects loss, it assumes the network is congested and cuts its rate, then grows it again gradually.
- **Checksums** detect corrupted segments.

~~~dg-seq TCP's three-way handshake costs one round trip before any data flows, and lost segments are resent.
Client > Server : SYN (seq = x)
Server -> Client : SYN-ACK (seq = y, ack = x+1)
Client > Server : ACK + first data
Server -> Client : data segments 1, 2, 3
Client > Server : ACK 1, ACK 1 again (2 missing)
Server -> Client : retransmit segment 2
~~~

TCP connections also have a lifecycle of their own, which matters when you run servers at scale:

~~~dg-state A simplified TCP connection lifecycle from the client side. The TIME-WAIT state can pile up on busy servers that open and close many short connections.
[*] -> Closed
Closed -> SYN-sent : connect(), send SYN
SYN-sent -> *Established : receive SYN-ACK, send ACK
Established -> FIN-wait : close(), send FIN
FIN-wait -> Time-wait : receive peer's FIN, send ACK
Time-wait -> Closed : wait 2 x max segment lifetime
~~~

The side that closes first sits in **TIME-WAIT** for up to a minute or two, so delayed packets from the old connection can't be confused with a new one. A proxy opening and closing thousands of short connections per second can run out of local ports this way, which is one more reason to reuse connections (connection pooling, HTTP keep-alive).

### The price: latency and head-of-line blocking

TCP's guarantees cost latency in three places:

- **Setup**: one round trip for the handshake (plus TLS) before any data.
- **Slow start**: a new connection can't use the full bandwidth immediately; it ramps up over several round trips.
- **Head-of-line blocking**: if one segment is lost, the receiver has later segments sitting in its buffer, but TCP will not hand them to the application until the missing one arrives, because it promised order. Everything waits at least one more round trip.

For a file download, that is exactly what you want: every byte, in the right order, and a few extra milliseconds don't matter. For live audio, a packet that arrives 300 ms late is useless: the moment it described has already been played.

<div class="ex">
<div class="lab">Worked example</div>

**how packet loss limits TCP throughput**

A rough rule for a long-running TCP flow: throughput ≈ segment size / (RTT × √loss rate).

With 1,460-byte segments, a 100 ms round trip and 1% packet loss: 1,460 bytes / (0.1 s × 0.1) = 146,000 bytes per second, about 1.2 Mbit/s, however fast the link is.

Cut the loss to 0.01% and the limit rises tenfold, to about 12 Mbit/s. Cut the RTT to 10 ms (a nearby CDN server) and it rises another tenfold. This is why CDNs place servers close to users, and why modern congestion control algorithms such as BBR, which react to measured bandwidth and delay rather than to every loss, help so much on lossy mobile links.

</div>

## UDP: just send it

**UDP** (User Datagram Protocol) is almost nothing on top of IP: ports, a length and a checksum, in an 8-byte header (TCP's header is at least 20 bytes). It sends **datagrams** straight to the target with no connection setup, no acknowledgements, no retransmission, no ordering and no congestion control. Each datagram arrives whole, or not at all, and possibly out of order. The application gets each one as it arrives.

That makes UDP fast and lightweight, but the application must tolerate loss, duplicates and reordering, or handle them itself. Many UDP applications add exactly the reliability they need and no more: a sequence number to detect gaps, perhaps a retransmission for important messages only.

Where UDP is used:

- **DNS**: a lookup is one small question and one small answer. A TCP handshake would double the cost. If the reply is lost, the client simply asks again after a short timeout.
- **Voice and video calls**: a call with a moment of static is better than one that is perfectly clear but a second behind. Codecs conceal lost packets (interpolating audio, repeating the last video frame) rather than wait for them.
- **Online games**: the latest position of a player matters, not an old one. If update 1,041 is lost, update 1,042 replaces it 16 ms later.
- **Live streaming**, telemetry, and other time-sensitive data.
- **QUIC** itself, which builds a full reliable protocol on top of UDP.

~~~dg-timeline A video call over UDP when one packet is lost. The receiver conceals the gap and keeps playing instead of waiting.
[Sender] 0 ms | frames 1, 2, 3 sent, 20 ms apart
[Network] 20 ms | *frame 2 packet lost
[Receiver] 60 ms | plays frame 1
[Receiver] 80 ms | frame 2 missing: conceal from frame 1
[Receiver] 100 ms | plays frame 3 on time
~~~

### The abuse problem

Because UDP has no handshake, the source address on a datagram is never verified. That makes it easy to abuse:

- A **UDP flood** sends huge volumes of datagrams to random ports. The server burns resources checking for a listening program and replying with "port unreachable" messages.
- **Reflection and amplification attacks** send small requests with a forged source address (the victim's) to servers that send large replies, such as open DNS resolvers, so the victim is flooded with replies it never asked for.

Defences include rate limiting replies, dropping unexpected traffic at the network edge, and absorbing attacks across a large distributed network (a key service of DDoS protection providers). Protocols built on UDP, such as QUIC, add their own address validation before sending large responses.

## QUIC: the modern middle ground

**QUIC** runs over UDP but rebuilds reliability, ordering, congestion control and encryption in user space (in the application or a library, rather than in the operating system kernel). It is the transport under **HTTP/3**. Its key properties:

- **Independent streams**: like HTTP/2, many streams share one connection, but each stream has its own ordering. A lost packet only stalls the stream it belongs to; the others keep flowing.
- **Faster setup**: the transport and TLS 1.3 handshakes are combined into one round trip. Reconnecting to a known server can send data in the very first packet (**0-RTT**), with care, because 0-RTT data can be replayed by an attacker and so should only be used for idempotent requests.
- **Connection migration**: a connection is identified by a connection ID, not by the IP address and port. When a phone switches from Wi-Fi to mobile data and its IP changes, the connection survives.
- **Always encrypted**, including most of the transport metadata, which also stops middleboxes from ossifying the protocol.
- **Evolves faster**: because it lives in user space, browsers and servers can ship improvements without waiting for operating system updates.

~~~dg-compare Head-of-line blocking when one packet of stream B is lost. TCP stalls everything; QUIC stalls only stream B.
[HTTP/2 over TCP]
Packet 2 lost -> TCP holds packets 3 to 9 -> Streams A, B, C all wait
- One loss delays every stream
- Worse on lossy mobile networks
[*HTTP/3 over QUIC]
Packet 2 lost (stream B) -> QUIC delivers A and C -> Only B waits
+ Losses stay local to one stream
+ 1-RTT setup, survives network changes
- More CPU per byte than kernel TCP
~~~

| | TCP | UDP | QUIC |
| --- | --- | --- | --- |
| Setup | Handshake (1 RTT, plus TLS) | None | 1 RTT including TLS, sometimes 0 |
| Delivery | Reliable, ordered | Best effort | Reliable, ordered per stream |
| Head-of-line blocking | Yes, whole connection | No | Only within a stream |
| Congestion control | Yes, in kernel | None (app's job) | Yes, in user space |
| Typical use | Web, APIs, databases, file transfer | DNS, VoIP, games, live media | HTTP/3 |

<div class="ex">
<div class="lab">Worked example</div>

**video call app**

Signalling (join a room, mute, chat messages) goes over WebSockets on TCP, because those messages must not be lost, and a 100 ms delay on a chat message is invisible.

The audio and video media go over UDP (via WebRTC, using RTP packets). Audio is sent in 20 ms packets. With 2% loss, about 1 packet in 50 is missing; the codec fills each gap with a plausible guess, and listeners rarely notice.

Over TCP, each lost packet would hold up everything behind it for at least one extra round trip, say 80 ms. The player would have to keep a larger buffer to hide those stalls, adding permanent delay to the conversation, which is exactly what makes calls feel awkward.

</div>

## In interviews

Default to TCP (usually via HTTP) for almost everything. Bring up UDP only when the problem is real-time media, games or similar data where low latency beats completeness, and say why: "UDP because a late frame is worse than a dropped frame." Mention QUIC and HTTP/3 if the question involves mobile clients on lossy networks or connection setup latency.

## Common mistakes

- **"UDP is faster, so use it for the API."** Then you rebuild reliability, ordering and congestion control yourself, badly.
- **Forgetting that UDP has no congestion control**, so a careless UDP sender can flood a network.
- **Ignoring connection setup cost**: opening a new TCP plus TLS connection per request to a distant server wastes several round trips each time.
- **Assuming TCP means data arrived.** TCP acknowledges receipt by the other machine's kernel, not that the application processed it. Application-level acknowledgements are still needed for business guarantees.

<div class="take">
<div class="lab">Key takeaway</div>

TCP trades latency for reliability and order, which suits downloads, APIs and databases. UDP trades reliability for speed and leaves loss handling to the application, which suits real-time media where stale data is worthless. QUIC rebuilds reliable streams on UDP without connection-wide head-of-line blocking.

</div>

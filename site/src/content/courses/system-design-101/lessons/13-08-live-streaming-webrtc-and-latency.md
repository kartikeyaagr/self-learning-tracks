---
id: "13.08"
course: "system-design-101"
module: "13"
slug: "live-streaming-webrtc-and-latency"
title: "Live streaming, WebRTC, and latency"
keyIdeas: ["Live streaming: ingest from the broadcaster (RTMP, SRT or WebRTC) → transcode into renditions → package into segments → deliver via CDN.", "HLS/DASH live latency is often 10-30 s, because players buffer several segments of several seconds each. Low-latency HLS/DASH use partial segments to get to 2-5 s.", "WebRTC sends media over UDP with tiny buffers, giving sub-second latency. It's used for calls and interactive streams, but scaling to huge audiences needs media servers (SFUs) and costs more than CDN delivery.", "Choose by need: watch-only audiences tolerate seconds of delay; auctions, betting and calls don't."]
question: "Why is HLS live latency 10+ seconds while WebRTC is sub-second?"
answer: "HLS works in whole files: the encoder must finish a multi-second segment before it can be published, the playlist must be updated and then discovered by the player's next poll, the CDN must fetch it, and players deliberately buffer about three segments to survive network hiccups, so 6-second segments alone add around 18 seconds and totals of 15 to 30 seconds are normal; that file-based, buffered, stateless design is what lets CDNs cache and serve millions cheaply. WebRTC streams individual frames as small RTP packets over UDP the moment they are encoded, does not block on retransmission (it conceals lost packets), keeps only a jitter buffer of tens to hundreds of milliseconds, adapts the encoder's bitrate in real time, and uses a direct or single-server path, so end-to-end delay is a few hundred milliseconds, at the cost of per-viewer media servers."
sources: [{"title": "MDN: WebRTC API", "url": "https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API", "note": "Peer-to-peer realtime media."}]
---

## Latency is a design choice

For on-demand video, nobody cares how old the content is. For **live** video, the delay from the camera to the viewer's screen (often called **glass-to-glass latency**) matters a lot, but how much depends on the use:

- A **video call** feels broken above about 300 to 400 ms; people talk over each other.
- A **live auction**, a sports bet placed during play or an interactive game stream needs under a second or it is unfair.
- A **sports broadcast** is fine at a few seconds, although a 30-second delay means your neighbour cheers a goal before you see it.
- A **concert stream or news channel** is fine at 15 to 30 seconds.

The two dominant technologies sit at opposite ends. HTTP segment streaming (HLS and DASH, lesson 13.06) typically delivers live video **10 to 30 seconds** behind real time and scales to millions of viewers cheaply. **WebRTC** delivers it in **a few hundred milliseconds** and costs far more per viewer. This lesson explains where each one's latency comes from, which is the same thing as explaining why each one scales the way it does.

## Where HLS latency comes from

HLS is built for scale and robustness, not immediacy. Follow one video frame from the camera to a viewer.

1. **Capture and encode.** The encoder compresses frames as they arrive: tens to hundreds of milliseconds.
2. **Wait for the segment to finish.** HLS works in whole files. A frame captured at the start of a 6-second segment cannot be published until the **whole segment** is complete, up to 6 seconds later. (Apple originally recommended 10-second segments; the guidance later moved to 6.)
3. **Package and upload to the origin.** The packager writes the segment and **updates the media playlist** to list it: a few hundred milliseconds.
4. **Discovery.** The player learns about the new segment only when it next **polls** the playlist, which it does roughly once per segment duration. On average that adds half a segment.
5. **CDN propagation.** The edge fetches the new playlist and segment from the origin or shield on the first request: hundreds of milliseconds.
6. **The player's buffer.** Players traditionally start **about three segments behind the live edge** to absorb network hiccups. With 6-second segments, that is **18 seconds of buffer by design**.

Add it up and a classic HLS live stream runs **15 to 30 seconds** behind real time.

~~~dg-bars Where a classic HLS live stream's delay comes from, with 6-second segments. The player's deliberate buffer dominates.
Capture and encode | 0.5 | 0.5 s
Wait for segment to complete | 6 | up to 6 s
Package and publish playlist | 0.5 | 0.5 s
Player discovers it by polling | 3 | about 3 s
CDN fetch from origin | 0.5 | 0.5 s
*Player buffer, 3 segments | 18 | 18 s
~~~

The design choices that cause the delay are exactly the ones that make HLS scale:

- **Files over HTTP** can be cached on CDN edges, so a million viewers fetching the same segment cause one origin request per edge.
- **Large buffers** ride out network variation, so viewers on poor connections rarely stall.
- **Polling and stateless servers** mean no per-viewer connection state anywhere.

### Low-latency HLS and DASH

**Low-Latency HLS** (LL-HLS) cuts the delay to about **2 to 5 seconds** while keeping CDN cacheability:

- **Partial segments** ("parts") of 200 ms to 1 s are published as soon as they are encoded, so the player does not wait for a 6-second segment to finish.
- **Blocking playlist reloads**: the player asks for "the playlist once part 7 of segment 120 exists", and the server holds the request open until it does, replacing polling delay with a push-like response.
- **Preload hints** tell the player the URL of the next part before it exists, so it can request it early.
- Players can then sit **one or two seconds** behind the live edge instead of three segments.

Low-latency DASH achieves the same with CMAF **chunked transfer**: a segment is sent over HTTP in small chunks as it is encoded. The cost is more requests per second, more sensitivity to network jitter (a smaller buffer means more stalls on bad networks), and CDNs that must handle held-open requests well.

## WebRTC

**WebRTC** (Web Real-Time Communication) is a set of browser APIs and protocols that let browsers capture and stream audio and video, and exchange arbitrary data, **directly between peers**, often without servers in the media path. Its pieces:

- **RTCPeerConnection**: the connection between two peers, carrying media **tracks** and **data channels** (an API deliberately similar to WebSocket, for arbitrary messages).
- **Signalling**: before connecting, peers must exchange **session descriptions** in **SDP** (session description protocol) format, an **offer** and an **answer** listing codecs, media and encryption fingerprints. WebRTC deliberately does not define how; applications usually use their own WebSocket server.
- **ICE** (interactive connectivity establishment) finds a working network path between the peers through NATs and firewalls. Each peer gathers **candidates**: its local addresses, its public address as seen by a **STUN** server, and a relay address on a **TURN** server. The peers test candidate pairs and use the best that works. When no direct path exists (strict corporate firewalls, some mobile carriers), media flows through the TURN relay.
- **Transport**: media travels over **RTP** (real-time transport protocol) on UDP, encrypted with **SRTP**, using keys negotiated by **DTLS**. Data channels use SCTP over DTLS.

~~~dg-seq Setting up a WebRTC call. Signalling goes through the app's server; once ICE finds a path, media flows directly over UDP.
Alice > Signalling server : SDP offer (codecs, fingerprints)
Signalling server > Bob : forward offer
Bob -> Signalling server : SDP answer
Signalling server -> Alice : forward answer
Alice > STUN server : what is my public address?
STUN server -> Alice : 203.0.113.7:54012
Alice > Bob : ICE candidates via signalling, connectivity checks
Bob -> Alice : check succeeds on best pair
Alice > Bob : DTLS handshake, then SRTP media over UDP
~~~

## Why WebRTC is sub-second

Every source of HLS delay is designed out:

- **No segments.** Frames are sent **as soon as they are encoded**, as a continuous stream of small RTP packets (about 1,200 bytes each). There is no waiting for a multi-second file to complete.
- **UDP rather than TCP.** With TCP, a lost packet blocks delivery of everything behind it until it is retransmitted (**head-of-line blocking**). Over UDP, the receiver can ask for a retransmission (NACK) if there is time, use forward error correction, or simply **conceal** the gap (repeat the last frame, smooth over the audio) and move on. For live conversation, a brief glitch is far better than a delay.
- **Tiny jitter buffers.** A **jitter buffer** holds incoming packets briefly to smooth out variations in arrival time. WebRTC keeps it at **tens to a few hundred milliseconds**, adjusted dynamically, instead of tens of seconds.
- **The encoder adapts in real time.** Congestion control estimates available bandwidth continuously from packet timing and loss and tells the **encoder** to lower its bitrate or resolution immediately. HLS instead switches between renditions encoded in advance, one segment at a time.
- **A direct path.** Peer to peer, or through one media server, rather than through a packager, an origin and several CDN tiers.

~~~dg-compare Two ways to deliver live video. Each one's latency comes from the same choices that set its scale and cost.
[HLS or DASH]
Encoder -(6 s files)-> Origin -(HTTP, TCP)-> CDN -> Player buffer
+ Millions of viewers through CDN caches
+ Cheap per viewer, robust on bad networks
- 10 to 30 s, or 2 to 5 s low-latency
[*WebRTC]
Encoder -(RTP packets, UDP)-> SFU -> Viewer
+ Under 0.5 s glass to glass
+ Two-way: talk, react, bid
- Media servers per few thousand viewers
- Drops quality or frames instead of buffering
~~~

| | HLS / DASH | Low-latency HLS / DASH | WebRTC |
| --- | --- | --- | --- |
| Typical latency | 10 to 30 s | 2 to 5 s | 0.2 to 0.5 s |
| Transport | HTTP over TCP, whole files | HTTP, partial segments | RTP over UDP, packets |
| Scale | Millions via CDN caching | Millions via CDN | Media servers (SFUs), cascaded |
| On a bad network | Buffers absorb problems | Some stalls | Drops quality or frames immediately |
| Typical use | Broadcasts, concerts, on-demand | Sports, live shopping | Calls, conferencing, auctions, interactive streams |

## Scaling WebRTC to an audience

Pure peer-to-peer only works for small calls. In a **mesh**, each sender uploads a separate copy to every other participant: in a 6-person call each person uploads 5 streams, which saturates home uplinks. Larger sessions use a **Selective Forwarding Unit** (SFU): a media server that receives one stream from each sender and **forwards** it to every receiver, without decoding or re-encoding (which is what keeps it cheap compared with an MCU, a server that mixes everything into one stream).

SFUs use **simulcast** to serve receivers with different bandwidths: the sender encodes 2 or 3 versions at once (say 180p, 360p and 720p), and the SFU forwards to each receiver the layer that fits its connection. **Scalable video coding** (SVC) achieves the same with layered encoding of one stream. For big audiences, SFUs **cascade**: an origin SFU forwards to regional SFUs, which forward to viewers.

~~~dg-flow Scaling a WebRTC broadcast: the presenter sends simulcast layers once; cascaded SFUs forward the right layer to each viewer.
Presenter -(simulcast 180p, 360p, 720p)-> *Origin SFU -(all layers)-> Regional SFU Europe -(720p or 360p)-> 1,000 viewers
Origin SFU -(all layers)-> Regional SFU US -(layer per viewer)-> 1,000 viewers
~~~

This works, but it **costs far more per viewer** than CDN delivery. Every viewer holds a live UDP session with an SFU, which must forward every packet individually, track congestion per receiver and handle retransmissions. A single SFU server might handle a few hundred to a few thousand viewers, and cloud bandwidth for SFUs is priced well above CDN bandwidth. That is why large broadcasts still use HLS or DASH, and reserve WebRTC for the interactive participants.

Newer protocols narrow the gap: **WHIP** and **WHEP** standardise WebRTC ingest and playback over simple HTTP requests (useful for broadcasters), and work on **Media over QUIC** aims at sub-second latency with CDN-style fan-out.

## Choosing by latency budget

~~~dg-tree Choosing a delivery technology from the latency the product needs and the audience size.
How fast must viewers see it?
  Under 1 s, two-way interaction
    *Small group: WebRTC with an SFU
    Large audience: WebRTC for participants, LL-HLS for spectators
  2 to 5 s, live but watch-only
    Low-latency HLS or DASH via CDN
  10 s or more is fine
    Standard HLS or DASH, largest buffers, cheapest
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a live auction**

Bidders must see the auctioneer within half a second or bids are unfair, so the auctioneer's stream goes out over WebRTC to **2,000 bidders** through two regional SFUs, each serving 1,000 viewers at 1.5 Mbps (720p simulcast layer) or 0.5 Mbps (360p). Peak SFU egress is about 3 Gbps.

**200,000 spectators** who cannot bid watch the same event over low-latency HLS from the CDN at about 4 seconds behind: 200,000 × 3 Mbps = 600 Gbps, served from CDN edges at a much lower price per gigabyte, with the origin seeing only a few requests per part per edge.

Delivering the spectators over WebRTC instead would need about 200 SFU servers and 600 Gbps of SFU egress, several times the cost, for a latency improvement they do not need.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**tuning a sports stream from 25 seconds to 4**

A sports app runs classic HLS with 6-second segments and players three segments back: about 25 seconds behind the TV broadcast, and users complain about spoilers from social media.

Step 1: 2-second segments and players two segments back: about 8 to 10 seconds. More playlist requests and a little more stalling on bad networks.

Step 2: LL-HLS with 333 ms parts, blocking playlist reloads and players about 1 second behind the live edge: about 3 to 4 seconds. The CDN must support held-open playlist requests; the encoder must produce parts aligned with keyframes every 2 seconds.

Going below 1 second would mean WebRTC for millions of viewers, an order of magnitude more expensive. The product decides 4 seconds is good enough.

</div>

## Pitfalls

- **Promising WebRTC latency at HLS prices.** Sub-second delivery to millions is expensive; check whether the product really needs it.
- **Forgetting TURN.** Without a TURN relay, a minority of users behind strict NATs or firewalls cannot connect at all. Budget TURN bandwidth.
- **Tiny buffers on bad networks.** Low latency means less protection against jitter; expect more stalls and quality drops, and measure them.
- **Mesh calls that grow.** Peer-to-peer is fine for 1:1 and maybe 3 or 4 people; above that, use an SFU.
- **Ignoring signalling.** WebRTC gives you no signalling; you need a reliable channel (usually WebSocket, lesson 11.02) to exchange offers, answers and candidates.

## In an interview

Explain HLS latency as a sum: whole segments must finish, playlists must be published and polled, and players buffer several segments, so 6-second segments give about 20 seconds; then say that these are the very features that let CDNs serve millions. Explain WebRTC's speed as the removal of each: packets instead of files, UDP without head-of-line blocking, tiny jitter buffers, real-time encoder adaptation, direct paths. Finish with the cost of scaling WebRTC (SFUs, simulcast, cascading) and a hybrid design: WebRTC for participants, LL-HLS for the audience.

<div class="take">
<div class="lab">Key takeaway</div>

HLS live latency comes from waiting for whole multi-second segments, publishing and polling playlists, and players deliberately buffering several segments, a design that buys cheap, robust CDN-scale delivery; low-latency HLS cuts it to a few seconds with partial segments and blocking playlist requests. WebRTC sends small RTP packets over UDP as frames are encoded, conceals losses instead of waiting, keeps tiny jitter buffers and adapts the encoder in real time, so it is sub-second, but it needs SFUs and costs far more per viewer.

</div>

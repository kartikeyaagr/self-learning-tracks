---
id: "13.06"
course: "system-design-101"
module: "13"
slug: "adaptive-bitrate-and-the-cdn-decider"
title: "Adaptive bitrate and the CDN decider"
keyIdeas: ["Adaptive bitrate streaming splits video into short segments (2-10 s) encoded at several bitrates, listed in a manifest (HLS `.m3u8`, DASH `.mpd`).", "The **player** decides which bitrate to fetch for each segment, based on measured bandwidth and buffer level. It switches down when the network slows and up when it improves.", "Segments are plain HTTP files, so ordinary CDNs cache them.", "Large services also choose which CDN or edge cluster serves each client (a \"CDN decider\"), based on performance measurements and cost."]
question: "Who decides the bitrate in ABR, the server or the player?"
answer: "The player. In HTTP adaptive streaming the server or CDN only serves the manifest and segment files and keeps no per-viewer state; the client measures its download throughput, buffer level and device capabilities and chooses which rendition to request for each next segment, starting low, stepping up when segments arrive fast and the buffer is healthy, and stepping down before the buffer empties. Keeping the logic in the client is what lets segments be cached on CDNs for millions of viewers. The service may separately steer which CDN or edge the player uses (through the URLs in the manifest, DNS or content steering), but the bitrate choice is client-side."
sources: [{"title": "Apple: HTTP Live Streaming", "url": "https://developer.apple.com/streaming/", "note": "HLS manifests and variants."}, {"title": "Wikipedia: Adaptive bitrate streaming", "url": "https://en.wikipedia.org/wiki/Adaptive_bitrate_streaming", "note": "How players switch quality."}]
---

## The problem: bandwidth changes every second

A viewer's network is not a constant. A train enters a tunnel, someone else in the house starts a download, the phone hands over from Wi-Fi to 4G, the mobile cell gets busy at rush hour. Within one episode, available bandwidth can swing from 50 Mbps to 1 Mbps and back.

If you stream one fixed quality, you lose either way. Stream high quality and the video **stalls** (rebuffers) whenever bandwidth dips, which viewers hate more than anything else. Stream low quality and everyone on a good connection gets a blurry picture. **Adaptive bitrate streaming** (ABR) solves this by switching quality on the fly.

## How ABR works

The ingredients come from the transcoding pipeline (lesson 13.05):

1. The same video is encoded at **several bitrates and resolutions**, the renditions of the bitrate ladder (240p at 0.3 Mbps up to 1080p at 5 Mbps, say).
2. Each rendition is cut into **short segments** of a few seconds (2 to 6 seconds is typical), with every segment starting at a keyframe and the boundaries **aligned across renditions**, so segment 37 covers the same moment in every rendition.
3. A **manifest** (playlist) lists the renditions, their bitrates and resolutions, and where to find each segment.
4. The player downloads the manifest, picks a starting rendition, and then, **for each next segment**, decides which rendition to request. Because segments align, switching is seamless: segment 37 at 1080p is followed by segment 38 at 480p and the viewer sees only a change in sharpness.

~~~dg-tree An HLS manifest hierarchy. The master playlist lists renditions; each media playlist lists that rendition's segments.
master.m3u8 | all renditions and their bandwidths
  360p/index.m3u8 | 0.8 Mbps
    seg-001.m4s | 4 s
    seg-002.m4s | 4 s
  720p/index.m3u8 | 3 Mbps
    seg-001.m4s
    seg-002.m4s
  *1080p/index.m3u8 | 5 Mbps
    seg-001.m4s
    seg-002.m4s
  audio/index.m3u8 | 128 kbps AAC
~~~

A master playlist looks like this (simplified):

```
  #EXTM3U
  #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
  360p/index.m3u8
  #EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1280x720
  720p/index.m3u8
  #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
  1080p/index.m3u8
```

## Who decides: the player

In HTTP-based ABR, **all the adaptation logic runs on the client**. The server side is an ordinary **web server or CDN** that serves files when asked. It keeps **no per-viewer session state**: it does not know or care which rendition a given viewer is on. That is precisely what makes it scale, because any edge server can serve any segment request from cache, and segments of a popular episode are shared by millions of viewers.

The client is also the only party that **can** decide well. It sees, in real time:

- **Throughput**: how fast its recent segment downloads completed.
- **Buffer level**: how many seconds of video it has downloaded but not yet played.
- **Device capabilities**: screen resolution (no point fetching 1080p for a 360-pixel-wide window), decoder support, CPU load, battery.
- **Dropped frames**: if the device cannot decode 1080p smoothly, step down even if the network could carry it.

~~~dg-seq The player drives everything. The CDN serves files; it never knows which rendition a viewer is on.
Player > CDN edge : GET master.m3u8
CDN edge -> Player : renditions 360p to 1080p
Player > CDN edge : GET 360p/seg-001 (fast start)
CDN edge -> Player : 0.4 MB in 0.3 s, about 11 Mbps
Player > Player : buffer 4 s, throughput 11 Mbps, step up
Player > CDN edge : GET 1080p/seg-002
CDN edge -> Player : 2.5 MB in 2 s, about 10 Mbps
Player > Player : buffer 10 s, stay at 1080p
~~~

### The algorithms

There are three main families of rate-selection algorithm.

- **Throughput-based.** Estimate available bandwidth from recent segments (bytes divided by download time), usually smoothed with a moving or harmonic average over the last few segments, multiply by a **safety factor** (say 0.8), and pick the highest rendition whose bitrate is below the result.
- **Buffer-based.** Use the buffer level as the main signal. Below a **reservoir** (say 10 seconds), pick the lowest rendition to refill quickly. Above it, raise quality as the buffer grows, reaching the top rendition at a full buffer (say 30 seconds or more). The insight: the buffer already integrates all the network's ups and downs, so it is a more stable signal than noisy throughput samples.
- **Hybrid.** Combine both: throughput to choose quickly at start-up, when the buffer is empty and tells you nothing, then buffer level in the steady state. Many production players (and open-source ones like dash.js, which can switch between a throughput rule and the buffer-based BOLA rule) work this way. More advanced players optimise a predicted score over the next few segments, trading quality against stall risk and switch frequency.

Good algorithms also avoid **oscillation**: switching up and down every segment is visually distracting, so players use hysteresis (step up only after several good segments, step down immediately if the buffer is falling).

<div class="ex">
<div class="lab">Worked example</div>

**the throughput rule on real numbers**

Segments are 4 seconds long. The player just downloaded segment 12 at 720p (3 Mbps, so 4 s × 3 Mbps = 12 Mbit = 1.5 MB) in 1.0 second: measured throughput 12 Mbps. The previous two segments measured 10 and 14 Mbps. The harmonic mean of 12, 10 and 14 is about 11.7 Mbps; with a 0.8 safety factor the budget is 9.4 Mbps. The highest rendition below that is 1080p at 5 Mbps, so segment 13 is requested at 1080p.

Segment 13 (2.5 MB at 5 Mbps) then takes 20 seconds to arrive because the train has entered a tunnel: throughput 1 Mbps. Meanwhile the buffer drained from 20 seconds to 4. Both signals now say "step down hard", and segment 14 is requested at 240p (0.3 Mbps, 150 KB), which arrives in about 1.2 seconds even at 1 Mbps.

</div>

~~~dg-timeline A commute: the player adjusts the rendition for each segment from measured throughput and buffer level, and playback never stalls.
0 s | Start at 480p for a fast first frame
4 s | Segments arrive at 12 Mbps, buffer 8 s, step up to 720p
12 s | *Buffer 20 s, step up to 1080p at 5 Mbps
60 s | Tunnel: throughput falls to 1 Mbps, buffer draining
72 s | Buffer at 8 s, drop to 360p at 0.8 Mbps
90 s | Out of the tunnel, buffer refills
110 s | Back to 1080p after several good segments
~~~

### Buffer states

From the viewer's point of view, the player is in one of a few states, and the algorithm's job is to stay out of the bad one.

~~~dg-state The player's buffer states. The adaptation logic exists to avoid Rebuffering, the state viewers notice most.
[*] -> Startup
Startup -> Playing : first segment buffered
Playing -> *Steady : buffer above target, best sustainable quality
Steady -> Playing : buffer falling, step down
Playing -> Rebuffering : buffer empty, video freezes
Rebuffering -> Playing : enough buffered again
Steady -> Startup : viewer seeks, buffer flushed
~~~

## Why HTTP and segments won

Before HTTP-based ABR, streaming used dedicated protocols (RTMP, RTSP) and special servers that held a stateful session per viewer and pushed media at a controlled rate. HTTP segments replaced them for on-demand and broadcast video because:

- **They pass through firewalls and NAT** like any web traffic, on port 443.
- **Segments are ordinary files**, so **browsers, proxies and CDNs cache them**. A popular episode's segments are served from thousands of edges, and the origin sees a trickle. This is the decisive advantage.
- **Ordinary infrastructure.** Any web server or object store can be an origin, any CDN can deliver.
- **The server stays stateless**, so it scales horizontally without coordination.

The costs: more encoding and storage (every rendition of every segment), extra latency (players buffer whole segments, which matters for live video, lesson 13.08), and some awkward interactions between the player's rate choices and TCP's own congestion control.

## The protocols

- **HLS** (HTTP Live Streaming): introduced by Apple in 2009 with iOS 3.0, later published as RFC 8216. Playlists are `.m3u8` text files that index MPEG-TS or fragmented MP4 segments. A **master playlist** lists renditions; each **media playlist** lists segments.
- **MPEG-DASH** (ISO/IEC 23009-1, 2012): the international standard, with an XML manifest called the MPD (media presentation description). Widely used on Android, smart TVs and in browsers through JavaScript players.
- **CMAF** (Common Media Application Format, 2018): a common fragmented-MP4 segment format usable by both, so **one set of segment files** can serve HLS and DASH players, halving storage and doubling cache efficiency.

HLS and DASH are the two formats still actively developed; older ones (Smooth Streaming, HDS) have faded.

## The CDN "decider"

The player decides **which quality**. A different decision, **which CDN or edge server** the bytes come from, is usually made by the **service**. Call it the **CDN decider** (or steering service).

- A CDN routes each viewer to a nearby edge, typically with DNS or anycast.
- Large services use **several CDNs** (commercial ones, or their own, like Netflix's **Open Connect** appliances placed inside internet providers' networks) and choose per session, using measured performance (real-user measurements of throughput and errors by region and network) and cost.
- The choice is communicated by giving the player **different URLs**: the playback API returns a manifest whose segment URLs point at the chosen CDN, or a manifest listing several base URLs in priority order.
- **Mid-stream changes**: HLS **Content Steering** lets a steering server tell players to switch between CDN "pathways" during playback, for example when one CDN is degraded in a region. DASH has a similar steering mechanism, and players can fall back to an alternative base URL after errors.

~~~dg-compare The two decisions in a streaming session, and who makes each.
[Service: where from]
Playback API -> Steering service -> CDN choice in URLs
+ Sees performance across all viewers
+ Balances cost and capacity between CDNs
+ Can move sessions when a CDN degrades
[*Player: which quality]
Measure throughput and buffer -> Pick rendition -> Next segment
+ Sees its own network in real time
+ Knows screen size and decoder limits
+ Server stays stateless and cacheable
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a commute with both decisions**

The viewer presses play. The playback API checks the subscription and, based on the viewer's ISP and region, steers the session to an Open Connect appliance inside their provider's network, falling back to a commercial CDN. It returns a manifest with those URLs.

The player starts at 480p to get a first frame within a second, sees segments arriving at 12 Mbps with 20 seconds buffered, and steps up to 1080p (5 Mbps). In a tunnel, throughput falls to 1 Mbps and the buffer drains from 20 to 8 seconds; the player requests the next segments at 360p (0.8 Mbps) and playback never stalls.

Meanwhile the appliance at the viewer's ISP has a fault. The steering service notices the error rate climbing for that pathway and tells players to switch to the commercial CDN. The player's next segment request goes there. The viewer notices nothing. Throughout, every edge simply served whichever segment files were requested.

</div>

## Pitfalls

- **Starting too high.** A first segment at 1080p on a slow network means a long wait for the first frame. Start low (or from the last session's measured throughput) and climb.
- **Trusting one throughput sample.** A single fast segment from a burst of Wi-Fi capacity leads to an over-ambitious switch and a stall. Smooth over several samples.
- **Misaligned segments.** If renditions do not share keyframe boundaries, switches produce visual glitches.
- **Long segments.** 10-second segments make adaptation sluggish (one decision per 10 seconds) and increase start-up time.
- **Ignoring the device.** Fetching 4K for a phone screen wastes the user's data allowance.

## In an interview

Answer directly: the player decides the bitrate, segment by segment, from throughput, buffer level and device capability; the server just serves files statelessly, which is what lets CDNs cache segments for millions of viewers. Then add the second decision: the service steers which CDN a session uses, through URLs, DNS or content steering. Mention the throughput, buffer-based and hybrid algorithm families and the start-low, avoid-stalls priorities.

<div class="take">
<div class="lab">Key takeaway</div>

ABR encodes several renditions cut into short, aligned segments and lists them in a manifest; the player decides which bitrate to fetch for each segment using measured throughput, buffer level and device limits (throughput-based, buffer-based or hybrid rules), while the server or CDN just serves files statelessly, which lets segments be cached anywhere. The service side, the CDN decider, chooses which CDN or edge serves the session and can steer it mid-stream.

</div>

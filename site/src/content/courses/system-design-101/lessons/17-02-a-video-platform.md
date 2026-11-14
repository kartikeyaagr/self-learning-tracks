---
id: "17.02"
course: "system-design-101"
module: "17"
slug: "a-video-platform"
title: "A video platform"
keyIdeas: ["A video platform has an upload path and a watch path.", "Upload: resumable direct-to-object-storage upload → transcoding DAG (chunk, encode many renditions, package HLS/DASH, thumbnails) → metadata update → publish.", "Watch: client fetches metadata and manifest, then streams segments from a CDN using adaptive bitrate.", "Popular videos are pushed to CDN edges; the long tail of rarely watched videos is served from origin or regional caches, since caching everything everywhere costs too much.", "View counts, comments, recommendations and search are separate services."]
question: "Which videos go to the CDN and which are served from origin?"
answer: "Popular and trending videos, especially their first segments and the common renditions, are served from CDN edge caches near viewers, because a small share of videos gets most views and most of the bandwidth cost; predictable hits like premieres are pushed to edges in advance during off-peak hours. The long tail of rarely watched videos is served from origin object storage, usually through a regional shield cache that collapses repeated misses, because caching millions of rarely watched videos at every edge would evict popular content for almost no gain. In practice pull-through caching with LRU eviction sorts this out automatically."
sources: [{"title": "ByteByteGo: Design YouTube", "url": "https://bytebytego.com/courses/system-design-interview/design-youtube", "note": "Upload, transcoding, CDN."}, {"title": "Hello Interview: YouTube breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/youtube", "note": "Interview breakdown."}]
tags: ["Design"]
---

## What we are building

A video platform in the style of YouTube: creators upload videos, the platform processes them, and viewers watch them on phones, laptops and TVs over networks that range from fibre to a train's patchy 4G. Module 13 covered the pipeline pieces one at a time: multipart uploads (13.04), parallel transcoding (13.05), adaptive bitrate streaming (13.06) and signed URLs (13.07). This lesson puts them together into one design and then answers the question that decides most of the cost: which videos are served from the CDN, and which from origin?

### Requirements

- Creators upload videos up to tens of GB, with **resumable** uploads (a dropped connection at 80% must not restart from zero).
- Viewers watch with playback that **adapts to network speed** and starts quickly.
- Viewers see metadata: title, description, view counts.
- Scale: about **1 million uploads** and **100 million views** a day.
- Out of scope: comments, recommendations (lesson 16.05), live streaming (lesson 13.08), monetisation.

## Numbers

<div class="ex">
<div class="lab">Worked example</div>

**storage and bandwidth**

1 million uploads a day at an average of 500 MB each is 500 TB of originals a day. Transcoding into about 6 renditions (240p to 4K) adds roughly the same again, so about 1 PB of new storage a day, around 365 PB a year. Storage is a real cost, which is why some platforms keep fewer renditions for unpopular videos.

100 million views a day, average 8 minutes watched at an average of 3 Mbit/s: each view streams 8 x 60 x 3 / 8 = 180 MB. That is 18 PB of egress a day, about 1.7 Tbit/s on average, perhaps 5 Tbit/s at the evening peak.

Egress at even 1 cent per GB would be 180,000 dollars a day. Bandwidth, not storage or compute, is the dominant bill, and where it is served from is the main lever on it.

</div>

Those numbers say: uploads are a large but manageable batch-processing problem, and playback is a massive content-delivery problem. They also say the backend servers should never touch video bytes on the playback path.

## Upload

1. The client sends metadata (title, size) to the API and receives a video ID and **presigned URLs** for an object store such as S3: time-limited URLs that let the client write one specific object without holding storage credentials.
2. The client uploads straight to S3 with **multipart upload**, in parts of 5 to 10 MB, several in parallel, recording each part's number and ETag (a checksum returned by S3).
3. To resume after a failure, the client asks which parts already exist (`ListParts`) and uploads only the missing ones.
4. When all parts are in, the client calls `CompleteMultipartUpload`, and S3 emits an "object created" event that starts processing.

The application servers only handle small JSON requests. A 20 GB upload never passes through them, so they need no special scaling for it.

## Processing: a DAG of jobs

A **DAG** (directed acyclic graph) of jobs is run by an orchestrator such as Temporal or a workflow engine (lesson 08.07):

- **Split** the original into segments of a few seconds each (ffmpeg can cut at keyframes).
- **Transcode each segment in parallel** into several **renditions**: combinations of resolution and bitrate (for example 240p at 400 kbit/s up to 4K at 15 Mbit/s), with codecs such as H.264 for compatibility and VP9 or AV1 for efficiency.
- Process audio tracks, thumbnails and captions in parallel branches.
- Write **manifests**: HLS or DASH playlists that list every segment of every rendition.
- Mark the video ready in the metadata store.

Workers pass S3 URLs between steps, not files, so any worker can pick up any step. Parallelism is what makes a 2-hour film available in minutes: 1,440 five-second segments transcoded across hundreds of workers instead of one machine working for hours.

<div class="ex">
<div class="lab">Worked example</div>

**transcoding time**

A 60-minute video split into 6-second segments gives 600 segments. Transcoding one segment into 6 renditions takes a worker about 30 seconds of CPU.

On one machine: 600 x 30 s = 5 hours. With 300 workers in parallel: about 60 seconds plus split and merge overhead, so the video is ready in roughly 2 to 3 minutes.

At 1 million uploads a day averaging 10 minutes each, total transcoding is about 1,000,000 x 100 segments x 30 s = 3 billion CPU-seconds a day, which is roughly 35,000 cores busy all day. That is why platforms buy dedicated hardware encoders or transcode the long tail lazily.

</div>

Metadata (title, owner, status, rendition list) lives in a scalable store such as Cassandra, partitioned by video ID, with a cache in front, because reads outnumber writes by far.

~~~dg-flow Upload goes to storage and a transcoding DAG; playback is just manifest and segment fetches.
[Upload] Creator -(multipart PUT, presigned)-> S3 original -(upload event)-> *Transcode DAG
[Outputs] Transcode DAG -(segments + manifest)-> S3 origin
[Watch] Viewer -(manifest, then segments)-> CDN edge -(miss)-> S3 origin
~~~

## Playback: adaptive bitrate

The player downloads the **manifest**, picks a starting rendition based on its estimate of bandwidth, and downloads segments of a few seconds each. After each segment it re-measures throughput and how much video is buffered, and switches rendition up or down for the next segment. That is **adaptive bitrate (ABR)** streaming: the picture gets softer on a bad connection instead of stopping to buffer.

~~~dg-seq A player starting playback: everything after the metadata call is a plain file fetch from the CDN.
Player > API : GET /videos/v42 (title, manifest URL)
API -> Player : metadata + signed manifest URL
Player > CDN edge : GET manifest.m3u8
CDN edge -> Player : list of renditions and segments
Player > CDN edge : GET 480p segment 1 (fast start)
Player > CDN edge : GET 1080p segment 2 (bandwidth good)
CDN edge -> Player : segment from edge cache
~~~

The key observation: if the CDN holds the manifest and segments, playback does not touch your backend at all apart from one metadata call. Every one of the 5 Tbit/s at peak can come from caches.

## Which videos go to the CDN?

This is the lesson's question. A **CDN** (content delivery network) is a fleet of cache servers placed in or near viewers' internet providers. Serving from an edge in the viewer's own network is faster (tens of milliseconds instead of hundreds) and much cheaper per gigabyte than serving across the internet from your origin. But edge storage is limited: an edge site might hold hundreds of terabytes, while the catalogue is hundreds of petabytes. So the question is which bytes deserve edge space.

The answer comes from one fact: video popularity is extremely **skewed**. A small fraction of videos gets most of the views, and a huge **long tail** gets a handful each. Roughly, the top 10% of videos might account for 80 to 90% of views.

~~~dg-bars Share of daily views by video popularity group (illustrative skew).
*Top 1% of videos | 50 | 50% of views
Next 9% | 35 | 35%
Next 40% | 13 | 13%
Bottom 50% | 2 | 2%
~~~

- **Popular and new-but-trending videos** go on the CDN, in edge caches near viewers. This is where the bandwidth is.
- **The long tail** (old or rarely watched videos) is served from **origin** (object storage), usually through a regional mid-tier cache. Keeping millions of rarely watched videos at every edge would push out content people actually watch. A video watched twice a month gains almost nothing from being at the edge.
- **Not all of a popular video.** Viewers mostly watch the first minutes, and mostly in a couple of renditions (say 720p and 1080p on phones and laptops). Caches naturally hold the hot segments (the start, the common bitrates) and evict the rest. A 2-hour film might have its first 10 minutes at every edge and its last hour only at a few.

### How it happens: pull, push and tiers

- **Pull caching** (the default): the CDN fetches a segment from origin on the first request and keeps it while it stays popular; least-recently-used segments are evicted when space is needed (lesson 05.03). This sorts popular from unpopular automatically, with no one deciding anything.
- **Push, or pre-positioning**: for videos you *know* will be big (a premiere, a trending creator's new upload, a new season), push them to edges ahead of time, usually during off-peak hours when bandwidth is cheap. Netflix fills its Open Connect appliances, cache servers placed inside internet providers, overnight based on predicted popularity.
- **Tiered caching**: edge caches fall back to a regional **shield** cache before origin, so a long-tail video requested in ten cities fetches from origin once rather than ten times.

~~~dg-layers Where a video segment is served from, by popularity.
*Edge cache near viewer | hot titles and opening segments; premieres pushed ahead
Regional shield cache | moderately popular; absorbs repeated misses
Origin object storage | every rendition of every video; the long tail lives here
~~~

~~~dg-compare Pull caching versus pre-positioning, and when each fits.
[*Pull through]
First request -(miss)-> Origin -> Cached at edge
+ Automatic, no prediction needed
+ Tracks real popularity
- First viewers in each city get a miss
[Push ahead]
Prediction -(off-peak)-> Copy to edges
+ No cold misses at launch
+ Uses cheap night bandwidth
- Wasted space if prediction is wrong
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two videos**

A trending music video gets 5 million views in its first day. Its first 30 seconds at 720p and 1080p are requested in every city, so they sit in nearly every edge cache; its CDN hit rate is above 99%, and origin serves it a few hundred times in total.

A 2014 lecture recording gets 3 views a week. Each view misses the edge (its segments were evicted long ago), may hit the regional shield, and otherwise streams from origin. That costs a few hundred megabytes of origin egress a week, a rounding error, and caching it everywhere would push out videos people actually watch.

</div>

### What the hit rate is worth

<div class="ex">
<div class="lab">Worked example</div>

**why a few points of hit rate matter**

Of 18 PB of daily egress, a 95% edge hit rate leaves 0.9 PB a day served from origin or shield. At 98% it is 0.36 PB. The difference of 0.54 PB a day, at an illustrative 2 cents per GB for origin egress against near zero inside a provider's network, is about 10,000 dollars a day, over 3 million dollars a year, for three points of hit rate.

That is why video platforms tune cache sizes, pre-position aggressively and design manifests so that the opening segments (which everyone requests) are identical across devices.

</div>

## Other cost levers

- **Storage**: many renditions per video, often several times the original's size. Some platforms transcode the long tail into fewer renditions, or only on first view, to save storage and compute.
- **Codecs**: AV1 or VP9 save around 30 to 50% bandwidth compared with H.264 at the same quality, but cost far more CPU to encode. Worth it for popular videos (encode once, save on millions of views), not for the long tail.
- **Cold storage** for originals that will never be re-transcoded.

## Security and failure handling

- **Signed URLs or signed cookies** (lesson 13.07) on manifests and segments let the CDN check that a viewer is allowed to watch without calling your backend. Private and paid videos add DRM.
- **Origin protection**: if a popular video's segments are evicted everywhere at once (say, after a CDN configuration mistake), millions of requests could stampede origin. The shield tier collapses identical concurrent misses into one origin fetch (request coalescing).
- **Transcode failures**: each DAG step retries independently; a poison video (corrupt file) is marked failed after a few attempts and the creator is told.
- **Multi-CDN**: large platforms use several CDNs and steer traffic between them by price and performance, which also covers a single CDN's outage.

## Common mistakes

- Uploading through application servers.
- Transcoding a whole video serially on one machine.
- Proposing to put every video on every edge.
- Ignoring that bandwidth dominates cost.
- Forgetting manifests, so there is nothing for the player to adapt with.

## In an interview

Do the egress calculation first; it is the number that makes everything else obvious. Draw upload and playback as separate paths. For the CDN question, say "popularity is skewed, so pull-through caching with LRU keeps the head at the edge and the long tail at origin behind a shield; pre-position predictable hits". Ask: what share of views is on new videos, are there premieres, and what is the device mix?

<div class="take">
<div class="lab">Key takeaway</div>

upload directly to object storage in parts, transcode segments in parallel into several renditions with manifests, and stream with adaptive bitrate so playback is just file fetches. Let the CDN pull and keep what is popular (and pre-position predictable hits); serve the long tail from origin through a shield cache, because popularity is highly skewed, edge space is limited, and bandwidth is the largest cost.

</div>

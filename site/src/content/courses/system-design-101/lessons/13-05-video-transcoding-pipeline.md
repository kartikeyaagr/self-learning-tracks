---
id: "13.05"
course: "system-design-101"
module: "13"
slug: "video-transcoding-pipeline"
title: "Video transcoding pipeline"
keyIdeas: ["Transcoding converts the uploaded video into many renditions: resolutions (240p to 4K), bitrates and codecs (H.264, VP9, AV1), packaged for streaming.", "Split the video into chunks (by scene or a few seconds each), transcode chunks in parallel across many workers, then stitch. This turns hours of encoding into minutes.", "Model the pipeline as a DAG: validate → split → encode per rendition → package (HLS/DASH segments) → thumbnails → publish.", "Netflix uses per-title and per-shot encoding to choose bitrates by content complexity.", "Store outputs in object storage and push popular content to the CDN."]
question: "Why split a video into chunks before transcoding?"
answer: "Encoding a whole film sequentially on one machine can take days, needs the entire huge source on local disk, and loses all progress if the machine fails late. Splitting it at keyframes into chunks lets many machines encode in parallel, so time stays roughly constant regardless of length (Netflix went from days to a few hours), a failed or reclaimed spot instance only costs one chunk's retry, each worker needs little storage, each chunk can be quality-checked as soon as it finishes, and the short tasks fit cheap elastic capacity. The price is reassembling chunks in the right order without dropped or duplicated frames and verifying the stitched stream, which Netflix does by comparing fingerprints with the source."
sources: [{"title": "Netflix: High quality video encoding at scale", "url": "https://netflixtechblog.com/high-quality-video-encoding-at-scale-d159db052746", "note": "Chunked parallel encoding."}, {"title": "ByteByteGo: Design YouTube", "url": "https://bytebytego.com/courses/system-design-interview/design-youtube", "note": "Transcoding DAG."}]
---

## What the pipeline is for

An uploaded video arrives as **one large file in one format**: a camera's high-bitrate H.264, a studio's ProRes master, a phone's HEVC recording. Viewers cannot use it as it is. A phone on 4G cannot stream a 50 Mbps master; an old smart TV cannot decode the newest codec; a player that adapts to bandwidth (lesson 13.06) needs the same video at several quality levels, cut into short segments. The **transcoding pipeline** turns the one upload into all of those.

A few terms first:

- A **codec** (coder-decoder) is the compression method: H.264 (AVC), H.265 (HEVC), VP9, AV1. Newer codecs compress better (AV1 is roughly 30% to 50% smaller than H.264 at the same quality) but are slower to encode and not supported everywhere.
- A **container** wraps encoded video, audio and subtitles into a file: MP4, MPEG-TS, WebM, MKV.
- **Transcoding** means decoding the source and re-encoding it with different settings: a different codec, resolution or bitrate.
- A **rendition** is one output version: for example "720p, H.264, 3 Mbps". The set of renditions is the **bitrate ladder**.

Netflix's pipeline is a good reference shape: ingest high-quality sources, **inspect** them, encode **multiple codec profiles** (historically VC1, H.264 Baseline, H.264 Main and HEVC) at **multiple quality levels from 100 kbps to 16 Mbps**, to serve connections from under 0.5 Mbps on cellular to over 100 Mbps on fibre, then package the results and deploy them to a CDN.

~~~dg-tree One source fans out into a ladder of renditions per codec. Every leaf is an independent encode, and every encode is split into chunks.
Source master, 4K
  H.264, plays everywhere
    1080p | 5 Mbps
    720p | 3 Mbps
    480p | 1.2 Mbps
    360p | 0.7 Mbps
    240p | 0.3 Mbps
  *AV1, newer devices
    2160p | 12 Mbps
    1080p | 3 Mbps
    720p | 1.8 Mbps
  Audio
    AAC stereo | 128 kbps
    Dolby surround | 640 kbps
~~~

## How video compression shapes the pipeline

To understand why you can split a video into chunks, you need one fact about how it is compressed. Video codecs store most frames as **differences** from other frames.

- An **I-frame** (intra frame, or **keyframe**) is a complete picture, compressed on its own like a JPEG. It is large.
- A **P-frame** stores only what changed since an earlier frame. A **B-frame** can refer to frames before and after it. These are small.
- A **GOP** (group of pictures) is a keyframe plus the frames that depend on it, typically 2 to 10 seconds long. A **closed GOP** does not reference frames outside itself.

So a decoder can start cleanly only at a keyframe. That is why chunk boundaries (and later, streaming segment boundaries) must fall on keyframes or closed GOPs: then each chunk can be decoded and encoded independently, and the encoded chunks can be joined without glitches.

## Inspect first

Netflix rejects bad sources rather than patching them: "garbage in, garbage out". A source with corrupted frames, a wrong frame rate or a production error (a few seconds of black, a missing audio channel) produces bad encodes for every rendition, and fixing that after encoding wastes all the compute. **Inspection** verifies the source conforms to its specification, detects problems, and generates metadata the encoder needs (resolution, frame rate, colour space, scene changes). If a source fails, the content partner is asked to redeliver it.

A 4K studio master can be **larger than an EC2 instance's local disk**, so inspection itself is parallelised:

1. **Index** the file: record byte offsets for frame-accurate seeking, the resolution and the frame count.
2. **Inspect chunks in parallel** on different instances, each reading only its byte range. Each computes spatial and temporal **fingerprints** (compact signatures of what each frame looks like and how it changes).
3. **Aggregate** the results into one report and one fingerprint set for the whole source.

Netflix inspects a 4K source in **under 15 minutes, independent of its length**: a longer film just has more chunks running at the same time.

## Why chunk the encoding

This is the heart of the design. Encoding a whole film sequentially on one machine is how pipelines started, and it fails in five ways that chunking fixes.

1. **Parallelism cuts wall-clock time.** Before chunked encoding, a 1080p movie **could take days** to encode across all its profiles. With chunks encoded on many instances at once, a title is inspected and encoded for every profile, with quality checks, **within a few hours**, and the time no longer grows with the length of the video.
2. **Small failures stay small.** If an instance dies or is reclaimed (Netflix runs encodes on spare capacity from an internal spot market, where instances can be taken back at any time), only that chunk's work is lost and rescheduled. A failure 40 hours into a 48-hour single encode would restart everything.
3. **Less local storage.** Each worker needs only its chunk (a few hundred MB), not the whole multi-hundred-gigabyte source.
4. **Earlier error detection.** Each encoded chunk is checked as soon as it finishes; problems are retried or triaged immediately instead of being discovered when the whole video is done.
5. **Elastic capacity.** Short tasks fit into spare compute anywhere, so the pipeline scales up for a big release and back down afterwards, using cheap interruptible capacity.

~~~dg-compare One long sequential encode versus many short parallel ones, for a two-hour film.
[One job per rendition]
2 h film -(one encoder)-> One long encode
- Days for all renditions
- A late failure restarts everything
- Needs the whole source on local disk
[*Chunked]
2 h film -(split at keyframes)-> 40 chunks on 40 workers -(assemble)-> Stream
+ Hours, nearly independent of length
+ A failure redoes one 3-minute chunk
+ Runs on cheap interruptible capacity
- Must reassemble and verify carefully
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 10-minute 1080p upload**

The file is split at keyframes into 20 segments of 30 seconds. Encode targets: 1080p, 720p, 480p, 360p and 240p in H.264, plus 1080p and 720p in AV1: 7 renditions × 20 segments = **140 tasks**.

Sequentially, H.264 at 1080p might run at about real time on one machine (10 minutes for the whole video), the lower resolutions faster, and AV1 several times slower: perhaps 60 minutes of compute in total for all renditions.

With 140 workers, every task runs at once. The job takes about as long as the slowest single task, one 30-second AV1 1080p segment, perhaps 2 to 3 minutes, plus splitting and assembly. If one worker dies, one 30-second task is retried, adding a couple of minutes, not an hour.

</div>

## The task model

A chunked pipeline is a **workflow**: a directed acyclic graph (DAG) of tasks with dependencies. "Split" must finish before "encode chunk 7 at 720p" can start; all 720p chunks must finish before "assemble 720p". A workflow engine (a managed one, or a queue plus a coordinator service) tracks each task's state and hands runnable tasks to a pool of workers.

~~~dg-state One encode task in the workflow. Tasks are idempotent, so a retry simply overwrites the output.
[*] -> Queued
Queued -> Running : worker claims task, lease 10 min
Running -> Verifying : chunk encoded, output written
Verifying -> *Done : quality and frame count pass
Verifying -> Queued : check fails, retry
Running -> Queued : worker lost or lease expired
Queued -> Failed : retries exhausted, alert
~~~

Two properties make this robust:

- **Idempotent tasks.** Each task writes its output to a deterministic key (`job/v77/720p/chunk-007.mp4`). Running it twice produces the same file, so retries are always safe.
- **Leases, not locks.** A worker claims a task for a limited time and renews the claim while it works. If the worker vanishes, the lease expires and the task returns to the queue (the same pattern as a work queue with visibility timeouts, lesson 08.06).

## Putting chunks back together safely

Chunking introduces new failure modes: chunks assembled in the **wrong order**, frames **dropped or duplicated** at chunk boundaries, slight timing drift that throws audio out of sync. Netflix's assembler stitches each stream's chunks and **validates the result by comparing fingerprints** of the encoded stream against the source's fingerprints from inspection: if a frame is missing or out of place, the fingerprints do not line up. It also computes **full-reference quality metrics** (which compare each encoded frame with the corresponding source frame; VMAF is Netflix's well-known one) for every output stream, to monitor quality and catch encoder bugs. For sources that pass inspection, Netflix achieves **99.99% or better** encode reliability.

There is one more subtlety: rate control. An encoder targeting 3 Mbps decides how many bits each frame gets by looking at what comes before and after. A chunked encoder sees only its chunk, so quality can jump slightly at boundaries. Pipelines handle this with some overlap between chunks, two-pass encoding per chunk, or choosing chunk boundaries at **scene changes**, where a jump in quality is invisible.

## Smarter ladders

A fixed ladder (the same bitrates for every video) wastes bits. A cartoon with flat colours looks perfect at 1080p and 1.5 Mbps; an action film full of rain and grain needs 6 Mbps for the same visual quality. **Per-title encoding** analyses each video's complexity and picks its own ladder. Going further, **shot-based encoding** picks settings per scene. Both rely on the chunked architecture: trying several settings per chunk and keeping the best is only affordable when chunks run in parallel.

~~~dg-bars Illustrative bitrates needed at 1080p for similar visual quality, by content type. A per-title ladder saves bits on easy content.
Animation | 1.5 | 1.5 Mbps
Talk show | 2.5 | 2.5 Mbps
Drama | 4 | 4 Mbps
*Sports, action | 6 | 6 Mbps
Fixed ladder, any title | 5.8 | 5.8 Mbps
~~~

## A YouTube-style pipeline for user uploads

For user-generated video the same structure applies, with a focus on speed to first playback:

1. The original is uploaded directly to object storage (lessons 13.01 and 13.04).
2. The object-created event starts a workflow that validates the file (is it really a video, is it too long, is it corrupt?), extracts metadata and a thumbnail, and splits the video into segments at keyframes.
3. A queue of encode tasks (segment × resolution × codec) feeds a pool of workers.
4. Assemblers join the encoded segments and **package** them into HLS and DASH segments and manifests (lesson 13.06), and write them to storage, from where the CDN pulls them.
5. The video is marked ready. To shorten the wait, publish a **low resolution first** (360p is quick to encode) so the video is watchable within a minute, and add higher renditions as they finish. Expensive codecs like AV1 can be encoded later, and only for videos that get enough views to justify the compute.

<div class="ex">
<div class="lab">Worked example</div>

**prioritising compute by popularity**

A platform gets 500 hours of video uploaded per minute. Encoding everything in AV1 at all resolutions might cost 5 times the H.264 compute. But most uploads are watched a handful of times.

Policy: encode every upload immediately in H.264 at 360p to 1080p. When a video passes 1,000 views, schedule AV1 renditions as low-priority tasks on spare capacity. If 2% of videos pass that threshold and they account for 80% of watch time, the platform gets most of AV1's bandwidth saving for about 10% extra compute instead of 400%.

</div>

## Pitfalls

- **Chunk boundaries off keyframes.** Chunks that cannot be decoded on their own, or that duplicate frames when joined.
- **No verification after assembly.** A dropped frame per chunk is invisible in a spot check and obvious to viewers as audio drift.
- **Non-idempotent tasks.** Retries that append rather than overwrite create corrupted outputs.
- **One queue for everything.** A celebrity's 4-hour upload should not block thousands of 30-second clips. Use priorities or separate queues by job size.
- **Encoding everything in every codec.** Most compute goes to videos nobody watches. Encode cheaply first, expensively when demand proves it.

## In an interview

Sketch the stages: upload to storage, validate and inspect, split at keyframes, encode chunks in parallel per rendition, assemble and verify, package for streaming, push to the CDN, mark ready. Then answer the "why chunks" question with the five reasons: parallelism (days to hours), small failures, little local disk, early error detection, elastic cheap capacity. Mention that the price is careful reassembly and verification, and that publishing a low rendition first improves time to playback.

<div class="take">
<div class="lab">Key takeaway</div>

a transcoding pipeline inspects the source, splits it at keyframes into chunks, encodes each chunk for every codec and bitrate in parallel as idempotent, leased tasks, then assembles and verifies the streams (Netflix compares fingerprints with the source) before packaging for the CDN. Chunking turns a days-long, fragile job into hours of small, retryable, parallel tasks with modest local storage and early error detection, at the cost of careful reassembly and boundary handling.

</div>

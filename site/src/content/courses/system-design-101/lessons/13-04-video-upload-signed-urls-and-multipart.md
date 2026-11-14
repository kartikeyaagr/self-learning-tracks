---
id: "13.04"
course: "system-design-101"
module: "13"
slug: "video-upload-signed-urls-and-multipart"
title: "Video upload: signed URLs and multipart"
keyIdeas: ["Large video uploads need to be resumable and fast.", "**Multipart upload**: split the file into parts (S3: 5 MB to 5 GB each, up to 10,000 parts), upload parts in parallel with their own presigned URLs, then call complete to assemble them.", "If an upload fails at 80%, only the failed parts are retried; finished parts are kept.", "Track upload state (upload ID, parts done) so a client can resume after an app restart.", "Clean up abandoned multipart uploads with a lifecycle rule, since stored parts cost money."]
question: "A 5 GB upload fails at 80%. How does multipart upload let you resume?"
answer: "The file was started as a multipart upload with an upload ID, and each part acknowledged so far (about 400 of 500 with 10 MB parts) has a part number and ETag saved persistently by the client, the server or both. Initiated uploads do not expire, so after the failure the client resumes the same upload ID, optionally verifies received parts with ListParts, gets fresh presigned URLs and uploads only the parts that were not acknowledged (roughly the last 20%, possibly in parallel), then calls complete with all part numbers and ETags so the store assembles the object. Only the parts in flight at the failure are re-sent, not the whole 5 GB."
sources: [{"title": "AWS S3: Multipart upload", "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html", "note": "Parallel, resumable uploads."}]
tools: ["S3"]
---

## Why one big upload fails

Lesson 13.01 moved uploads off your servers with presigned URLs. For a photo, a single presigned `PUT` is enough. For a **5 GB video** from a phone, a single HTTP request is fragile, slow and sometimes impossible.

- **All or nothing.** A single request either completes or fails. A network blip at 80% (the train enters a tunnel, Wi-Fi hands over to 4G, the app is backgrounded and the OS kills its socket) loses all 4 GB sent so far, and the next attempt starts from zero. On a 10 Mbps uplink, 4 GB is nearly an hour of wasted transfer.
- **One connection is slow.** A single TCP connection's throughput is limited by its congestion window and the round-trip time. Over a long-distance path it may reach only a fraction of the available bandwidth. Several connections in parallel fill the pipe much better.
- **Hard limits.** S3 accepts at most **5 GB in a single PUT**. Larger objects must be uploaded in parts.
- **Unknown length.** A single PUT needs the total size up front. A live recording or a file being generated on the fly does not have one yet.

The fix, in S3 and in every major object store, is to split the upload into independently uploaded pieces.

## Multipart upload

**Multipart upload** sends one object as a set of independently uploaded **parts**, in any order and in parallel, which the store assembles into a single object when you finish. S3 recommends it for objects of **100 MB or more** and requires it above 5 GB. It has three steps.

1. **Initiate** with `CreateMultipartUpload`. S3 returns an **upload ID**, a handle used in every later call.
2. **Upload parts** with `UploadPart`. Each part has a **part number from 1 to 10,000** chosen by you, and S3 returns an **ETag** (an identifier for that part's content, in practice derived from its MD5) for each. Uploading the same part number again simply **overwrites** that part, which makes retries safe.
3. **Complete** with `CompleteMultipartUpload`, sending the upload ID and the list of part numbers with their ETags. S3 concatenates the parts **in part-number order** and creates the object atomically: before this call, no object exists at the key; after it, the whole object does.

At any time you can **abort** with `AbortMultipartUpload`, which discards the uploaded parts.

### The limits

| Limit | Value |
| --- | --- |
| Part size | 5 MiB to 5 GiB (the last part may be smaller) |
| Number of parts | 1 to 10,000 |
| Object size | up to 5 TiB |
| Single PUT without multipart | up to 5 GB |
| Parts listed per ListParts call | up to 1,000 |

Part numbers do not need to be consecutive (you could use 1, 5 and 9), and parts can be uploaded in any order; only the numbers in the completion request decide the final order.

## Doing it directly from the client

Combining this with lesson 13.01, the client uploads parts **straight to storage** with presigned URLs, while your API keeps control of the upload. Each `UploadPart` request is signed for a specific upload ID and part number, so a client can only upload the parts you authorised.

~~~dg-seq Multipart with presigned parts: the API starts the upload and signs one URL per part; the client uploads parts in parallel and the API completes it.
Client > API : start upload (name, 5 GB)
API > S3 : CreateMultipartUpload
S3 -> API : uploadId
API -> Client : uploadId + presigned URLs for parts 1 to 500
Client > S3 : PUT parts 1..500, 4 at a time
S3 -> Client : ETag for each part
Client > API : complete(uploadId, part numbers + ETags)
API > S3 : CompleteMultipartUpload
S3 -> API : object created
API -> Client : upload done
~~~

For very many parts, the API can hand out presigned URLs in batches (say 50 at a time) rather than all 500 up front, so that URLs do not expire before the client reaches them and so that the server sees progress.

## The benefits

- **Throughput.** Upload parts **in parallel** over several connections. On a long-distance link where one connection reaches 50 Mbps, eight connections may reach 300 to 400 Mbps.
- **Quick recovery from network issues.** A failed part is retried on its own; parts already acknowledged are unaffected. The most you lose to a failure is the parts in flight.
- **Pause and resume.** An initiated multipart upload **does not expire**: it stays until you complete or abort it. The client can stop for an hour, or a day, and continue.
- **Start before you know the final size.** Upload parts as the file is being produced (a recording, a database dump being streamed), then complete when it ends.

~~~dg-bars Wall-clock time to upload 5 GB on a link that one connection can only drive at 50 Mbps, as parallel part uploads increase.
1 connection | 800 | about 13 min
2 parallel parts | 400 | about 7 min
4 parallel parts | 200 | about 3.5 min
*8 parallel parts | 110 | about 2 min, link nearly full
~~~

## Resuming at 80%

Here is the scenario from the question, worked through. With **10 MB parts**, a 5 GB file has **500 parts**. At 80%, about **400 parts** have been uploaded and acknowledged with ETags; a few more were in flight when the network dropped.

1. **Remember what is done.** The client (or the server, which tracked progress) knows the upload ID and the part numbers and ETags of every completed part.
2. **Reconnect and continue the same upload.** Because the upload has not expired, the client asks the API for fresh presigned URLs (the old ones may have expired) for the remaining parts, **401 to 500**, plus any earlier part whose upload never returned an ETag.
3. **Upload only those parts.** About 1 GB is sent, not 5 GB.
4. **Complete** with all 500 part numbers and ETags.

~~~dg-timeline A 5 GB upload in 10 MB parts that fails at 80% and resumes. Only the missing parts are re-sent.
10:00 | Initiate, uploadId u-7f3, 500 parts planned
10:00 to 10:27 | Parts 1 to 400 uploaded, ETags saved on the device
10:27 | *Network drops, parts 401 to 404 were in flight
11:00 | App reopens, reads saved state, verifies with ListParts
11:01 | Fresh URLs for parts 401 to 500, upload 100 parts
11:08 | Complete with 500 part numbers and ETags
~~~

### Where the progress is recorded

The client should **record part numbers and ETags itself**, persistently (on disk, not just in memory, so it survives the app being killed). The server can also keep track, for example by storing per-part status in the file's metadata record as parts are reported, so another device or a support tool can see progress.

S3's `ListParts` call returns the parts it has received for an upload. Use it to **verify** before completing (for example after a crash, to check which parts really arrived), but not as the only source for building the completion request: the client knows which ETags it expects, and comparing the two catches parts that were overwritten or corrupted.

<div class="ex">
<div class="lab">Worked example</div>

**the resume in numbers**

5 GB file, 10 MB parts, 500 parts. Uplink 20 Mbps (2.5 MB/s), 4 parts in parallel. Each part takes about 16 seconds on its share of the link, and the whole upload would take about 34 minutes.

At 80% (about 27 minutes in), the connection drops with 4 parts in flight. Lost work: at most 4 partial parts, 40 MB. On resume, the client uploads 100 parts (1 GB), taking about 7 minutes.

Without multipart, the failure would have cost 27 minutes of transfer and 4 GB of mobile data, and the retry would need the full 34 minutes again, with the same chance of failing.

</div>

## Choosing a part size

Part size is a trade-off.

- **Larger parts** mean fewer requests (less per-request overhead and fewer presigned URLs to manage), but more data re-sent per failure and coarser progress.
- **Smaller parts** mean cheap retries and smooth progress, but more requests and a lower maximum object size (10,000 parts × part size).

A good rule: size parts so each takes a few seconds to tens of seconds on the expected network, and make sure 10,000 parts covers the largest file you accept. Ranges of **8 to 64 MiB** are common.

<div class="ex">
<div class="lab">Worked example</div>

**picking a part size**

A 5 GB file in 5 MiB parts is about 1,000 parts: fine. A 200 GiB file in 5 MiB parts would need about 41,000 parts, over the 10,000 limit, so it needs parts of at least about 21 MiB; in practice 32 MiB, giving about 6,400 parts.

On a phone uploading at 4 Mbps, a 32 MiB part takes over a minute, so a dropped connection loses a lot. For mobile clients and files up to a few GB, 8 MiB parts (about 16 seconds each at 4 Mbps) are a better fit. Some clients choose the part size from the file size: max(8 MiB, file size ÷ 9,000).

</div>

## The upload as a state machine

It helps to see the multipart upload, as your system tracks it, as a lifecycle.

~~~dg-state A multipart upload as the application tracks it. Abandoned uploads must be aborted, because their parts are billed.
[*] -> Initiated
Initiated -> Uploading : first part acknowledged
Uploading -> Paused : network lost or app closed
Paused -> Uploading : resume with fresh URLs
Uploading -> Completing : all ETags collected
Completing -> *Complete : CompleteMultipartUpload succeeds
Paused -> Aborted : abandoned for 7 days, lifecycle rule
Completing -> Uploading : a part is missing or bad, re-send it
~~~

## Cleaning up and integrity

- **Incomplete uploads cost money.** S3 stores and bills for uploaded parts until the upload is completed or aborted, and these parts are invisible in normal object listings. A busy app can quietly accumulate terabytes of abandoned parts. Abort abandoned uploads in your reconciliation job, and add a bucket **lifecycle rule** with `AbortIncompleteMultipartUpload` after N days (say 7) as a safety net.
- **Checksums.** SDKs send a checksum with each part (CRC32, CRC32C, SHA-1, SHA-256 or CRC64NVME), and S3 rejects a part whose bytes do not match (`BadDigest`). **Full-object** checksums (such as CRC64NVME) cover the whole object; **composite** checksums are a checksum of the per-part checksums. With full-object checksums, part numbers must be consecutive starting at 1.
- **The final ETag is not the file's MD5.** For a multipart object it is the MD5 of the concatenated part MD5s followed by a dash and the part count, such as `"9b2cf535...-500"`. Do not compare it with a locally computed MD5 of the file; use the checksums above.
- **Concurrent uploads to one key.** Two multipart uploads to the same key can both complete, and the last one wins. If overwriting must be prevented, use a conditional write (`If-None-Match: *`) on completion, or give every upload a unique key.

## Other resumable protocols

The same idea appears elsewhere with different shapes:

- **Google Cloud Storage resumable uploads** use one session URL; the client sends chunks sequentially and, after a failure, asks the server how many bytes it has, then continues from that offset.
- **tus** is an open protocol for resumable HTTP uploads with the same offset-based design, used by several upload libraries and video platforms.

Offset-based protocols are simpler (no ETag list) but sequential; S3-style parts allow parallelism.

## Common mistakes

- **Keeping progress only in memory.** The app is killed, the upload ID and ETags are gone, and the next attempt starts a new upload while the old parts sit there being billed.
- **No lifecycle rule.** Abandoned parts accumulate forever.
- **Parts too small for the largest file.** 5 MiB parts cap uploads at about 48 GiB.
- **Presigning all parts with a short expiry.** The later URLs expire before the client reaches them. Issue them in batches or with expiries that match the expected upload time.
- **Comparing the multipart ETag with an MD5.** It never matches.

## In an interview

Describe the three calls (initiate, upload parts, complete) and the limits, then answer the resume question directly: the upload ID persists, completed parts and their ETags are recorded, so after a failure only missing parts are sent, possibly in parallel with fresh presigned URLs, before completing with the full list. Add the operational points that show experience: lifecycle rules for incomplete uploads, choosing part size against the 10,000-part limit, and checksums.

<div class="take">
<div class="lab">Key takeaway</div>

multipart upload splits a large object into numbered parts (1 to 10,000, 5 MiB to 5 GiB each) uploaded independently and in parallel, then assembled atomically on completion. A failure only costs the parts in flight: resume with the same upload ID, upload the missing parts with fresh presigned URLs and complete with every part's ETag. Persist progress on the client, pick part sizes for the network and the largest file, use checksums, and abort or expire incomplete uploads, since their parts are billed.

</div>

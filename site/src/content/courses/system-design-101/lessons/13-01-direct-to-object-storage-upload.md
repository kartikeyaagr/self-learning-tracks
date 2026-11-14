---
id: "13.01"
course: "system-design-101"
module: "13"
slug: "direct-to-object-storage-upload"
title: "Direct-to-object-storage upload"
keyIdeas: ["Instead of streaming file uploads through your app servers, let clients upload directly to object storage.", "Flow: client asks your API for permission → API checks auth and returns a presigned URL (time-limited, scoped to one key and method) → client PUTs the file straight to S3 → S3 event or client callback tells your API the upload finished.", "Your servers never carry the bytes, so they don't need big bandwidth, memory or long-running connections.", "Validate after upload: check size and type, scan for malware, then mark the file as available.", "Use multipart upload for large files."]
question: "Why upload directly to object storage instead of through your app servers?"
answer: "Proxying makes every byte cross your servers twice, ties up connections, threads and buffers for the full length of slow uploads, runs into gateway request size and timeout limits, and forces you to scale API servers just to move data that object storage is built to absorb. With a presigned URL, the backend only authorises the upload and signs a short-lived URL for one server-chosen object key (optionally fixing size, type and checksum); the client sends the bytes straight to object storage, and an object-created event updates the pending database record, with reconciliation jobs and lifecycle rules cleaning up abandoned uploads."
sources: [{"title": "AWS S3: Presigned URLs", "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html", "note": "Clients upload directly to S3."}, {"title": "Hello Interview: Handling large blobs", "url": "https://www.hellointerview.com/learn/system-design/patterns/large-blobs", "note": "The pattern in interviews."}]
tools: ["S3"]
---

## The obvious path, and why it hurts

Users upload things: profile photos, documents, videos, backups. The obvious design is that the client sends the file to your **API server**, which checks the user is allowed, then forwards the bytes to **object storage** (S3, Google Cloud Storage, Azure Blob Storage; lesson 07.13 covers why large files belong there rather than in a database). For a 200 KB profile photo, that is fine. For videos and large files, it is wasteful in four ways.

- **You pay for the bandwidth twice.** Every byte crosses your servers' network once on the way in and once on the way out to storage. Your load balancers, NAT gateways and servers all carry it.
- **Slow clients tie up servers.** A 2 GB video from a phone on a 5 Mbps uplink takes nearly an hour. For that whole time the upload holds a connection, a thread or event-loop slot, memory buffers, and often a slot on the load balancer.
- **Limits and timeouts.** Load balancers, API gateways and serverless functions cap request size and duration (an API gateway may cap bodies at 10 MB and requests at 30 seconds; serverless functions often cap payloads at a few MB). Long uploads hit them and fail at the worst moment.
- **You scale the wrong thing.** API servers exist to run business logic. Adding servers just to shovel bytes is expensive, and a burst of uploads can starve the normal API traffic on the same fleet.

Object storage, by contrast, is built for exactly this job: it accepts enormous numbers of parallel uploads, from anywhere, at high throughput, and bills you for storage rather than for your servers' time. So the better design lets the client upload **directly** to object storage, while your API keeps control of **who** may upload **what**.

~~~dg-compare Proxying uploads through the API versus uploading directly to storage. The direct path keeps control in the API but takes the bytes off it.
[Through the API]
Client -(2 GB)-> API server -(2 GB)-> Object storage
- Bytes cross your network twice
- Server busy for the whole upload
- Hits gateway size and time limits
[*Direct to storage]
Client -(small request)-> API -(signed URL)-> Client -(2 GB)-> Object storage
+ API handles only metadata
+ Storage absorbs the bandwidth
+ Upload can be parallel and resumable
~~~

<div class="ex">
<div class="lab">Worked example</div>

**what proxying costs at scale**

A video app receives 1 million uploads a day, averaging 50 MB: 50 TB a day. Averaged over the day that is about 4.6 Gbps, with peaks perhaps 3 times higher at about 14 Gbps. Proxied, your fleet carries 14 Gbps in and 14 Gbps out at peak.

Now the connection time. Mobile uplinks average perhaps 4 Mbps, so a 50 MB upload takes about 100 seconds. At peak, about 35 uploads start per second, so by Little's law (concurrency = arrival rate × duration) about 3,500 uploads are in progress at any moment, each holding a connection and buffers on an API server.

Direct: the API handles 35 small JSON requests per second to issue upload URLs and 35 completion events per second. A single small server could do it. The 14 Gbps and the 3,500 slow connections become the object store's problem, which is what it is designed and priced for.

</div>

## Presigned URLs

A **presigned URL** is a URL that grants **time-limited** permission to perform **one specific operation** on **one specific object**, to anyone who holds it, without that holder having any cloud credentials. Your backend creates it by **signing** a description of the request with its own credentials:

- the **bucket** and **object key** (the path of the object);
- the **HTTP method**: `PUT` to upload, `GET` to download;
- the **expiry**;
- optionally, **headers** that must be present with exact values, such as `Content-Type` or `Content-Length`, or a checksum.

The signature is an HMAC computed with a key derived from your secret access key, the date, the region and the service. No call to S3 is needed to create one: signing is pure computation on your server, taking microseconds. S3 recomputes the same HMAC when the request arrives and rejects it if anything was changed.

```
https://media-uploads.s3.eu-west-1.amazonaws.com/raw/9f1c2e.mp4
  ?X-Amz-Algorithm=AWS4-HMAC-SHA256
  &X-Amz-Credential=AKIA.../20261112/eu-west-1/s3/aws4_request
  &X-Amz-Date=20261112T101500Z
  &X-Amz-Expires=900
  &X-Amz-SignedHeaders=content-type;host
  &X-Amz-Signature=5f2a...c91e
```

~~~dg-seq Direct upload: the API checks permission and signs a URL; the bytes go straight from the client to object storage, and an event closes the loop.
Client > API : POST /uploads (name, size, type)
API > API : auth, quota, create record (pending)
API -> Client : presigned PUT URL, expires 15 min
Client > S3 : PUT bytes to presigned URL
S3 -> Client : 200 OK + ETag
S3 > Queue : event: object created
Queue > Worker : validate, scan, start processing
Worker > API : record status = uploaded
~~~

### The rules that matter

- **Permissions come from the signer.** The URL carries the permissions of whoever created it. The signing identity must itself be allowed to do the operation (for example `s3:PutObject` on that bucket prefix). Give the signing service a narrow role that can only write to the upload prefix.
- **Expiry limits.** With Signature Version 4 a presigned URL can last at most **7 days** when signed with long-term IAM user keys. The console allows 1 minute to 12 hours. And it can never outlive the credentials used: a URL signed by a role session or EC2 instance role dies when those temporary credentials expire, which may be sooner than the expiry you asked for.
- **Expiry is checked when the request starts.** An upload or download in progress finishes even if it passes the expiry; a request started after expiry fails.
- **It is a bearer token.** Anyone who has the URL can use it until it expires. Keep expiries short (minutes for uploads), never log full presigned URLs, and send them only over HTTPS.
- **Extra restrictions.** Bucket policies can reject presigned requests whose signature is older than N minutes (the `s3:signatureAge` condition), or that come from outside certain IP ranges or VPC endpoints.
- **Integrity.** You can require a checksum header (SHA-256, CRC32C, CRC64NVME and others) as part of the signature, so S3 rejects an upload whose bytes do not match.
- **Unique keys.** A `PUT` to an existing key **replaces** the object. Generate the key on the server (a UUID or the record ID), never from the user's file name, or one user could overwrite another's file.

### Controlling size and type

A plain presigned `PUT` does not limit size unless you sign the `Content-Length` header. Two ways to enforce limits:

- **Sign the exact size and type**: the client declares them up front, your API checks they are acceptable, and the signature covers `Content-Length` and `Content-Type`, so the upload must match.
- **Presigned POST**: an HTML-form style upload with a signed **policy document** that can say, for example, `content-length-range` between 1 byte and 50 MB, and key must start with `raw/u42/`. S3 enforces each condition.

Even then, never **trust** the declared type: a file named `cat.jpg` with `Content-Type: image/jpeg` may be anything. Check the real content after upload (lesson 13.02 covers image validation).

### Browser uploads need CORS

If a web page uploads directly to the bucket, the browser makes a cross-origin request. The bucket must have a **CORS** (cross-origin resource sharing) rule allowing `PUT` from your site's origin and exposing the `ETag` header, or the browser blocks the request. This is the most common reason a first direct-upload implementation "works in curl but not in the browser".

## Keeping the database in step

Once bytes bypass your servers, your **database** (which holds the file's metadata: owner, name, size, status) and the **bucket** (which holds the bytes) can disagree. The client could get a URL and never upload; it could upload and never tell you; it could upload something different from what it declared. The standard pattern makes the database the record of intent and lets storage events confirm it.

1. Create the metadata record **first**, with status `pending` and the object key you chose.
2. The client uploads with the presigned URL.
3. Object storage emits an **event notification** (S3 can send to a queue, a topic or a function) when the object is created. A handler checks the object exists with the expected size and type, then marks the record `uploaded` and starts processing (thumbnails, virus scan, transcoding).
4. A **reconciliation job** handles the leftovers: `pending` records older than the URL expiry plus a margin are marked `expired`, and a **lifecycle rule** deletes objects in the upload prefix that never got a matching record.

Do not rely only on the client calling "I'm done": phones lose connectivity right after finishing, and malicious clients lie. The storage event is the trustworthy signal. Many apps still accept a client "complete" call as a fast path, and let the event confirm it.

~~~dg-state The lifecycle of an upload record. The API creates it; storage events and background jobs move it forward or clean it up.
[*] -> Pending
Pending -> Uploaded : object-created event, size matches
Pending -> Expired : no object after URL expiry
Uploaded -> Processing : scan and transcode start
Uploaded -> Rejected : wrong type or malware
Processing -> *Ready : all variants done
Processing -> Failed : processing error, retry or alert
~~~

### Quarantine first

A good habit is to upload into a **quarantine prefix** (say `raw/`) that nothing serves to users. Only after validation and scanning does the worker write processed outputs to a public or CDN-served prefix (`media/`). That way, a malicious file never reaches other users even briefly.

## Downloads use the same idea

Downloads of private files work the same way in reverse: your API checks permission and returns a presigned `GET` URL (or redirects to one), and the client downloads straight from storage. Better still, serve through a **CDN with its own signed URLs or cookies** (lesson 13.07), so popular files come from edge caches near users and the bucket is not hit for every request.

<div class="ex">
<div class="lab">Worked example</div>

**a 1 GB video upload, step by step**

1. The app calls `POST /videos` with `{size: 1 GB, type: video/mp4}`. The API checks the user is logged in and has 1 GB of quota left.

2. It inserts `{id: v77, key: raw/9f1c.mp4, status: pending}` and returns a presigned URL. For a file this large it actually starts a **multipart upload** and returns one presigned URL per 16 MB part, 64 in total, valid for 30 minutes (lesson 13.04).

3. The phone uploads parts straight to S3, four at a time. The API servers never touch the gigabyte.

4. When the upload completes, S3 emits an object-created event. A worker confirms the size is 1 GB, checks the file really is an MP4 container, marks v77 `uploaded` and starts the transcoding pipeline (lesson 13.05).

5. If the user had closed the app at 30% and never come back, the reconciliation job would mark v77 `expired` the next hour, and the lifecycle rule would abort the incomplete multipart upload after a day so its parts stop costing money.

</div>

## Common mistakes

- **User-controlled object keys.** Letting the client pick `uploads/report.pdf` invites overwrites and path tricks. Generate keys on the server.
- **Long expiries.** A presigned URL valid for 7 days, pasted into a chat, is a 7-day public link.
- **No size limit.** Without a signed length or a POST policy, a client can upload 5 GB where you expected 5 MB, and you pay to store it.
- **Trusting the client's "done".** The record says uploaded; the object is not there. Use storage events.
- **Serving straight from the upload bucket.** Unvalidated files become reachable by others. Quarantine, validate, then publish.
- **Forgetting CORS** for browser uploads.

## In an interview

When a design involves user uploads larger than a few megabytes, say early that bytes go **directly to object storage with presigned URLs**, and that the API only authorises and records metadata. Then cover the follow-ups interviewers expect: how the database learns the upload finished (storage events plus reconciliation), how you limit size and type, how you handle large files (multipart, lesson 13.04), and how downloads are served (CDN with signed URLs). That shows you understand both the data path and the control path.

<div class="take">
<div class="lab">Key takeaway</div>

let clients upload and download large files directly to object storage using short-lived presigned URLs that your backend signs after checking permissions; the API handles only metadata, so bandwidth and slow connections never touch your servers. Create a pending record first, let an object-created event confirm the upload, and clean up with reconciliation and lifecycle rules. Treat presigned URLs as bearer tokens: short expiry, server-generated keys, signed size and checksums, and a quarantine prefix until the content is validated.

</div>

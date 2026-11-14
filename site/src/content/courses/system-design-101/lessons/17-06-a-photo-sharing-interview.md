---
id: "17.06"
course: "system-design-101"
module: "17"
slug: "a-photo-sharing-interview"
title: "A photo-sharing interview"
keyIdeas: ["Photo sharing design (Instagram-style), common interview question.", "Upload: client gets a presigned URL, uploads the photo to object storage; a processing job creates sizes; the post record (user, caption, media keys) is saved.", "Feed: fanout on write into followers' feed caches (pull for celebrities), ranked at read time.", "Media served via CDN; metadata in sharded Postgres (Instagram's early stack) keyed by user.", "Likes and comments counts are denormalized and cached.", "Instagram's early lessons: keep it simple, use proven tools (Django, Postgres, Redis, S3), and shard with logical shards from day one."]
question: "Walk through a photo upload from phone to followers' feeds."
answer: "The app asks the posts service for a presigned URL and uploads the photo directly to object storage, so API servers never handle the bytes. An object-created event triggers media workers that validate the file and generate thumbnails and sized WebP or AVIF variants, stored under predictable keys and served via the CDN. The app finalises this post with a caption; the posts service stores it once with a time-sortable ID and publishes a post-created event. The fan-out service reads the poster's followers in batches, and workers push this post ID onto each follower's feed list in Redis (trimmed to a few hundred), unless the poster is a celebrity, whose posts are merged in at read time. When a follower opens the app, the feed service reads their list, merges celebrity posts, hydrates the posts and authors from caches, and returns CDN URLs that the phone loads from the nearest edge."
sources: [{"title": "Hello Interview: Instagram breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/instagram", "note": "Upload, feed, media."}, {"title": "Instagram: What powers Instagram", "url": "https://instagram-engineering.com/what-powers-instagram-hundreds-of-instances-dozens-of-technologies-adf2e22da2ad", "note": "The real early stack."}]
tags: ["Design"]
---

## The interview prompt

"Design Instagram." This lesson is a complete walkthrough of the kind you would give in an interview, following one photo from the moment the user taps "Share" to the moment it appears in a follower's feed. It assembles parts from earlier lessons: presigned uploads (13.01), image variants and CDNs (13.02), the feed design (16.07 and 11.05), and ID generation (04.08). The aim is to show how the pieces connect and where each decision comes from.

### Requirements

- Users **post photos and videos** with a caption.
- Users **follow** other users.
- Users see a **feed** of posts from people they follow, newest first.
- Out of scope: likes, comments, search, stories, direct messages, ranking.

### Non-functional requirements

- **500 million daily users**, **100 million posts a day**.
- Feed loads in under **500 ms**.
- Photos up to 8 MB, videos up to 4 GB.
- A new post may take a few seconds to reach followers' feeds (eventual consistency is fine); the poster should see their own post immediately.
- Media must never be lost once the user sees "posted".

## Back-of-the-envelope numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing the system**

Posts: 100 million a day is about 1,200 per second on average, perhaps 3,000 at peak.

Media storage: if the average upload is 2 MB after on-device compression, and processing creates variants adding about 50%, each post costs about 3 MB. 100 million x 3 MB = 300 TB a day, about 110 PB a year. Object storage is the only sensible home for this.

Feed reads: 500 million users open the app 10 times a day: 5 billion feed loads, about 58,000 per second on average and 150,000 at peak. Each load returns about 10 posts, so the CDN serves around 1.5 million image requests per second at peak.

Fan-out: with 200 followers per author on average, 1,200 posts per second become 240,000 feed-list inserts per second.

</div>

Three conclusions follow: media bytes must never pass through API servers, the feed must be precomputed because reads dwarf writes, and images must come from a CDN.

## Data model and API

| Entity | Key fields | Store |
| --- | --- | --- |
| User | user_id, name, avatar_url | relational (sharded) |
| Follow | follower_id, followee_id | relational or graph store, indexed both ways |
| Post | post_id, author_id, caption, media_ids, status, created_at | relational or wide-column, by post_id |
| Media | media_id, post_id, variants, status | with posts |
| Feed list | user_id to list of post_ids | Redis |

```
POST /posts/uploads {type, size}       -> media_id, presigned_url
POST /posts {media_id, caption}        -> post_id
POST /follows {followee_id}            -> 204
GET  /feed?cursor=...                  -> posts with CDN URLs + next cursor
```

**Post IDs** should be unique across shards and roughly time-sortable, so feeds can be sorted by ID. Instagram's own design used 64-bit IDs made of a millisecond timestamp, a logical shard ID and a per-shard sequence (lesson 04.08), generated inside PostgreSQL.

## Step 1: upload the photo

1. The app asks the **post service** to start a post: `POST /posts/uploads {type: photo, size}`. The service checks the user is allowed to post and the size is within limits, then creates a media record with status `pending` and a **media ID**.
2. It returns a **presigned URL** for `raw/{media_id}.jpg` in object storage, valid for perhaps 15 minutes and only for that one key.
3. The phone uploads the file **straight to object storage**. Phones usually resize and compress first, so an 8 MB photo might upload as 1 to 2 MB. Large videos use multipart upload so they can resume (lesson 13.04).

The API servers never handle image bytes. At 3,000 uploads per second of 2 MB each, that is 6 GB per second of traffic you do not have to scale your application tier for.

## Step 2: process the media

1. Object storage emits an "object created" event onto a queue.
2. **Media workers** validate the file (is it really an image, is it within limits), strip location metadata if the user's settings require it, run safety checks, and generate **variants**: a small thumbnail for grids, a few widths (for example 320, 640 and 1080 pixels) for different screens, in modern formats such as WebP or AVIF, which are 25 to 50% smaller than JPEG at the same quality. Videos go through the transcoding pipeline instead (lesson 17.02).
3. Variants are written to object storage under predictable keys (`media/{media_id}/1080w.webp`), and served through a **CDN**.
4. The media record becomes `ready` with its variant keys.

If a worker crashes, the queue redelivers the event and another worker repeats the job. Because the variant keys are deterministic, repeating the work simply overwrites the same objects: the job is idempotent.

~~~dg-state A media item's lifecycle. Followers only ever see posts whose media is ready.
[*] -> Pending
Pending -> Uploaded : object created
Uploaded -> Processing : worker picks up
Processing -> *Ready : variants written
Processing -> Failed : invalid file
Pending -> Expired : no upload in 24 h
~~~

## Step 3: create a post and fan out

1. The app finalises this post with a caption: `POST /posts {media_id, caption}`. The posts service saves it **once** in the **posts store**, keyed by post ID. If the media is not ready yet, this post is saved with status `processing` and published when the media worker reports `ready`.
2. The posts service publishes an event `post_created {post_id, author_id}`, using a transactional outbox so the event cannot be lost if the service crashes after saving.
3. The **fan-out service** checks the poster's follower count:
   - **Normal author** (say under 100,000 followers): read follower IDs from the social graph in pages of a few thousand and enqueue batches. **Fan-out workers** push this post ID onto the front of each follower's **feed list** in Redis (`LPUSH feed:{user} post_id` then `LTRIM feed:{user} 0 499`), keeping the most recent few hundred.
   - **Celebrity**: no fan-out. Their posts are pulled at read time (the hybrid from lesson 16.07).
4. The poster's own feed and profile get it immediately, so they see it at once even before fan-out finishes (read-your-own-writes).

~~~dg-flow From the phone to followers' feeds: bytes, processing, post record, then fan-out of IDs.
[Upload] Phone -(presigned PUT)-> Object storage -(object created)-> *Media workers
[Post] Phone -(caption, media id)-> Post service -(save once)-> Posts store
[Fan-out] Post service -(post_created)-> Queue -(followers in batches)-> Feed lists in Redis
~~~

<div class="ex">
<div class="lab">Worked example</div>

**fan-out for one post**

An author with 2,000 followers posts. The fan-out service reads followers in 2 pages of 1,000 and enqueues 2 messages. Two workers each pipeline 1,000 LPUSH plus LTRIM commands to Redis, which takes about 10 ms per batch.

Within a second of this post being saved, all 2,000 followers' feed lists contain its ID. A follower who opens the app 3 seconds later sees it at the top.

A celebrity with 50 million followers posts. Nothing is fanned out. Each of the followers who opens the app in the next hour merges that celebrity's recent posts at read time, a single read of a hot key replicated across cache nodes.

</div>

## Step 4: a follower opens the app

1. The **feed service** reads the follower's feed list (post IDs) from Redis, merges in recent posts from celebrities they follow, and takes the newest page of about 10.
2. It **hydrates** the IDs in parallel: post details from a post cache, author names and avatars from a user cache. Posts that were deleted or whose author blocked this follower are dropped here.
3. The response contains **CDN URLs** for the photo variants sized for this device (a 390-pixel-wide phone gets the 640w or 1080w variant, not the original).
4. The phone loads images directly from the nearest CDN edge, usually in tens of milliseconds, and prefetches the next page as the user scrolls.

~~~dg-seq A follower's feed request: IDs from the precomputed list, bytes from the CDN.
Follower > Feed svc : GET /feed
Feed svc > Redis : feed list for follower (post IDs)
Feed svc > Posts : recent posts of followed celebrities
Feed svc > Cache : hydrate posts + authors
Feed svc -> Follower : page of posts with CDN image URLs
Follower > CDN : GET photo 1080w.webp
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the latency budget for one feed load**

Redis read of the feed list: 1 ms. Celebrity posts for the 4 celebrities followed: 2 ms, in parallel. Hydration multi-gets to post and user caches: 3 ms. Service overhead and serialisation: 10 ms. Network to the phone: 50 to 150 ms.

Total time to the JSON response: well under 200 ms, comfortably inside 500 ms. Images then load from the CDN in parallel; the first photo is visible a few tens of milliseconds later.

</div>

## Why each choice

- **Presigned upload**: 100 million posts a day of photos and videos would otherwise flow through API servers.
- **Async processing**: the user sees "posted" quickly; thumbnails appear within seconds; a slow image library cannot slow down the API.
- **Fan-out on write for most authors**: reads outnumber writes about 50 to 1, so precomputing feeds makes the 500 ms target easy. Feed lists hold IDs only, so they are small (about 500 IDs x 8 bytes = 4 KB per user).
- **Pull for celebrities**: pushing one post to 300 million feeds would take minutes and enormous write capacity.
- **CDN for media**: the bytes are by far the heaviest part of the feed; edges near users make images load fast and keep origin egress low.

~~~dg-compare Two ways to get the photo bytes into storage. Presigned uploads keep API servers out of the data path.
[Through the API]
Phone -(8 MB)-> API servers -> Object storage
- API must scale for 6 GB/s
- Slow uploads hold connections
[*Presigned URL]
Phone -(8 MB, signed PUT)-> Object storage
+ API handles only small JSON
+ Storage scales on its own
- URL must be scoped and short-lived
~~~

## Failure handling

- **Upload succeeds but no post is ever created** (the app crashed): the media stays `pending` or `ready` without a post; a cleanup job deletes orphaned media after 24 hours.
- **Media processing fails**: retried a few times, then marked failed; the user is asked to try again.
- **Fan-out worker crash**: the queue redelivers the batch; `LPUSH` of an ID already present would duplicate it, so the feed service deduplicates IDs when reading (or workers check the head of the list first).
- **Redis feed node lost**: rebuild affected users' feeds on demand by pulling recent posts from the accounts they follow.
- **Post deleted**: the ID remains in feed lists, but hydration finds this post gone and drops it.

## Instagram's early lessons

Early Instagram shows how far this simple shape goes. With a handful of engineers and millions of users, the team's principles were **keep it very simple**, **do not reinvent the wheel**, and **use proven technologies**. The stack: Django on Amazon EC2 behind load balancers and nginx; **PostgreSQL** for users and photo metadata, sharded, with streaming replicas; photos stored on **S3** and served through **CloudFront**; **Redis** for the feed and sessions; **memcached** for caching; a task queue for fan-out and other async work; and push notifications through APNs. That is the same shape as this lesson: object storage plus CDN for bytes, a relational store for metadata, Redis feeds and async workers. Lesson 18.01 looks at it more closely.

## In an interview

Follow the photo: upload, processing, post, fan-out, read. Give the numbers that justify each step (6 GB per second of uploads, 150,000 feed reads per second). Name the celebrity problem before being asked. Ask: are videos as common as photos, is the feed ranked, and how fast must followers see a new post?

<div class="take">
<div class="lab">Key takeaway</div>

upload the photo directly to object storage with a presigned URL, process variants asynchronously and serve them from a CDN, store each post once, then fan out only its ID into followers' Redis feed lists, skipping celebrities, whose posts are merged at read time. Reading a feed is a list lookup, hydration from caches, and image loads from the CDN.

</div>

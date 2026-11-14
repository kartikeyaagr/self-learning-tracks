---
id: "13.02"
course: "system-design-101"
module: "13"
slug: "image-cdn-and-resizing"
title: "Image CDN and resizing"
keyIdeas: ["Images are served in many sizes (thumbnails, mobile, retina). Generate variants either on upload or on demand.", "**On upload**: predictable, fast reads, but you store variants nobody may request, and adding a new size means reprocessing everything.", "**On demand**: a resizing service (imgproxy, Thumbor, Cloudflare Images) creates the variant when first requested; the CDN caches it after that. Flexible, but the first request is slow and URLs must be signed or limited to prevent abuse.", "Serve modern formats (WebP, AVIF) based on the browser's Accept header.", "A CDN caches content at edge locations close to users, cutting latency and origin load."]
question: "Resize on upload or resize on request? Tradeoffs?"
answer: "Resizing on upload makes every variant a ready static file with no request-time work and catches bad images early, but you must know the sizes up front, adding a size or format means reprocessing every image ever uploaded, and you store and compute variants that may never be viewed. Resizing on request generates any size or format the first time it is asked for and lets the CDN cache it for a year, so storage and compute follow actual demand and new layouts need no backfill; the costs are a one-time delay on the first view of each variant (and cold-cache storms after a redesign, softened by an origin shield or pre-warming) and the need for signed URLs, allowed sizes and decode limits to prevent abuse. Many systems pre-generate the common sizes and serve the rest on demand."
sources: [{"title": "Cloudflare Learning: What is a CDN?", "url": "https://www.cloudflare.com/learning/cdn/what-is-a-cdn/", "note": "Edge caching basics."}, {"title": "imgproxy docs", "url": "https://docs.imgproxy.net/", "note": "On-the-fly resizing service."}]
---

## What a CDN does

A **content delivery network** (CDN) is a geographically distributed group of servers that **caches content close to end users**. Instead of every browser in Paris, Tokyo and São Paulo fetching an image from your servers in Virginia, each fetches it from a nearby CDN server that already has a copy. CDNs deliver HTML, JavaScript, stylesheets, images and video, and they bring four benefits:

- **Lower latency.** Content is served from a nearby **point of presence** (PoP), a data centre the CDN runs in or near a city, often at an **internet exchange point** where many networks interconnect. A round trip from Paris to Virginia is about 80 ms; to a Paris PoP, a few milliseconds. For an image that needs several round trips (TCP and TLS handshakes, then the request), the saving is hundreds of milliseconds.
- **Lower bandwidth costs.** Cached responses never reach your origin, so you pay for much less origin egress.
- **Availability.** The CDN absorbs traffic spikes and can keep serving cached content when your origin is slow or down.
- **Security.** TLS terminates at the edge, and large CDNs absorb DDoS attacks that would flatten a single origin.

A CDN **does not host** your content: it caches it. You still need an **origin**, the authoritative source it fetches from on a miss: an object storage bucket, a web server, or (for images) an image-processing service.

## How the tiers fit together

A request for an image passes through several caches, each larger and further away than the last.

~~~dg-layers The caching tiers an image request passes through, from nearest and smallest to furthest and authoritative.
Browser cache | on the device, reused across pages, controlled by Cache-Control
*Edge PoP | hundreds of cities, milliseconds away, most hits happen here
Origin shield | one regional cache in front of the origin, collapses misses from all PoPs
Image service | resizes and encodes on a miss, for example imgproxy on libvips
Object storage | the uploaded originals, the source of truth
~~~

The **origin shield** (also called a tier-2 or mid-tier cache) deserves a note. Without it, a new image requested in 300 PoPs causes 300 separate misses to your origin. With it, all PoPs miss to the shield, and the shield misses to the origin once. For an image service that does expensive work per miss, a shield can cut origin load by one or two orders of magnitude.

Cache behaviour is controlled by HTTP headers. `Cache-Control: public, max-age=31536000, immutable` tells browsers and CDNs to keep a response for a year and never revalidate it, which is right for any URL whose content never changes. The CDN's **cache key** is normally the URL (plus any headers listed in `Vary`); two URLs that differ in any way are two cache entries.

## Images need many variants

A single uploaded photo is displayed in many places: a 64-pixel avatar, a 320-pixel feed thumbnail, a 1080-pixel phone view, a 2× version for high-density ("retina") screens, and in modern formats such as **WebP** or **AVIF** for browsers that support them, because they are 25% to 50% smaller than JPEG at similar quality. Serving the 8 MB original everywhere would waste bandwidth and make pages slow.

~~~dg-bars Bytes for one product photo in different variants. Serving the original to a phone is about 90 times more data than the right variant.
Original JPEG 6000 x 4000 | 8000 | 8 MB
1600 px JPEG | 420 | 420 KB
800 px JPEG | 150 | 150 KB
800 px WebP | 105 | 105 KB
*800 px AVIF | 90 | 90 KB
320 px WebP | 22 | 22 KB
~~~

So you need variants. The design question is **when** to make them.

## Option 1: resize on upload

An upload pipeline (triggered by the object-created event from lesson 13.01) generates a fixed set of sizes and formats as soon as the image arrives, and stores them all in object storage. Serving is then trivial: each variant is a static file the CDN caches.

- **Reads are simple and fast.** No work at request time, no first-view delay.
- **Failures are visible early.** A corrupt image fails in the pipeline, not in front of a user.
- **You must know the sizes in advance.** A redesign that needs a 480-pixel version means **reprocessing every image ever uploaded**: a backfill job over perhaps billions of objects.
- **Storage multiplies.** Five sizes in three formats is 15 files per image, including variants nobody ever requests. Most photos on most platforms are viewed rarely, so most of that storage and compute is wasted.

## Option 2: resize on request

An **image server** generates a variant the first time it is requested, and the CDN caches the result. The URL itself describes the variant. **imgproxy** is a typical example: a standalone HTTP server that receives a source image location and processing options **in the URL** (resize, crop, rotate, quality, watermark, output format), fetches the original, processes it with the **libvips** library (fast and memory-efficient because it streams the image in small regions instead of decoding it all at once), and returns the result.

`https://img.example.com/<signature>/rs:fit:800:0/q:75/f:avif/plain/s3://products/p91.jpg`

Three practices make this safe in production.

- **Put it behind a CDN.** The image server is meant to sit behind a CDN or caching proxy, so each variant is generated once per cache and then served from cache. On its own, it would redo the work on every request.
- **Sign the URLs.** Without a signature, anyone could ask your server to resize arbitrary images or generate millions of random sizes, burning your CPU and filling your cache (a denial-of-service attack). The URL includes an HMAC of the options and source, computed by your application with a secret key; the server rejects anything unsigned. Header-based authorisation can additionally stop clients from bypassing the CDN.
- **Defend against image bombs.** An **image bomb** (decompression bomb) is a tiny file that decodes into an enormous image, say a 50 KB PNG declaring 50,000 × 50,000 pixels, which would need 10 GB of memory. The server checks the real image type and dimensions from the header **before** fully downloading and decoding, and rejects anything over a pixel limit. It can also restrict allowed sources, sizes and options.

~~~dg-seq Resize on request: the first request for a variant is processed once and cached; later requests are pure edge hits.
Browser > CDN edge : GET /sig/rs:fit:800/f:avif/p91.jpg
CDN edge > Origin shield : miss, forward
Origin shield > Image service : miss, forward
Image service > Object storage : GET original p91.jpg (8 MB)
Object storage -> Image service : original bytes
Image service > Image service : check size, decode, resize, encode AVIF
Image service -> Origin shield : 90 KB, max-age 1 year
Origin shield -> CDN edge : cached at shield
CDN edge -> Browser : cached at edge, about 400 ms first time
~~~

## Comparing the two

| | Resize on upload | Resize on request |
| --- | --- | --- |
| New sizes or formats | Reprocess every image | Just use a new URL |
| Storage | All variants for all images | Originals plus cached hot variants |
| First-view latency | None | One-time processing delay per variant per cache |
| Compute | Paid for every image, viewed or not | Paid only for variants actually requested |
| Main risk | Pipeline backlog after a big upload burst | Abuse through arbitrary sizes, cold-cache storms |

~~~dg-compare When variants are made. Most production systems combine the two: pre-generate the common sizes, generate the rest on demand.
[Resize on upload]
Upload -> Pipeline makes 15 variants -> Storage -> CDN
+ No request-time work
+ Errors caught at upload
- Redesign means a full backfill
- Stores variants nobody views
[*Resize on request]
Request -> CDN miss -> Image service -> CDN cache
+ Any size or format via the URL
+ Compute follows real demand
- First view of a variant is slower
- Needs signed URLs and size limits
~~~

Common practice combines them: generate the one or two sizes you know are always needed (a thumbnail and a standard view) at upload, so the first view of a new image is fast and broken uploads are caught early, and serve everything else on demand, from a **limited set of allowed sizes** (for example 160, 320, 640, 960, 1280, 1920), behind a CDN with long cache lifetimes. Rounding requested sizes up to the nearest allowed width keeps the number of variants, and therefore cache misses, small.

<div class="ex">
<div class="lab">Worked example</div>

**a product photo**

A seller uploads a 6000 × 4000 JPEG (8 MB) to object storage. The product page asks for `/img/<sig>/rs:fit:800/f:avif/products/p91.jpg`.

The CDN edge misses, the shield misses, and imgproxy fetches the original (about 80 ms from storage in the same region), decodes and resizes it (about 150 ms on libvips) and encodes AVIF (about 100 ms; AVIF encoding is slow). It returns 90 KB with `Cache-Control: public, max-age=31536000, immutable`. The first viewer waits roughly 400 ms longer than a cache hit.

The next million views come from CDN edges in a few milliseconds each. When the design changes to 960 pixels, the page simply starts requesting `rs:fit:960`, and those variants are generated as they are requested. No backfill job.

</div>

## Format negotiation without fragmenting the cache

Browsers advertise supported formats in the `Accept` request header (`image/avif,image/webp,*/*`). Two approaches:

- **Put the format in the URL**, chosen by your HTML (the `<picture>` element lets the browser pick from AVIF, WebP and JPEG sources). Each URL is a clean cache entry.
- **Negotiate on the server** with `Vary: Accept`. This is convenient but dangerous: raw `Accept` strings differ between browser versions, so the cache can fill with near-duplicate entries. CDNs that support it normalise `Accept` into a few buckets (AVIF, WebP, other) before using it in the cache key.

## Cache arithmetic and cold starts

Resize on request shifts cost to the first request of each variant, **per cache**. With an origin shield, that is roughly once per variant overall; without one, once per variant per PoP.

<div class="ex">
<div class="lab">Worked example</div>

**a cold-cache storm**

A marketplace launches a redesign that uses a new 960-pixel size. 2 million product images are viewed in the first hour, across 200 PoPs.

Without a shield: each image is first requested in, say, 30 different PoPs during that hour, so the image service sees about 60 million resize requests. At 300 ms of CPU each, that is 5,000 CPU-hours in one hour: about 5,000 cores. The service falls over and images break site-wide.

With a shield: about 2 million resize requests (one per image), 170 CPU-hours, so about 170 cores for that hour, and much less after as the cache warms. Better still, pre-warm the most viewed 100,000 products before launch by requesting their new URLs from a script.

</div>

## Pitfalls

- **Unsigned or unlimited URLs.** Anyone can make your image service do arbitrary work. Sign URLs and allow only a fixed set of sizes.
- **Short cache lifetimes.** Variants of an immutable original never change; cache them for a year. If the original can change, put a version in the URL (lesson 13.03) instead of shortening the lifetime.
- **No origin shield.** Every PoP's first miss hits your image service.
- **Decoding untrusted images naively.** Without dimension checks, one image bomb can exhaust memory on every worker that touches it.
- **Upscaling.** Requests for 1920 pixels from a 640-pixel original should return the original size, not an enlarged blurry copy that wastes bytes.
- **Stripping nothing.** Originals from phones carry EXIF metadata, including GPS coordinates. Strip metadata from public variants for privacy.

## In an interview

Draw the tiers: browser, CDN edge, shield, image service, object storage. Explain both options, and recommend the hybrid: a couple of standard sizes on upload, everything else on request from an allowed list, with signed URLs and long-lived immutable caching. Mention image bombs and the cold-start storm, and how the shield and pre-warming handle it. That covers latency, cost, flexibility and abuse, which is what the question is testing.

<div class="take">
<div class="lab">Key takeaway</div>

a CDN caches content at edges near users, cutting latency and origin load, but needs an origin behind it, ideally protected by an origin shield. Resizing on upload gives predictable, pre-made variants but is rigid and stores everything; resizing on request (an image service such as imgproxy behind a CDN, with signed URLs, allowed sizes and image-bomb checks) is flexible and pays only for variants actually used, at the cost of a one-time delay per variant. Most systems pre-generate the common sizes and serve the rest on demand.

</div>

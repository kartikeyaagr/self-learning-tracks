---
id: "13.03"
course: "system-design-101"
module: "13"
slug: "an-avatar-service"
title: "An avatar service"
keyIdeas: ["An avatar service maps a user to a profile image URL, with a default when none exists.", "Gravatar uses a hash of the normalized email as the key (`/avatar/<hash>?s=200`), so any site can show the avatar without an API call or knowing the user.", "Store originals in object storage, resize to standard sizes, and serve through a CDN with long cache lifetimes.", "When a user changes their avatar, change the URL (add a version) instead of waiting for caches to expire.", "Default images (initials, identicons) can be generated deterministically from the hash."]
question: "How does Gravatar serve an avatar from just an email, and how would you cache it?"
answer: "The site trims and lower-cases the user's email, hashes it with SHA-256 and requests gravatar.com/avatar/<hash> with a size (1 to 2048, default 80) and a default-image option; Gravatar looks up the account by hash and returns its image resized, or, if there is none or its rating is too high, a 404, a silhouette or a deterministic generated image such as an identicon, so no lookup call is ever needed. To cache it, put a CDN in front with a small set of allowed sizes and normalised parameters; fixed URLs need a modest expiry (minutes to an hour) or purges, so it is better, in your own service, to use versioned image URLs cached for a year as immutable, changing the URL whenever the avatar changes, and to cache \"no avatar\" responses briefly too."
sources: [{"title": "Gravatar docs", "url": "https://docs.gravatar.com/", "note": "Hash-based avatar URLs."}]
---

## The problem an avatar service solves

Almost every product shows small pictures of people: next to comments, in chat lists, on profile pages, in email clients. An **avatar service** stores those images and serves them. It looks trivial, but it has an unusual load profile: avatars are **tiny** (a few kilobytes), **extremely numerous** (one per user, in several sizes), **read constantly** (a comment thread with 200 comments shows 200 avatars) and **changed rarely** (most people change their picture a few times in their life). That profile makes it almost entirely a **caching** problem.

This lesson looks at how Gravatar, a public avatar service, lets any website show a user's picture knowing only their email address, then at how to cache avatars well, and finally at designing your own.

## How Gravatar works

Gravatar ("globally recognised avatar") lets a user upload a picture once, linked to their email address, and have it appear on any site that supports it. The site never calls an API to find the avatar. It **builds the URL itself**:

1. Take the user's email address and **normalise** it: trim leading and trailing spaces and convert it to lower case. `  Alice@Example.com ` becomes `alice@example.com`. Without normalisation, the same person would get different hashes on different sites.
2. Compute its **SHA-256 hash** (older integrations used MD5, which is still accepted). The result is 64 hexadecimal characters.
3. Request `https://gravatar.com/avatar/HASH`, adding query parameters for size and fallback behaviour.

The hash identifies the account without putting the email address itself into page HTML, where it could be scraped. Gravatar keeps a map from hash to account and serves the account's image.

~~~dg-flow Building a Gravatar URL needs no lookup call: the key is derived from data the site already has.
Alice@Example.com -(trim, lowercase)-> alice@example.com -(SHA-256)-> *Hash 3f2a... -(add s=96, d=identicon)-> Avatar URL
~~~

### The parameters

- `s` (size): one number, since avatars are square; **1 to 2048** pixels, default **80**. The service resizes the stored image to that size.
- `d` (default): what to return if the hash has no avatar. `404` returns an HTTP 404 so the site can handle it; `mp` ("mystery person") returns a generic grey silhouette; `identicon` (a geometric pattern), `monsterid`, `wavatar`, `retro` (8-bit style) and `robohash` return generated images unique to the hash; `blank` returns a transparent image; `initials` draws letters from the user's name; or you can pass a custom image URL of your own.
- `f=y`: force the default image even if an avatar exists (useful for testing).
- `r` (rating): `g` (default, suitable for all audiences), `pg`, `r` or `x`. Users rate their own image; if it is rated above what the site asks for, the default is returned instead.

~~~dg-seq One Gravatar request. The default is resolved on the server, so the browser always receives an image.
Browser > CDN edge : GET /avatar/3f2a...?s=96&d=identicon
CDN edge > Gravatar origin : miss, forward
Gravatar origin > Gravatar origin : look up hash, check rating
Gravatar origin > Gravatar origin : no avatar, generate identicon from hash
Gravatar origin -> CDN edge : 96 x 96 PNG, cache for minutes
CDN edge -> Browser : image
Browser > CDN edge : same URL on next page
CDN edge -> Browser : cache hit
~~~

### The design ideas worth stealing

Two ideas stand out, and both apply well beyond avatars.

- **Derived keys.** The lookup key is computed from data the caller already has, so there is no extra round trip to ask "what is this user's avatar ID?". Any page can render avatar tags immediately.
- **Server-side defaults.** The fallback is resolved on the server. The client always gets an image, so there is no broken-image icon and no client logic for "user has no avatar". A generated identicon is also **deterministic**: the same hash always gives the same pattern, so it is cacheable like a real image and users stay visually distinguishable.

### A privacy weakness

A hash hides the address only from someone who does not already have a list of candidate addresses. Anyone with a list of email addresses (from a leak, or from guessing common patterns like `firstname.lastname@gmail.com`) can hash them all and match the hashes they see on a site, linking accounts across sites. Hashes of email addresses have been reversed in bulk this way. If you design your own service, use a **random, application-generated ID** in avatar URLs instead of anything derived from personal data.

## Caching avatars

### Fixed URL or versioned URL?

This is the central design decision.

A **fixed URL**, like `/avatar/HASH?s=96`, always refers to "this user's current avatar". When the user changes their picture, the content behind the URL changes. Caches must therefore **expire** the old one: you set a `max-age` of minutes to an hour and accept that the new picture takes that long to appear everywhere, or you **purge** the URL from the CDN when it changes (purges take seconds to propagate and cannot reach browser caches).

A **versioned URL**, like `/avatars/u42/v7-96.jpg`, refers to one specific image that never changes. The user's profile record stores the current version (`avatar_version = 7`). When the user uploads a new picture, the app writes v8 and updates the profile. Pages rendered after that reference v8. Since v7's content never changes, it can be cached **forever**: `Cache-Control: public, max-age=31536000, immutable`. Nothing needs purging, because nobody asks for v7 any more and it ages out of caches on its own.

~~~dg-compare Fixed avatar URLs must expire to show changes; versioned URLs never change, so they can be cached for a year.
[Fixed URL]
Profile -> /avatar/HASH?s=96 -> CDN, max-age 1 hour
+ Any site can build it from the email
- New picture takes up to an hour to appear
- Every hour each cache revalidates or refetches
[*Versioned URL]
Profile v7 -> /avatars/u42/v7-96.jpg -> CDN, cached 1 year
+ Change is instant: new pages use v8
+ Near 100% hit rate, no purges
- Needs the current version in the profile record
~~~

Gravatar uses fixed URLs because third-party sites only know the email address. Inside your own application you control the page rendering, so versioned URLs are almost always better.

~~~dg-timeline Bob changes his picture at 12:00. With a one-hour fixed URL, viewers see a mix; with versioned URLs, every new page shows the new one.
12:00 | Bob uploads a new picture
12:00 | *Versioned: profile now points at v4, next page load shows it
12:01 | Fixed URL: edge caches still serve the old image
12:30 | Fixed URL: some PoPs refreshed, others not, viewers disagree
13:00 | Fixed URL: all caches expired, everyone sees the new picture
Weeks later | Versioned: v3 copies age out of caches unused
~~~

### Other caching rules

- **Put a CDN in front.** Avatars are the ideal CDN workload: small, numerous, read constantly, identical for every viewer.
- **Limit the sizes.** Support a handful (32, 64, 128, 256, perhaps 512 for profile pages) and round requests up to the nearest. Allowing all 2,048 sizes would fragment the cache into thousands of variants per user, each with a cold first request.
- **Normalise the URL.** Fixed parameter order, lower-case hash, no unused parameters, so equivalent requests share one cache entry. `?s=96&d=mp` and `?d=mp&s=96` are different cache keys unless the CDN normalises them.
- **Negative caching.** "No avatar" (a 404 or a generated default) should also be cached, briefly (say 10 minutes), so pages full of users without pictures do not hammer the origin with the same lookup.
- **Application-side caching.** Store the avatar URL (or the version) on the user object your app already loads. A comments API should return each author's avatar URL alongside their name; it should not make a call to the avatar service per comment.
- **Use `srcset`.** Let the browser choose 1× or 2× density: `<img src=".../v7-48.jpg" srcset=".../v7-96.jpg 2x">`.

<div class="ex">
<div class="lab">Worked example</div>

**a comment thread with 200 comments**

The page includes 200 `<img>` tags. The comments API already returned each author's avatar URL (versioned), so rendering needs no extra lookups. Many commenters repeat, so there are only 60 distinct avatars.

The browser requests 60 images. Twenty are already in its own cache from earlier visits. The other 40 come from the CDN edge, nearly all hits because popular commenters' avatars are requested by thousands of readers. The origin might see one or two requests. Each avatar at 48 × 48 is about 2 to 4 KB in WebP, so the whole set is under 200 KB.

When Bob changes his picture, his profile switches from v3 to v4. New pages reference v4; the first request for v4 at each edge misses once and is cached for a year. Old v3 copies age out unused. No purge was needed.

</div>

## Designing your own avatar service

Putting the pieces from this module together:

- **Upload.** The client gets a presigned URL and uploads the original directly to object storage (lesson 13.01). Limit the size (say 10 MB) and accept only image types.
- **Processing.** An object-created event triggers a worker that validates the image (real type, dimensions, no image bomb), crops it to a square (centred, or using a face-detection crop), strips EXIF metadata, and generates the standard sizes in WebP and JPEG (lesson 13.02). Writing a handful of fixed sizes at upload is the right choice here: the set of sizes is small and stable, and every avatar is viewed.
- **Moderation.** Optionally run the image through an automated classifier before publishing, since avatars appear next to everything a user does.
- **Publish.** Write the variants under `avatars/<user_id>/<version>-<size>.webp`, then update `avatar_version` in the user record. The order matters: write files first, then flip the pointer, so no page ever references a version that does not exist yet.
- **Serving.** A CDN in front of the bucket, with year-long immutable caching. URLs use the user ID (or a random avatar ID), not the email.
- **Defaults.** If a user has no avatar, render initials or an identicon generated from the user ID, either on the client (an SVG) or by a small service whose output is cached like any other image.
- **Deletion.** When a user deletes their account or picture, update the record immediately (new pages stop showing it) and delete the objects. For privacy-sensitive deletions, also purge the CDN so cached copies vanish within minutes rather than ageing out.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the service**

100 million users, 60% with a custom avatar: 60 million originals at about 1 MB each is 60 TB, plus 5 sizes in 2 formats at an average of 10 KB each, about 6 TB more.

Read load: 2 million page views per minute, each showing 20 avatars, is about 670,000 avatar requests per second. With a 99% CDN hit rate (realistic for year-long immutable URLs), the origin sees about 6,700 requests per second, all simple object reads from storage. Writes: perhaps 50,000 avatar changes a day, under 1 per second. Processing them needs a couple of small workers.

</div>

## Pitfalls

- **Fixed URLs with long cache lifetimes.** Users change their picture and see the old one for a day. Version the URL.
- **Arbitrary sizes.** Cache fragmentation and an easy abuse vector. Allow a fixed set.
- **Email hashes in URLs.** Reversible with a dictionary of addresses; use random IDs.
- **Per-item lookups.** Calling the avatar service once per comment turns one page into 200 backend calls. Return URLs with the data.
- **Flipping the pointer before the files exist.** Users briefly see broken images.

## In an interview

Explain Gravatar's trick (a derived key from the normalised, hashed email, with server-side defaults), then pivot to caching, because that is what the question is really about. Recommend versioned, immutable URLs cached for a year, a fixed set of sizes, negative caching, and URLs returned with the data so pages make no extra calls. Mention the privacy issue with email hashes. A quick sizing estimate showing that the CDN absorbs almost all the load lands the point.

<div class="take">
<div class="lab">Key takeaway</div>

Gravatar serves avatars by a SHA-256 hash of the trimmed, lower-cased email, with size, default-image and rating parameters, so any site can build the URL without a lookup and always gets an image back. To cache avatars well, put a CDN in front, allow only a few sizes, cache "no avatar" responses briefly, and prefer versioned image URLs that can be cached for a year, changing the URL (not purging) when the image changes. Use random IDs rather than email hashes in your own service.

</div>

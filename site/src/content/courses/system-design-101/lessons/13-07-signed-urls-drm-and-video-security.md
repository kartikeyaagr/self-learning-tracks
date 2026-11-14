---
id: "13.07"
course: "system-design-101"
module: "13"
slug: "signed-urls-drm-and-video-security"
title: "Signed URLs, DRM, and video security"
keyIdeas: ["**Signed URLs/cookies**: the CDN only serves content when the URL carries a valid signature and unexpired timestamp. This stops casual link sharing and hotlinking.", "Signed URLs don't stop someone who has access from downloading and redistributing the video.", "**DRM** (Widevine, FairPlay, PlayReady) encrypts the content and only releases decryption keys to licensed players in protected environments, so copying is much harder.", "Other layers: token-based auth for manifests, per-session keys, watermarking to trace leaks, and geo-restrictions."]
question: "Signed URLs stop link sharing. What do they not stop that DRM tries to?"
answer: "Signed URLs only control access to the files: who can download them and until when (and optionally from which IP), checked at the CDN edge. They do not stop an authorised user, while the URL or cookie is valid, from saving the downloaded segments, stitching them into a file and sharing or re-hosting it, because the content itself is unencrypted. DRM addresses that: the video is encrypted, and keys are issued through a licence server, after an entitlement check and with rules such as expiry, resolution limits and device counts, only to a protected decryption module (via EME in browsers) that decrypts for playback without exposing the key or, on hardware-backed devices, the decoded frames. Even DRM cannot stop filming the screen, which is why premium services add forensic watermarking."
sources: [{"title": "AWS CloudFront: Signed URLs", "url": "https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-signed-urls.html", "note": "Expiring access to private content."}, {"title": "Wikipedia: Encrypted Media Extensions", "url": "https://en.wikipedia.org/wiki/Encrypted_Media_Extensions", "note": "DRM in browsers."}]
---

## Two different questions

A video service has paid content and must protect it. "Protect" covers two quite different questions, and the technologies that answer them are different:

1. **Who may download the bytes, and until when?** This is **access control**, and signed URLs and signed cookies answer it.
2. **What can someone do with the bytes once they have them?** This is **content protection**, and DRM (digital rights management) tries to answer it.

Most confusion in this area comes from expecting a tool for the first question to solve the second. This lesson walks through both, using CloudFront (Amazon's CDN) as the concrete example of signed delivery, and the browser DRM stack for content protection.

## What signed URLs do

A plain CDN URL is public: anyone who has it can fetch the file, forever. A **signed URL** adds a cryptographic proof that **your application** authorised this particular request, plus a **policy** saying what is allowed. CloudFront's version works like this:

1. You create a **trusted key group** on the distribution containing one or more **public keys** (RSA 2048 or ECDSA 256). You keep the matching **private keys** on your application servers.
2. When a user asks to watch something, your application checks they are **entitled** (logged in, subscribed, in a permitted country), then builds a **policy** (which resource, until when, optionally from which IP range) and signs it with the private key. The signature and policy go into the URL's query string.
3. When the request reaches a CloudFront edge, CloudFront verifies the **signature** with the public key (so the URL was not forged or tampered with) and checks the **policy** (not expired, right resource, right IP). Only then does it serve the file, from cache or from the origin.

The edge verifies the signature locally with the public key: it does not call your application. That is what makes the scheme scale to millions of requests per second.

### Canned versus custom policies

| Policy feature | Canned policy | Custom policy |
| --- | --- | --- |
| Expiry time | Yes | Yes |
| Start time (not valid before) | No | Optional |
| IP address or range | No | Optional |
| Covers multiple files with wildcards | No | Yes |
| URL length | Short | Longer (policy is embedded) |

A **canned** policy is just "this exact URL until this time", encoded compactly. A **custom** policy is a small JSON document (base64-encoded into the URL) that can add a start time, a source IP range, and a wildcard resource like `https://cdn.example.com/series/s1e3/*`.

Expiry is checked when a request **starts**: a download already in progress finishes, but a restarted download or a later **range request** (a request for part of a file, which video players use constantly) after expiry fails. Adding or changing query parameters after signing breaks the signature and returns 403.

### Signed cookies for streaming

An HLS or DASH video (lesson 13.06) is not one file: a two-hour film is thousands of segment files across several renditions. Signing every segment URL individually means rewriting every manifest per viewer, which also defeats manifest caching. **Signed cookies** solve this: your application sets three cookies (the policy, the signature and the key pair ID) scoped to the CDN domain, with a custom policy covering `/series/s1e3/*`. Every segment request automatically carries the cookies, and the manifests stay identical for all viewers and cacheable.

### Lock the origin

Signed URLs are useless if users can bypass the CDN and fetch the files straight from the bucket. CloudFront's **origin access control** (OAC) makes the bucket accept requests only from the distribution: the bucket policy allows reads only when the request is signed by CloudFront itself. Every protected setup needs this.

~~~dg-flow Signed delivery: the app checks entitlement and signs; the CDN verifies the signature and policy at the edge before serving. The bucket only talks to the CDN.
User -(watch episode)-> App: check subscription -(signed cookie, 4 h, /s1e3/*)-> *CDN edge verifies signature -(origin access control)-> Bucket of segments
~~~

## What signed URLs stop, and what they do not

Signed URLs and cookies control **who can fetch the bytes, and for how long**.

- A link posted publicly **stops working when it expires**. With a short expiry (minutes for single files, a few hours for a viewing session), sharing it is pointless.
- With an **IP-restricted** custom policy, the link does not work from anyone else's network at all.
- **Hotlinking** (another site embedding your content using your bandwidth) stops working too.

They do **not** control what happens **after** the bytes arrive. The segments are delivered as ordinary, **unencrypted** media files. A paying subscriber with a valid cookie can:

- save the segments and stitch them into a single MP4 to share forever;
- run a downloader that fetches every segment of every rendition while the cookie is valid;
- re-host the video on another site, where your signature no longer matters.

Access control protects the **door**. Once someone is legitimately through the door, the content is in the clear. For free or low-value content, that is often good enough. For premium films and live sport, studios and leagues require more, usually contractually.

### A half-way step: encrypted segments with a plain key

HLS supports **AES-128 encryption** of segments, with the decryption key fetched from a URL listed in the playlist. If the key URL is protected (signed, short-lived), casual copying of segments fails because they are encrypted. But the key is delivered to the player's JavaScript in the clear, so anyone who can play the video can also fetch the key and decrypt the files. It raises the bar slightly; it is **not DRM**.

## What DRM adds

**DRM** tries to protect the **content itself**, even on the authorised user's own device.

- **Encryption at rest and in transit.** The video is encrypted once at packaging time, typically with **Common Encryption** (CENC, ISO/IEC 23001-7), and stored and cached encrypted. Downloaded segments are useless on their own. CENC allows **one** set of encrypted files to work with **several** DRM systems, because only the key delivery differs.
- **Licences.** To play, the player obtains a **licence** containing the content key from a **licence server**. The licence server checks entitlement (the same checks your app does) and attaches **rules**: expiry (a rental lasts 48 hours), whether offline playback is allowed, a maximum resolution, how many devices.
- **A sealed decryption module.** Decryption happens inside a **Content Decryption Module** (CDM), a component supplied by the DRM vendor that runs outside the web page's JavaScript and, at the highest security levels, inside the device's **trusted execution environment** or secure video path. The key never reaches the application, and on hardware-backed devices decrypted frames never reach memory the operating system can read.

~~~dg-seq DRM playback: segments stay encrypted end to end; only the CDM, holding a licence, can decrypt them for display.
Player > CDN : GET encrypted segment (signed cookie)
CDN -> Player : encrypted segment, key ID in header
Player > CDM : need key for this key ID
CDM -> Player : licence request, signed by CDM
Player > Licence server : forward request + user token
Licence server > Licence server : check entitlement, set rules
Licence server -> Player : licence, key sealed to this CDM
Player > CDM : load licence
CDM > CDM : decrypt and decode, frames to screen
~~~

### In browsers: EME

**Encrypted Media Extensions** (EME), a **W3C Recommendation since September 2017**, is the browser API that lets an HTML `<video>` element talk to a CDM. It replaced the old browser plugins (Flash, Silverlight) that previously handled protected video. EME works with **Media Source Extensions** (the API JavaScript players use to feed segments to the video element) and CENC content, so one JavaScript player can handle all of them. The main CDMs:

- **Widevine** (Google): Chrome, Firefox, Edge, Android and many smart TVs.
- **PlayReady** (Microsoft): Edge on Windows, Xbox, many smart TVs.
- **FairPlay** (Apple): Safari, iOS, Apple TV.

A service that wants to play everywhere uses **multi-DRM**: one set of CENC-encrypted files (the `cbcs` encryption mode now works with all three), and a licence service that speaks all three licence protocols.

EME was controversial. CDMs are proprietary, closed components with per-browser licensing, raising concerns about open-source browsers, security research, privacy and accessibility; the **EFF resigned from the W3C** the day EME was published as a standard.

## What DRM does not stop either

DRM raises the bar a lot, but it is not absolute.

- **The analogue hole.** Pointing a camera at the screen still works. So does capturing a display output on hardware that ignores protection.
- **Compromised devices.** Software-only CDMs (Widevine **L3**) can be and have been broken; extracted keys let people decrypt content.

Services respond with layers:

- **Security levels.** Issue HD and 4K keys only to hardware-backed CDMs (Widevine **L1**, PlayReady SL3000); software-only clients get SD at most.
- **Output protection.** **HDCP** encrypts the link from the device to the display, so a cheap capture device on the HDMI cable gets nothing.
- **Forensic watermarking.** Embed an invisible, per-viewer mark in the picture so a leaked copy can be traced to the account that leaked it. A common scalable technique is **A/B watermarking**: encode two slightly different variants of each segment, and serve each viewer a unique sequence of A and B segments, chosen at the CDN edge. The sequence spells out a session ID, readable from a leaked copy.
- **Business controls.** Concurrent-stream limits, device registration, and fast takedown of leaked streams (especially for live sport, where the value is in the next 90 minutes).

~~~dg-layers Layers of protection for premium video, from controlling the door to tracing leaks after the fact.
Signed URLs and cookies | who may download, until when; stops shared links
Origin access control | no way around the CDN to the bucket
*DRM encryption and licences | files useless without a key held by a sealed CDM
Hardware security levels and HDCP | HD keys only to secure devices; protected display link
Forensic watermarking | leaks traced to the account that leaked them
~~~

~~~dg-matrix Which control stops which threat. No single layer covers everything, which is why premium services stack them.
Threat | Signed URL | DRM | Watermark
Link posted publicly | yes, expires | yes, no licence | partial, traces the poster
Hotlinking from another site | yes | yes | no
Subscriber saves segments | no | yes, encrypted | partial, traceable
Subscriber re-hosts the video | no | yes, if not cracked | yes, traceable
Camera pointed at screen | no | no | yes, traceable
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a subscription video service, tier by tier**

**Free trailers**: plain public CDN URLs with long cache lifetimes. Protection would cost more than it saves.

**Episodes**: signed cookies valid for 4 hours covering `/series/s1e3/*`, set when the user presses play after an entitlement check, so every HLS segment request is authorised while manifests stay cacheable. The bucket is reachable only through the CDN with origin access control. A shared link stops working within 4 hours.

**Premium films and live sport**: additionally CENC-encrypted and licensed through Widevine, FairPlay and PlayReady. HD and 4K keys are issued only to hardware-secure devices with HDCP; software-only browsers get 720p at most. Each stream carries an A/B forensic watermark, so a pirated copy found online can be traced back to the subscription within minutes, and that account's sessions are revoked.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**what each layer costs**

Signed cookies: one signing operation per session on your app servers (microseconds) and no change to caching. Cheap.

DRM: a licence server (or a multi-DRM vendor billed per licence, often fractions of a cent each), packaging changes, and player integration on every platform. For 10 million plays a day, licence fees and engineering are a real budget line.

Watermarking: two encoded variants of every segment (double the storage and packaging for protected titles) plus edge logic to pick variants per session. That is why it is usually reserved for the highest-value content.

</div>

## Common mistakes

- **Thinking signed URLs prevent copying.** They prevent sharing links, not saving files.
- **Long expiries.** A 7-day signed URL is close to a public link.
- **Leaving the bucket open.** Signed CDN URLs with a public origin bucket protect nothing.
- **Per-segment signed URLs in manifests.** Breaks manifest caching; use signed cookies or tokenised paths.
- **Treating AES-128 HLS as DRM.** The key goes to JavaScript in the clear.
- **DRM everywhere.** Protecting trailers and user-generated clips adds cost and playback failures for no benefit.

## In an interview

Separate the two questions explicitly: signed URLs (or cookies) control **access**, meaning who can fetch and until when, verified at the edge without calling your backend; DRM controls **use**, by encrypting content and releasing keys only to a sealed decryption module after a licence check. Then list what each does not stop: signed URLs do not stop a legitimate user saving and redistributing the files; DRM does not stop a camera pointed at the screen, which is why watermarking exists. Matching protection to content value (public, signed, DRM, watermarked) shows judgement.

<div class="take">
<div class="lab">Key takeaway</div>

signed URLs and cookies control who may download content and until when, verified at the CDN edge with a public key, so shared links die; with origin access control there is no way around them. They do nothing once a legitimate user has the bytes. DRM encrypts the content (CENC) and gives the key only to a sandboxed decryption module after a licence check (in browsers through EME with Widevine, PlayReady or FairPlay), so copied files are useless and usage rules are enforced; even DRM cannot stop camera capture, which forensic watermarking addresses by tracing leaks.

</div>

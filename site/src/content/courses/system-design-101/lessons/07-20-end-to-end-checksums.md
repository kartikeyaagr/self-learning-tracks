---
id: "07.20"
course: "system-design-101"
module: "07"
slug: "end-to-end-checksums"
title: "End-to-end checksums"
keyIdeas: ["Data can be corrupted anywhere: in memory (bit flips), in software bugs, on the network, on disk, or during copying.", "TCP and disk checksums only protect one hop. The end-to-end argument: only the endpoints can truly verify the data is what was sent.", "Compute a checksum (CRC32C, SHA-256) when the client creates the data, carry it with the data through every layer, and verify it on read and during background scrubbing.", "S3 lets clients send a checksum on upload and checks it before accepting the object.", "Background scrubbers read stored data periodically to catch silent corruption before all replicas go bad."]
question: "TCP already has checksums. Why also checksum data end to end?"
answer: "TCP's checksum only protects data on the wire between two TCP endpoints, and it is a weak 16-bit sum that misses whole classes of errors. Corruption also happens where TCP can't see it: in the sender's and receiver's memory, in proxies, load balancers and TLS terminators that end one connection and start another, in application and storage-engine bugs, and on disks (bad reads, misdirected or lost writes), as in the classic case of a gateway that swapped bytes in its own memory while every hop's checksum passed. At large scale these rare events happen daily. Only a checksum computed by the original sender from the data as it should be, carried with the data and verified by the final receiver (and rechecked on reads and at rest), confirms the data is intact end to end; lower-layer checks are still useful because they catch errors early and make end-to-end retries rare."
sources: [{"title": "Saltzer, Reed, Clark: End-to-End Arguments in System Design", "url": "https://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf", "note": "Why checks belong at the ends."}, {"title": "AWS S3: Checking object integrity", "url": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html", "note": "Checksums in practice."}]
---

## The question

Data travelling from your application to a storage service passes through many hands: your process's memory, a TLS library, the kernel, a network card, switches, a load balancer, a proxy, the storage service's front end, its memory, its storage engine, a disk controller, the disk itself, and later the whole chain in reverse. Several of those steps already check for errors. TCP has a checksum. Ethernet has a CRC. Disks have error-correcting codes. So why do serious storage systems insist on **another** checksum, computed by the sender and verified by the final receiver?

The answer is one of the most important principles in systems design, and it applies far beyond storage.

## The end-to-end argument

The **end-to-end argument**, first set out in 1981, says roughly this: a function such as "make sure the data arrived intact" can be implemented **completely and correctly only with the knowledge and help of the application at the endpoints**. Providing it inside the communication system alone is not enough. A partial version inside the system can still be worth having, but only as a **performance enhancement**.

The standard illustration is **careful file transfer** from host A to host B. List what can go wrong:

1. The file on A's disk may now read back incorrectly because of a disk fault.
2. The file system, the transfer program or the network software may make a mistake while buffering or copying the data, on A or B.
3. The processor or its memory may have a transient error during copying, on A or B.
4. The network may drop, change, lose or duplicate packets.
5. Either host may crash partway through.

Now suppose you make the network perfectly reliable. That fixes threat 4 and nothing else. Threats 1, 2, 3 and 5 all happen **outside** the network, inside the hosts, where no network check can see them. The only check that covers everything is end to end: A computes a checksum of the file as it should be; after B has written it, B reads it back, computes the same checksum and compares. If they differ, retry.

Does that make network reliability useless? No. If the network corrupted 10% of packets, the end-to-end check would fail constantly and every transfer would be retried many times. Lower-level checks are worth having **because they catch errors early and make end-to-end retries rare**. They just can't be the guarantee.

~~~dg-layers The path of an upload and which checks cover each part. Every layer's check covers only its own segment; only an end-to-end checksum covers all of them.
*Sender application | computes the end-to-end checksum
Client memory, TLS, kernel | covered by ECC memory at best
TCP between client and proxy | 16-bit TCP checksum, this hop only
Proxy or load balancer memory | not covered by any network check
TCP between proxy and server | a new checksum, this hop only
Server memory and storage engine | not covered by TCP
Disk controller and media | sector ECC, misdirected writes not caught
*Receiver verification | recomputes and compares the checksum
~~~

## A real failure: the byte-swapping gateway

A well-known incident shows exactly how per-hop checks fail. A network made of local networks joined by **gateways** used a checksum on every hop. One gateway developed a transient hardware fault in the code that copied data between its buffers: about once in every million bytes, it **swapped a pair of bytes**. Every hop's checksum passed, because each hop computed its checksum on the data **after** the gateway had already corrupted it. The corruption happened inside the gateway's memory, between two checked segments.

Operating system source files transferred through that gateway were silently corrupted. Nobody noticed until much later, and the owners had to repair them by hand, comparing against old printed listings. An end-to-end checksum, computed by the sender and checked by the receiver, would have caught every one of those transfers immediately.

~~~dg-compare Per-hop checksums versus an end-to-end checksum, for corruption that happens inside an intermediate device.
[Per-hop checks only]
Sender -(checked)-> Gateway memory -(checked)-> Receiver
- Corruption inside the gateway passes
- Each hop checksums already-bad data
- Silent damage discovered much later
[*End-to-end checksum]
Sender computes C -> any path -> Receiver recomputes C
+ Catches corruption anywhere on the path
+ Detects bugs in proxies and software
+ Failed check triggers a clean retry
~~~

## Why TCP's checksum is not enough

- **It covers one hop of one connection.** TCP checksums protect data between two TCP endpoints. Load balancers, TLS terminators and proxies end one TCP connection and start another, and in between the data sits in their memory, unprotected. So do your application's buffers, the storage server's memory, the storage engine's code and the disk firmware. That's the gateway problem, in modern form.
- **It is weak.** The TCP checksum is a 16-bit ones'-complement sum. It misses whole classes of errors, including reordered 16-bit words (exactly the gateway's byte-pair swaps, if aligned) and a fraction of random errors. Measurements on real networks have found that somewhere between about one in tens of millions and one in billions of packets carry errors that the TCP checksum fails to catch, usually introduced by faulty hardware on hosts or routers rather than on the wire. Ethernet's CRC is stronger, but it is recomputed at each link, so it too only covers one hop.
- **Scale makes rare errors routine.** Hard drives specify an unrecoverable read error rate of about 1 per 10^15 bits (enterprise) to 10^14 (consumer): roughly one bad read per 125 TB or 12.5 TB. Memory without ECC suffers bit flips; even CPUs occasionally compute wrong results silently. Disks sometimes write data to the wrong place (**misdirected writes**) or report a write that never happened (**lost writes**), and the disk's own error correction can't detect either, because the sector it reads back is internally consistent, just wrong. At exabyte scale, events that are one-in-a-trillion happen every day.

~~~dg-bars Expected unrecoverable disk read errors per day at the enterprise rate of 1 per 10^15 bits, by daily read volume. At large scale, rare errors become routine.
10 TB read per day | 0.08 | about 1 every 12 days
1 PB read per day | 8 | about 8 a day
*1 EB read per day | 8000 | about 8,000 a day
~~~

## How storage systems apply it

The principle, made practical:

- **Compute the checksum as early as possible**, ideally in the client application before the data leaves it, from the data as the application knows it.
- **Carry it with the data** through every layer.
- **Verify it at every place the data lands**: when the service receives it, before acknowledging; when it's written to disk; on every read; and in periodic background **scrubbing** of data at rest, to find silent corruption before a user does and while other redundant copies or shards can still repair it.
- **Store it alongside the data**, so a reader can verify what it got.

Inside storage systems you've met in this module: every block in the WAS stream layer carries its own checksum, verified on every read (lesson 07.18); RocksDB checksums every SST block and WAL record (lesson 07.10); Bitcask stores a CRC per record (lesson 07.08). Filesystems such as ZFS store each block's checksum in the **parent** block pointer rather than next to the block itself, so a misdirected or lost write (which leaves an old but internally valid block in place) is caught, because the parent's checksum doesn't match.

## How S3 does it

S3 exposes end-to-end integrity to clients directly:

- **Algorithms**: CRC64NVME (the default), CRC32, CRC32C, SHA-1, SHA-256, MD5, XXHash64, XXHash3, XXHash128 and SHA-512. The client sends its value in an `x-amz-checksum-...` header. S3 computes its own checksum on the bytes it received and **validates it before storing** the object, then stores the checksum with it. A mismatch rejects the PUT.
- **Automatic**: AWS SDKs compute a checksum by default if you don't supply one; if neither client nor SDK does, S3 computes CRC64NVME itself on arrival (which protects storage but not the path from your application).
- **Trailing checksums**: for streaming uploads, where the checksum isn't known until all the data has been read, it can be sent as a **trailer** after the body, announced with `x-amz-trailer`. The client computes as it streams, with no second pass over the data.
- **Multipart uploads** have two kinds. A **full-object** checksum covers every byte from first to last (CRC64NVME, CRC32, CRC32C), because CRCs of parts can be mathematically combined. A **composite** checksum is a checksum of the per-part checksums (for SHA and MD5 families, which can't be combined).
- **Downloads**: `GetObject` and `HeadObject` return the stored checksum when checksum mode is enabled, so the client can verify what it received.
- **ETags are not checksums.** The ETag is the MD5 of the data only for single-part uploads that are plaintext or SSE-S3 encrypted. For multipart uploads it's an MD5 of the parts' MD5s with "-N" appended. Don't rely on it for integrity.
- **At rest**: S3 Batch Operations' **Compute checksum** recalculates checksums for billions of stored objects without downloading them and produces an integrity report.

~~~dg-seq End-to-end integrity with S3: the client checksums before sending, S3 validates before storing, and the client verifies again on download.
Client > Client : compute CRC64NVME of the file
Client > S3 : PUT + x-amz-checksum-crc64nvme
S3 > S3 : recompute on received bytes, compare
S3 -> Client : 200 OK (or 400 on mismatch)
Client > S3 : GET with checksum mode enabled
S3 -> Client : bytes + stored checksum
Client > Client : recompute and compare
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a corrupting proxy**

A company's upload service sits behind a TLS-terminating proxy with a faulty memory module that flips about one bit per 10^12 bytes. The service uploads 50 TB a day, so about 50 objects a day are silently corrupted inside the proxy. TCP checksums pass on both hops, because each hop checksums the already-damaged data.

The client computes SHA-256 before upload and sends it as `x-amz-checksum-sha256`. For those 50 objects, S3's computed value doesn't match, the PUT is rejected with an error, and the client retries cleanly, probably through a different proxy. Without the end-to-end checksum, 50 corrupt objects a day would be stored with an "OK", and discovered months later when someone opened a damaged backup.

</div>

<div class="ex">
<div class="lab">Worked example</div>

**choosing the algorithm**

A backup tool uploads 5 GB files. Computing SHA-256 runs at roughly 0.5 to 2 GB/s per core; CRC64 or CRC32C with hardware instructions runs at 10 GB/s or more. For pure corruption detection, a CRC is ample (accidental corruption doesn't conspire to match a 64-bit CRC), costs well under a second per file, and supports full-object checksums for multipart uploads. Use SHA-256 when you also need protection against deliberate tampering, or a content-addressed key.

</div>

## Common mistakes

- **Trusting TCP or TLS.** TLS does authenticate data in transit, but only between its two endpoints; a TLS-terminating proxy decrypts into plain memory, and corruption there is invisible.
- **Checksumming after the data has been through several layers.** Compute it where the data originates.
- **Using the ETag as a checksum** for multipart uploads.
- **Verifying on write but never at rest.** Bit rot is found by scrubbing, not by the next customer who reads the file.
- **Detecting but not repairing.** A checksum failure on read should trigger a read from another replica or shard and a repair of the bad copy.

## In an interview

Whenever data integrity matters (backups, financial records, media pipelines), say: "The client computes a checksum (CRC32C or CRC64 for speed, SHA-256 if we need tamper-evidence) and sends it with the upload; the service verifies before acknowledging, stores the checksum with the data, verifies on every read and scrubs at rest; downloads verify too." If asked why, give the end-to-end argument in one sentence: lower-layer checks only cover their own segment, and corruption happens in memory, proxies, software and disks between them.

<div class="take">
<div class="lab">Key takeaway</div>

only the endpoints can confirm that what was stored is what was sent. Checksum at the source, carry the checksum with the data, verify before acknowledging, on every read and at rest, and treat lower-layer checks like TCP's as a way to catch errors sooner, not as the guarantee.

</div>

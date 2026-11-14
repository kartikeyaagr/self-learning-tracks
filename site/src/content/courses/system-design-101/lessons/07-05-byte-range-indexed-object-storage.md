---
id: "07.05"
course: "system-design-101"
module: "07"
slug: "byte-range-indexed-object-storage"
title: "Byte-range indexed object storage"
keyIdeas: ["If an object has an index of where each record lives (offset and length), a client can fetch just those bytes with an HTTP Range request (`Range: bytes=1000-1999`).", "Object stores like S3 support range GETs, so a 10 GB file can serve point reads without downloading it all.", "Formats like Parquet put a footer at the end listing row groups and column chunks with byte offsets. Readers fetch the footer first, then only the ranges they need.", "Pack many small items into one large object plus an index; this cuts per-object overhead and request costs (Facebook Haystack did this for photos)."]
question: "How can a client read one record from a 10 GB object without downloading it all?"
answer: "Store an index that maps each record to a byte offset and length, either inside the object (typically a footer at the end, as Parquet and ZIP do, with its length in the last few bytes) or alongside it in a database. The client first fetches the index with a small range request (for example `Range: bytes=-8` to get the footer length, then the footer), looks up the record's offset and length, then sends `Range: bytes=start-end` and receives a `206 Partial Content` with only those bytes. Statistics in the index can also rule out whole blocks, and an ETag check (If-Range or If-Match) ensures the pieces come from the same version of the object."
sources: [{"title": "MDN: HTTP range requests", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests", "note": "Fetching a byte range of an object."}, {"title": "Apache Parquet: File format", "url": "https://parquet.apache.org/docs/file-format/", "note": "Footer index that lets readers fetch only what they need."}]
---

## The problem: big objects, small reads

Object stores such as S3, Google Cloud Storage and Azure Blob Storage hold files (objects) that can be gigabytes or terabytes in size. Often you only need a small part: one record, one column, one frame of a video, the last kilobyte of a log. Downloading a 10 GB object to read 4 KB of it would take a minute and cost real money in transfer. So two questions matter:

1. **Can the store return just a slice of an object?** Yes: HTTP range requests.
2. **How do you know which slice you need?** You need an index that maps "the thing I want" to a byte offset and length.

Put those together and object storage starts to behave like a very large, very slow, random-access file. That is the foundation of modern data lakes, where query engines read Parquet files directly from S3, and of features like video seeking and resumable downloads.

## HTTP range requests

HTTP has supported partial downloads since HTTP/1.1. The mechanism is small:

- A server advertises support with the response header `Accept-Ranges: bytes`.
- The client sends `Range: bytes=1000-1999` to ask for bytes 1,000 to 1,999 inclusive (1,000 bytes). Variants: `bytes=1000-` means "from 1,000 to the end"; `bytes=-500` means "the last 500 bytes", which is very useful when the index is at the end of the file.
- The server replies **`206 Partial Content`** with `Content-Range: bytes 1000-1999/10737418240`, where the number after the slash is the full object size.
- Several ranges can be requested at once (`bytes=0-99,5000-5099`) and come back as a multipart response. S3's GetObject supports only **one range per request**, so clients issue several requests in parallel instead.
- A range that starts past the end of the object gets `416 Range Not Satisfiable`.
- `If-Range: "etag"` means "send the range only if the object still has this version; otherwise send the whole new object". It protects a resumed download from stitching together pieces of two different versions.

~~~dg-seq A range request: the client asks for 1,000 bytes and the server returns only those, with status 206 and the total size.
Client > Server : GET /big.bin, Range: bytes=1000-1999
Server > Server : check range within object size
Server -> Client : 206 Partial Content
Server -> Client : Content-Range: bytes 1000-1999/10737418240
Server -> Client : 1,000 bytes of body
~~~

### What a range request costs

On object storage, each GET has a fixed overhead regardless of size: **time to first byte** of roughly 10 to 100 ms, plus a per-request charge. After that, a single connection streams at maybe 50 to 100 MB/s. So the economics are different from a local disk:

- Reading 4 KB costs about as much latency as reading 1 MB. Small ranges are fine for single lookups, but reading thousands of tiny ranges one after another is slow.
- Large reads are best split into parallel ranges (say 8 to 16 MB each over many connections). This is how fast S3 downloaders reach gigabytes per second: many concurrent range GETs on one object.
- Nearby ranges should be **coalesced** into one request if the gap between them is small. Reading 100 KB of unwanted bytes is cheaper than paying another 30 ms round trip.

## You need an index to know which bytes

A range request is only useful if you know where the data is. There are three common ways to provide the index:

- **An index inside the object**, usually in a footer at the end (Parquet, ORC, ZIP's central directory).
- **A separate index object** stored beside the data (`data.bin` plus `data.idx`), or rows in a database that hold the offset and length of each record.
- **Fixed-size records**, where record N starts at N × record size and no index is needed at all.

### Why the index goes at the end

Writers produce large files in one streaming pass. They don't know where each block will land until they've written it, so they can't put a complete index at the front without either buffering the whole file or going back to patch the start (which object stores don't allow: objects are immutable once written). The solution is to **write the data first and the index last**, followed by the index's length at a fixed position from the end. A reader then starts from the end.

## Case study: Parquet

Parquet is the standard columnar file format for analytics. "Columnar" means values of the same column are stored together, so a query that needs 2 columns out of 50 reads only those 2.

~~~dg-tree The structure of a Parquet file. Readers navigate down this hierarchy using offsets stored in the footer.
Parquet file | starts and ends with magic PAR1
  Row group 1 | for example 128 MB of rows
    Column chunk: user_id | contiguous bytes
      Page 1 | about 1 MB, compressed
      Page 2
    Column chunk: event_type
    Column chunk: payload
  Row group 2
    Column chunk: user_id
    Column chunk: event_type
  *Footer metadata | schema, offsets, min/max per chunk
  Footer length + PAR1 | last 8 bytes
~~~

- The file starts and ends with the 4-byte magic `PAR1`.
- Rows are split into **row groups** (commonly 128 MB to 1 GB of data each).
- Within a row group, each column is stored as a **column chunk**, a contiguous byte range, made of **pages** (around 1 MB) that are compressed and encoded independently.
- The **footer** holds the file metadata: the schema, and for every row group and column chunk its byte offset, size, and **statistics** such as min and max values and null counts.
- The last 8 bytes are a 4-byte footer length and the closing `PAR1`.

A reader does the reverse of the writer:

1. Read the last 8 bytes (a suffix range, `bytes=-8`) to get the footer length and check the magic. Many readers simply read the last 64 KB in one go, which usually captures the whole footer and saves a round trip.
2. Read the footer with one range request.
3. Use the statistics to skip row groups that cannot match the query (**predicate pushdown**). If the query wants events from 12 March and a row group's timestamp min/max is 1 to 5 March, it is skipped without reading a byte of it.
4. Fetch only the needed column chunks of the surviving row groups, with range requests in parallel.

~~~dg-seq Reading one column from a Parquet file in object storage: two small reads to find the index, then only the bytes that matter.
Query engine > Object store : GET Range: bytes=-8
Object store -> Query engine : footer length 200 KB + PAR1
Query engine > Object store : GET footer range
Object store -> Query engine : schema, offsets, min/max stats
Query engine > Query engine : stats prune 77 of 80 row groups
Query engine > Object store : 3 parallel GETs, user_id chunks only
Object store -> Query engine : about 30 MB
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one column from a 10 GB file in S3**

A query needs `user_id` for events on one day. The engine range-reads the last 8 bytes of the 10 GB Parquet file, then the 200 KB footer. The footer lists 80 row groups of about 125 MB each, with min and max timestamps. Statistics show only 3 of the 80 row groups overlap that day.

Each of those row groups has 20 columns; `user_id` is about 10 MB per row group compressed. The engine issues 3 range GETs, about 30 MB in total, in parallel. Total: about 30.2 MB read and 5 requests, versus 10 GB and one very long download. At 100 MB/s per connection, that is under half a second instead of over a minute and a half.

</div>

~~~dg-bars Bytes transferred to answer the same query, by approach.
Download whole object | 10240 | 10 GB
Read only matching row groups | 375 | about 375 MB
*Matching row groups, one column | 30 | about 30 MB
Footer only | 0.2 | 200 KB
~~~

## Other designs that rely on byte ranges

The same pattern shows up in many places:

- **Packed small objects.** Storing billions of tiny files as separate objects is expensive (per-request costs, metadata overhead). Systems pack many small records into one large object and keep an index of `(object, offset, length)` per record in a database. Facebook's Haystack (lesson 07.13) does this for photos; many log and backup systems do it too.
- **Video streaming.** An MP4's `moov` box is an index of where every frame lives. Players read it, then fetch the ranges for the part being watched, which is how seeking works without downloading the whole film. (Encoders often move `moov` to the front, called "fast start", so playback can begin immediately.)
- **ZIP files.** The central directory is at the end. A tool can list or extract one file from a remote ZIP with two or three range requests.
- **Resumable and parallel downloads.** Download managers split a file into ranges, fetch them concurrently and retry only failed ranges, using `If-Range` or an ETag check to make sure the object didn't change between pieces.
- **Cloud-optimised formats.** Geospatial imagery (Cloud-Optimised GeoTIFF) and genomics files are laid out with internal tiling and an index so that a map viewer can fetch only the tiles on screen.

~~~dg-compare Two ways to give a reader the offsets it needs. Both turn a whole-object download into a few small range reads.
[Index inside the object]
Read footer -(offsets)-> Range GET data
+ One object, can't get out of sync
+ Written in one streaming pass
- Two round trips before data
[Index in a database]
Query DB -(object, offset, len)-> Range GET data
+ One round trip to storage
+ Can index billions of packed records
- Index and data can drift apart
~~~

## Pitfalls

- **Too many tiny ranges.** Thousands of 4 KB reads at 30 ms each are slower than one larger read. Coalesce nearby ranges and read in parallel.
- **Mixing versions.** If an object is overwritten while you read it in pieces, you can stitch together parts of two versions. Pin the version with `If-Match` or `If-Range` on the ETag, or use a versioned object ID.
- **Tiny row groups or pages.** A Parquet file written with 1 MB row groups has a huge footer and thousands of chunks to fetch. Too large and pruning becomes coarse. 128 MB to 512 MB row groups are a common middle ground.
- **Assuming multi-range support.** S3 serves one range per request; code written for multi-range HTTP servers must split requests.
- **Forgetting cost.** Requests are billed. A design that issues 50 range GETs per user query at millions of queries a day can cost more in requests than in storage.

## In an interview

When a design stores large files in object storage but needs fine-grained access, say: "I'll store an index, either in the file's footer or in the metadata database, with offset and length per record, and read with HTTP range GETs." Mention parallel ranged reads for throughput, and coalescing for latency. If the prompt involves analytics on a data lake, name Parquet's footer and min/max statistics as the reason a query on a 10 TB table can read only a few gigabytes.

<div class="take">
<div class="lab">Key takeaway</div>

HTTP range requests let you read any slice of a large object, returning 206 with just those bytes, and an index (a footer such as Parquet's, or offsets in a database) tells you which slice to ask for. Together they make immutable object storage behave like a random-access file.

</div>

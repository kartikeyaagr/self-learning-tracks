---
id: "07.04"
course: "system-design-101"
module: "07"
slug: "custom-binary-file-format"
title: "Custom binary file format"
keyIdeas: ["A binary file format lays out records as bytes with a defined structure, which is smaller and faster to parse than text.", "Typical parts: a magic number (identifies the file type), a version (allows format changes), a header, the records, and often a footer with an index and checksum.", "Length-prefix variable-size fields so readers can skip them. Varints save space for small numbers.", "Fix the byte order (endianness) and alignment.", "Add checksums per block or record to detect corruption. Design for evolution: readers should skip fields they don't understand."]
question: "Why put a magic number and version at the start of a file format?"
answer: "A magic number lets any program instantly tell this file type from every other and reject garbage, an empty file or the wrong file before misreading it, and a well-chosen one also exposes transfer damage such as line-ending conversion. A version number lets readers cope with format changes: new code can recognise and still read old layouts, and old code can detect a newer file and refuse it cleanly (or, with separate read and write versions as in SQLite, open it read-only) instead of interpreting the new bytes with the old layout and silently returning or writing corrupt data."
sources: [{"title": "Protocol Buffers: Encoding", "url": "https://protobuf.dev/programming-guides/encoding/", "note": "Varints, tags, wire format."}, {"title": "SQLite: Database file format", "url": "https://www.sqlite.org/fileformat.html", "note": "A real binary format, fully documented."}]
---

## Why design a binary format at all

Sooner or later every storage engine has to decide exactly which bytes go into its files: log records, index pages, data blocks, headers. You could use text, such as JSON or CSV. Text is easy to read in a terminal and easy to debug, but it is a poor choice for the core of an engine:

- **It is large.** The number 1700000000 takes 10 characters as text but fits in 4 bytes as a binary integer. Field names repeat in every record.
- **It is slow to parse.** To find the fifth field of a JSON record you must scan every byte before it, handling quotes, escapes and nesting. A binary format with fixed sizes or explicit lengths lets you jump straight to it.
- **It is ambiguous.** Is `1.10` a number or a version string? What character encoding is this? Binary formats state their types and byte order explicitly.

But binary has a cost too. You cannot eyeball a file and see what's wrong with it, and a reader that misunderstands the layout will happily turn garbage into plausible-looking numbers. So a good binary format is designed to **identify itself, describe its own structure, evolve over time, and fail loudly**. This lesson takes those properties one at a time, using two well-known formats as worked examples: Protocol Buffers for messages and SQLite for whole database files.

## The design checklist

Here is what every serious file format needs, and why. The rest of the lesson shows each item in practice.

- **A magic number** at the very start: a fixed byte sequence that says "this is a file of type X". Tools can recognise the file instantly and reject the wrong one.
- **A version** next to it, so new software can read old files and old software can refuse new ones cleanly.
- **Explicit lengths or fixed sizes**, so a reader can skip what it doesn't need or doesn't understand.
- **Checksums** on blocks or records, to detect corruption and torn writes (lesson 07.03).
- **A fixed byte order** (endianness), so a file written on one machine reads correctly on another.
- **Room to grow**: reserved bytes, flags, or a tagging scheme that lets you add fields later.

## Magic numbers: what is this file?

A **magic number** is a short, fixed sequence of bytes at offset 0. Some familiar ones:

| Format | First bytes | Notes |
| --- | --- | --- |
| PNG | `89 50 4E 47 0D 0A 1A 0A` | `\x89PNG\r\n\x1a\n` |
| gzip | `1F 8B` | two bytes |
| ELF executable | `7F 45 4C 46` | `\x7fELF` |
| Parquet | `50 41 52 31` | `PAR1`, at the start and the end |
| SQLite | `SQLite format 3\0` | 16 bytes |

Magic numbers do several jobs. A tool like `file` can identify a file without trusting its name. A database that is pointed at the wrong path (a log file, a backup in another format, an empty file) refuses to open it instead of interpreting random bytes as pages and "repairing" them. Good magic numbers are also chosen to catch specific damage. PNG's begins with a byte above 127, which breaks if the file passes through a 7-bit channel, and contains `\r\n` and `\n`, which change if a tool converts line endings. One look at the first eight bytes tells you the file was mangled in transit.

~~~dg-tree What a careful reader does when it opens a file. Every branch that does not match ends in a clear error, not a guess.
Open file
  Read first 16 bytes
    Magic matches? | no: "not a data file", stop
      Version known? | read version field
        *Version I understand | parse normally
        Newer read version | refuse: "upgrade software"
        Newer write version only | open read-only
      Checksum of header OK? | no: "corrupt header", stop
~~~

## Versions: surviving change

Formats change. You add compression, a new index, a bigger counter. Without a version number a reader has no way to know which layout it is looking at, and the failure mode is the worst kind: silently wrong data.

With a version, the rules become simple:

- **New code reading old files**: check the version and use the old layout. Keep that code path for as long as old files exist.
- **Old code reading new files**: see an unknown version and refuse with a clear message ("this file was written by version 3; please upgrade").

SQLite refines this with **two** version numbers, which is a neat trick. The header holds a **write version** and a **read version**. If a reader doesn't recognise the write version, it can still read the file safely but must not modify it, so it opens it read-only. If it doesn't recognise the read version, it can't even read the file, so it refuses. This lets a format add a feature that only matters to writers (such as a new journaling mode) without locking out older readers.

## Case study: the SQLite file header

An SQLite database is one file made of fixed-size **pages**. The first page begins with a 100-byte header, and all multi-byte integers in it are **big-endian** (most significant byte first), so the file reads identically on any CPU.

~~~dg-layers The start of every SQLite file. The first 100 bytes tell any reader what the file is and whether it may open it.
*Magic: "SQLite format 3\0" | bytes 0 to 15, identifies the file
Page size | bytes 16 to 17, how to cut the file into pages
Write version, read version | bytes 18 to 19, open read-only or refuse
Reserved space per page | byte 20, room for extensions such as checksums
File change counter | bytes 24 to 27, lets caches detect changes
Page count, freelist, schema cookie | more fields, up to byte 99
B-tree pages | the rest of page 1 and every following page
~~~

A few details show the thinking. The page size (any power of two from 512 to 65,536 bytes) is stored once, so every later offset can be computed. The **file change counter** is incremented on each transaction, so a process holding cached pages can tell cheaply whether another process has changed the file. The "reserved space" byte lets extensions (like encryption or per-page checksums) claim a few bytes at the end of every page without changing the format. Crash recovery is also part of the format: SQLite's rollback journal and its WAL file each have their own headers with magic numbers, salts and checksums, so a leftover journal from a crash can be recognised and replayed or discarded correctly.

## Case study: Protocol Buffers on the wire

Protocol Buffers (protobuf) is built for messages rather than files, and it solves a different problem well: compactness and **schema evolution**, meaning old and new programs can exchange messages even after fields are added.

### Varints

Most integers are small, so protobuf stores them as **varints** (variable-length integers) of 1 to 10 bytes. Each byte holds 7 bits of the value plus a **continuation bit** (the top bit): 1 means "more bytes follow", 0 means "this is the last byte". The least significant group comes first.

Take 150. In binary that's `1001 0110`, which needs 8 bits, so it takes two bytes. The low 7 bits are `001 0110` (0x16); add the continuation bit and you get `1001 0110` = **0x96**. The remaining bits, `1`, go in the next byte with no continuation bit: **0x01**. So 150 encodes as `96 01`. Any number below 128 takes a single byte.

### Tags and wire types

A message is a sequence of fields, each written as a **key** followed by a **value**. The key is `(field_number << 3) | wire_type`. The **wire type** (3 bits) says how to find the end of the value: 0 is a varint, 1 is 64-bit fixed, 2 is length-delimited (a varint length, then that many bytes: strings, bytes, nested messages), 5 is 32-bit fixed. Field 1 as a varint gives key `(1 << 3) | 0 = 08`, so "field 1 = 150" is the three bytes `08 96 01`.

~~~dg-seq How a reader decodes "field 1 = 150". The wire type tells it how long the value is, even for a field it has never heard of.
Reader > Byte 08 : read key varint
Byte 08 -> Reader : field 1, varint
Reader > Byte 96 : more follows, bits 0x16
Reader > Byte 01 : last byte, value 1
Reader > Reader : 0x16 + (1 << 7) = 150
~~~

The wire type is the clever part. Because every value's length can be determined without knowing the schema, a reader that meets **an unknown field number simply skips it**. That is what makes protobuf evolve safely: a new writer adds field 4; an old reader skips field 4 and carries on. The rule for schema authors follows: never reuse or renumber a field, only add new ones.

### ZigZag for negative numbers

A plain `int32` of -1 is stored as a 64-bit two's-complement number, all ones, which takes the maximum **10 bytes** as a varint. The `sint32` and `sint64` types fix this with **ZigZag** encoding, which interleaves signs: 0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4. Small magnitudes, positive or negative, become small unsigned numbers and take one byte. Use `sint` types for fields that are often negative, such as deltas.

<div class="ex">
<div class="lab">Worked example</div>

**one sensor reading, JSON versus protobuf**

The record is user_id 123456, timestamp 1700000000, temperature 21.5. As JSON, `{"user_id":123456,"ts":1700000000,"temp":21.5}` is 46 bytes.

As protobuf: field 1 (varint 123456, which is below 2^21 so 3 bytes) plus its 1-byte key is 4 bytes; field 2 (varint 1700000000, below 2^35 so 5 bytes) plus key is 6 bytes; field 3 (a 32-bit float, wire type 5) is 1 + 4 = 5 bytes. Total **15 bytes**, about a third of the JSON. At 1 billion readings a day that is 31 GB saved daily, and decoding needs no string parsing at all.

</div>

~~~dg-bars Size of the same sensor reading in three encodings.
JSON text | 46 | 46 bytes
Fixed-width binary struct | 20 | 20 bytes (8 + 8 + 4)
*Protobuf | 15 | 15 bytes
~~~

## Byte order and checksums

**Endianness** is the order in which a multi-byte number's bytes are stored. The 32-bit number 0x01020304 is `01 02 03 04` in big-endian and `04 03 02 01` in little-endian. x86 and ARM CPUs are little-endian in practice; network protocols and SQLite use big-endian. Which you choose matters less than choosing **one** and writing it down. A format that "uses the machine's native order" breaks the day a file moves between machines.

**Checksums** belong on every unit that can be independently damaged: each record in a log, each page in a data file, each block in an object. Use CRC32C for speed or a cryptographic hash (SHA-256) if you also need to detect deliberate tampering. Store the checksum next to the data it covers and verify it on every read. Lesson 07.20 explains why checksums at this level catch errors that the network and disk never report.

## Common mistakes

- **No version field.** The first format change forces a painful guess about which files are which.
- **Native byte order or struct dumps.** Writing a C struct straight to disk bakes in the compiler's padding and the CPU's endianness.
- **Implicit lengths.** If a reader must understand a record to know where it ends, it can't skip unknown records, and one bad byte desynchronises everything after it.
- **Reusing protobuf field numbers.** An old reader decodes the new field as the old one, silently.
- **Silent "best effort" parsing.** A format should fail with a clear error, not return wrong numbers.

<div class="ex">
<div class="lab">Worked example</div>

**a log segment header that evolves**

A team designs a log segment file: 4 bytes magic `SEG1`, 2 bytes format version, 2 bytes flags, 8 bytes base offset, all little-endian, then records of [4-byte length][4-byte CRC32C][payload]. That is a 16-byte header and 8 bytes of framing per record.

A year later they add compression. They bump the version to 2 and set flag bit 0 to mean "payload is LZ4-compressed". New readers handle both versions. An old reader that finds a version-2 segment on a restored backup refuses it with "unsupported segment version 2" instead of feeding compressed bytes to the application as if they were records. Without the version field, it would have read garbage lengths, failed checksums on every record, and possibly truncated the file as a "torn tail".

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a binary format should identify itself with a magic number, state its version, make every piece's length explicit, fix its byte order and checksum its blocks. Then it can be read fast, evolve safely, and fail loudly instead of silently returning wrong data.

</div>

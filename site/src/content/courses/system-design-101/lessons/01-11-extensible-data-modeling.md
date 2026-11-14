---
id: "01.11"
course: "system-design-101"
module: "01"
slug: "extensible-data-modeling"
title: "Extensible data modeling"
keyIdeas: ["Requirements change, so design data models that can grow without painful migrations.", "Options: add nullable columns (simple, explicit), use a JSON/JSONB column for optional or per-tenant attributes, or use separate extension tables.", "FriendFeed stored entities as opaque blobs in MySQL and built separate index tables for fields they query. New fields need no schema change; new queries need a new index table.", "JSONB in Postgres can be indexed (GIN) and queried, so it's a good home for sparse attributes. Keep core, frequently filtered fields as real columns.", "For data passed between services, use schemas with compatibility rules (Avro, Protobuf): add optional fields, never reuse field numbers, never change types."]
question: "When would you use a jsonb column instead of adding a new column?"
answer: "Use a jsonb column for optional, sparse or per-tenant attributes that vary from row to row and are mostly read together with the record, such as product specifications that differ by category or user-defined custom fields, especially when new attributes appear often and you don't want a migration for each. Add a real column instead for fields you filter, sort or join on heavily, need constraints or foreign keys on, or update often on their own, since the database can type, enforce and index those far better."
sources: [{"title": "How FriendFeed uses MySQL to store schema-less data", "url": "https://backchannel.org/blog/friendfeed-schemaless-mysql", "note": "Blobs plus separate index tables to avoid migrations."}, {"title": "PostgreSQL docs: JSON Types", "url": "https://www.postgresql.org/docs/current/datatype-json.html", "note": "json vs jsonb and indexing JSON fields."}, {"title": "Kleppmann: Schema evolution in Avro, Protobuf and Thrift", "url": "https://martin.kleppmann.com/2012/12/05/schema-evolution-in-avro-protocol-buffers-thrift.html", "note": "Adding fields without breaking old readers."}]
---

## Requirements change; schemas resist

Products keep adding fields: a new profile attribute, a per-customer setting, a new event type, a product category with its own specifications. In code, adding a field is a one-line change. In a large, busy database, it can be anything from instant to a multi-hour operation that locks the table.

FriendFeed, an early social network, described tables of 10 to 20 million rows where adding an index or a column locked MySQL for hours, so the team could not ship features without downtime. That pain shaped a famous design. This lesson is about modelling data so it can evolve: when real columns are fine, when to store flexible documents, how to index them, and how to change formats that flow between services.

## Option 1: just add columns

For core fields that you filter on, join on, sort by or constrain (unique, not null, foreign key), real columns are still the best home. The database knows their type, enforces their rules, stores them compactly and can index them efficiently.

Schema changes are much less scary than they used to be:

- **Postgres** adds a nullable column, or a column with a constant default (since version 11), as a metadata-only change. It takes milliseconds regardless of table size, because existing rows are not rewritten; the default is stored once in the catalogue.
- **MySQL 8.0** supports `ALGORITHM=INSTANT` for adding columns in most cases, also without rewriting rows.
- **New indexes** can be built without blocking writes (`CREATE INDEX CONCURRENTLY` in Postgres, online DDL in MySQL), though the build still takes time and I/O on a big table.

What remains genuinely painful: changing a column's type, adding a column with a default that must be computed per row, backfilling values into hundreds of millions of rows, and adding constraints that must be checked against existing data. Those need online migration techniques (Module 3): add the new column, backfill in batches, write to both, switch reads, drop the old one.

~~~dg-timeline An online migration that renames and retypes a column on a 500-million-row table without downtime. Each step is safe to pause or roll back.
Day 1 | add new nullable column: instant
Day 1 | deploy code that writes both columns
Days 2 to 4 | *backfill old rows in batches of 10,000
Day 5 | switch reads to the new column
Day 12 | stop writing old column, then drop it
~~~

## Option 2: schemaless storage inside SQL (FriendFeed)

FriendFeed stored every entity (posts, comments, likes) in one MySQL table with only a few real columns:

- `added_id`: an auto-incrementing primary key, so new rows are appended in order on disk. That keeps writes sequential and fast.
- `id`: a UUID identifying the entity.
- `updated`: a timestamp.
- `body`: a compressed, serialised dictionary holding all of the entity's properties. MySQL treated it as an opaque blob.

Adding a property to an entity now needed no schema change at all: the application simply started writing it into the dictionary.

To query by a property, they created **separate index tables**, each holding just the indexed value and the entity ID, for example `index_user_id (user_id, entity_id)`. Each index table was sharded by its own key, so "all entities by user 42" was one lookup on one shard. Adding a new queryable field meant creating a new index table (a fresh, empty table, so no locks on anything existing) and letting a background process called **the Cleaner** populate it by scanning the entities.

~~~dg-flow FriendFeed's write path: the entity is stored as one opaque blob, and each index gets its own small table.
[Write] Application -(serialised dict)-> *entities table
[Write] Application -(user_id, entity_id)-> index_user_id table
[Write] Application -(link_url, entity_id)-> index_link table
~~~

Index tables could briefly disagree with the entities, because they lived on different shards and were written in separate steps, with no transaction spanning them. FriendFeed accepted that:

- The entities table was the **source of truth**.
- Reads used an index only to find **candidate IDs**, then fetched the real entities and **re-checked** the condition, discarding any that no longer matched.
- The Cleaner kept fixing inconsistencies in the background, within seconds.

~~~dg-seq A query by user under FriendFeed's scheme. The index may be stale, so the reader re-checks every candidate against the entity itself.
App > index_user_id : entity ids for user 42
index_user_id -> App : e1, e2, e3
App > entities : fetch e1, e2, e3
entities -> App : 3 blobs
App > App : decode, drop e2 (user changed)
App -> App : return e1, e3
~~~

The trade-off: total flexibility for adding fields, in exchange for doing in application code what a database normally does (indexing, consistency, type checking). Many later document stores and key-value systems offer a similar model natively.

## Option 3: JSON columns (Postgres JSONB)

Modern relational databases let you mix the two: real columns for core fields, plus a JSON column for the flexible part. Postgres has two JSON types:

- `json` stores the text exactly as given (keeping whitespace, key order and duplicate keys) and re-parses it on every access.
- `jsonb` stores a decomposed binary form. It is slightly slower to write, much faster to query, and indexable. Postgres recommends `jsonb` for most uses.

The main query operators:

- **Containment** `@>` tests whether a document contains a given structure: `attrs @> '{"colour": "red"}'` or `doc @> '{"tags": ["sale"]}'`.
- **Existence** `?` tests for a top-level key: `attrs ? 'warranty_years'`.
- **Extraction** `->` returns a JSON value and `->>` returns text: `attrs ->> 'size'`.

Indexing:

- A **GIN index** (generalised inverted index, which maps each key and value to the rows containing it) speeds up containment and existence queries. The default operator class `jsonb_ops` indexes every key and every value: flexible, but larger. `jsonb_path_ops` only supports `@>` but is smaller and faster.
- An **expression index** on one key, such as `CREATE INDEX ON products ((attrs ->> 'brand'))`, is a normal B-tree on that extracted value. It is the best choice when queries target one specific key.
- A **generated column** can promote a JSON key to a real typed column that the database keeps in sync: `brand text GENERATED ALWAYS AS (attrs ->> 'brand') STORED`.

Two cautions:

- **Keep a somewhat predictable structure.** "Schemaless" in practice means the schema lives in your application code instead. Document which keys exist and what types they have, and validate on write.
- **Keep documents small.** An update to any key rewrites the whole value, and with it a new version of the row. Values over about 2 KB are compressed and moved out of line into a separate storage area (TOAST), so frequently updating a large document is expensive. Put a counter that changes every second in a real column, not inside a 50 KB JSON blob.

~~~dg-compare Two ways to store product attributes that differ by category. Real columns suit shared, queried fields; JSONB suits sparse, varying ones.
[One column per attribute]
products: id, name, price, screen_size, wattage, fabric, ...
+ Typed, constrained, compact
- Hundreds of mostly null columns
- Every new category needs a migration
[*Core columns plus JSONB]
products: id, name, price, category, attrs jsonb
+ New attributes need no migration
+ GIN or expression indexes when needed
- Types and rules enforced by your code
- Whole value rewritten on update
~~~

<div class="ex">
<div class="lab">Worked example</div>

**an online shop's product attributes**

The shop sells 2 million products in 400 categories. Every product has a name, price, category and stock count: those are real columns with constraints and indexes. Televisions have screen_size and refresh_rate; kettles have wattage and capacity; shirts have fabric and fit. No attribute applies to more than 5% of products.

As real columns, that would be over 3,000 mostly-null columns, and every new category would need a migration. Instead, `attrs jsonb` holds the category-specific attributes, typically 300 bytes per product.

The filter "televisions over 55 inches" runs as `WHERE category = 'tv' AND (attrs ->> 'screen_size')::numeric > 55`, using the category index to narrow to 40,000 rows first. When marketing wants "brand" filtering everywhere, the team adds an expression index on `attrs ->> 'brand'`, then later promotes brand to a generated column once every category uses it.

</div>

### The anti-pattern: entity-attribute-value

A common alternative is a table of `(entity_id, attribute_name, value)` rows, one row per attribute. It looks flexible but performs badly: rebuilding one product means joining or pivoting dozens of rows, values are all stored as text, and constraints are impossible. A JSON column gives the same flexibility with far less pain.

## Option 4: schemas that evolve (for data between services)

Data passed between services, written to logs, or stored in files outlives any single deploy. A producer and its consumers often run different versions of the code at the same time. Serialisation formats such as Protocol Buffers, Thrift and Avro are designed so different schema versions can still read each other's data:

- **Protobuf and Thrift** tag every field with a number, and the number is what goes on the wire. New fields get new numbers; old readers skip tags they don't recognise. You may rename a field freely (names are not encoded), but you must **never change a field's type or reuse a tag number**, because old data with that tag would be misread. Mark removed tags as `reserved`.
- **Avro** has no tags. Data is written without field names or numbers, and the reader needs the **writer's schema** to decode it, matching fields by name against its own schema. A new field must have a **default value**, so a new reader can fill it in when reading old data. You can only remove a field that had a default, so an old reader can fill it in when reading new data.

The general rules:

- Add new fields as **optional**, with defaults.
- **Never repurpose** an old field for a new meaning, or reuse its name or number.
- When a change does require coordination, **roll out readers before writers**, so nothing ever sees data it can't understand.
- For a truly breaking change, create a new message type or version and run both until everyone has moved.

Lesson 02.08 applies the same rules to events and schema registries.

<div class="ex">
<div class="lab">Worked example</div>

**evolving a protobuf message safely**

Version 1: `message User { string id = 1; string name = 2; string email = 3; }`.

The team adds a phone number: `string phone = 4;`. Old services reading new messages skip tag 4. New services reading old messages see an empty phone. Both keep working, in any deploy order.

Later they drop email in favour of a contacts list. They delete field 3 and write `reserved 3; reserved "email";`, then add `repeated Contact contacts = 5;`. A developer who later tries to reuse tag 3 for a different string gets a compile error instead of silently misreading years of stored data.

</div>

~~~dg-matrix Where each kind of field belongs. Real columns win when the database must understand the field; documents win when only the application does.
Need | Real column | JSONB | Separate index table
Filter or sort often | yes | partial, with index | yes
Join or foreign key | yes | no | partial
Constraints (unique, not null) | yes | no | partial
Sparse, varies by row | no | yes | partial
Add without migration | partial, often instant | yes | yes, build in background
Updated independently, often | yes | no, rewrites value | yes
~~~

## Common mistakes

- **Putting everything in JSON** because migrations feel scary, then reimplementing constraints and types badly in application code.
- **Joining or filtering heavily on JSON keys** without an index, turning queries into full scans.
- **Large, frequently updated documents**, which rewrite kilobytes for every small change.
- **Reusing a protobuf tag or Avro field name** for a new meaning.
- **Entity-attribute-value tables** in place of a JSON column.

## In an interview

When requirements say "users can define custom fields" or "each category has different attributes", say you will keep core fields as columns and put the variable part in a JSONB column (or a document store), with expression or GIN indexes for the few keys you filter on. For data passed between services, mention a schema format with compatibility rules and additive changes only.

<div class="take">
<div class="lab">Key takeaway</div>

keep core, frequently queried, constrained fields as real columns, since adding one is often instant now. Put sparse or fast-changing attributes in JSONB with targeted indexes, and use evolvable schemas (protobuf, Avro) with strict additive-only rules for data passed between systems.

</div>

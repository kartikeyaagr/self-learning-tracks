---
id: "14.13"
course: "system-design-101"
module: "14"
slug: "multi-tenant-design"
title: "Multi-tenant design"
keyIdeas: ["Multi-tenant systems serve many customers on shared infrastructure.", "**Pool model**: shared tables with a tenant_id column. Cheapest and easiest to operate, but every query must filter by tenant (row-level security helps), and noisy neighbours affect each other.", "**Silo model**: separate database or stack per tenant. Strong isolation and per-tenant customization, but costly and harder to operate at thousands of tenants.", "**Bridge**: mix, e.g. silo for big enterprise customers, pool for small ones.", "Limit noisy neighbours with per-tenant quotas and rate limits. Shuffle sharding assigns each tenant a random small subset of servers, so one bad tenant only affects the few tenants sharing its exact subset."]
question: "Shared tables with tenant_id vs a database per tenant: tradeoffs?"
answer: "Shared tables with a tenant_id are cheapest for many small tenants, make onboarding an insert, need one schema migration for everyone, and make cross-tenant analytics easy. But isolation depends on every query filtering correctly (row-level security and scoped credentials are the backstop), one tenant's heavy load can slow everyone, and per-tenant restore, data residency and per-tenant keys are hard. A database per tenant gives strong isolation even when code has bugs, no noisy neighbours, easy per-tenant backup, restore, residency and cost tracking, and suits regulated customers. But it wastes money on idle small tenants, onboarding means provisioning, and every migration runs thousands of times and can leave tenants on mixed versions. A common answer is to pool by default and silo large or regulated tenants."
sources: [{"title": "AWS Builders' Library: Workload isolation using shuffle sharding", "url": "https://builder.aws.com/content/3F06NpJ8YeoIGP8VHTw4n81pFn8/workload-isolation-using-shuffle-sharding", "note": "Limiting noisy-neighbour blast radius."}, {"title": "AWS: SaaS tenant isolation strategies", "url": "https://docs.aws.amazon.com/whitepapers/latest/saas-tenant-isolation-strategies/saas-tenant-isolation-strategies.html", "note": "Silo, pool, bridge."}]
---

## What multi-tenancy means

A **tenant** is a customer organisation in a software-as-a-service (SaaS) product: a company using Slack, a shop on Shopify, a team on Jira. A **multi-tenant** system serves many tenants from one codebase and, to a varying degree, one set of infrastructure. The alternative, a separate installation per customer, is how enterprise software used to be sold, and it is why upgrades took years.

Multi-tenancy is an economic decision first. Sharing infrastructure lets you serve 10,000 small tenants for a fraction of what 10,000 separate stacks would cost, and ship one change to all of them at once. But sharing creates three engineering problems that every design must answer:

1. **Isolation**: tenant A must never see or change tenant B's data, even when code has a bug.
2. **Noisy neighbours**: one tenant's heavy usage must not degrade everyone else.
3. **Operations per tenant**: onboarding, backup and restore, migration, data residency and billing for each tenant.

## Tenant isolation is more than login

The key point comes first: **authentication and authorisation are not isolation**. Getting past the login screen or API gateway does not mean tenant A's code path cannot touch tenant B's data. If a handler builds a query from a request parameter and forgets the tenant filter, an authenticated user of tenant A reads tenant B's rows. Isolation means each request is limited to its own tenant's resources **all the way down**, so a bug or a malicious query cannot cross over.

## Silo, pool and bridge

There are three basic models, distinguished by what tenants share.

~~~dg-compare The three tenancy models: what tenants share, and what that buys or costs.
[Silo]
Tenant A -> App A -> Database A
+ strongest isolation, no noisy neighbours
+ per-tenant cost, backup and residency are easy
- idle capacity per tenant
- onboarding means provisioning
- N deployments and N migrations
[*Pool]
Tenants A, B, C -> shared app -> shared DB
+ cheapest, capacity follows total load
+ onboarding is inserting a row
+ one deploy, one migration
- isolation depends on every query
- noisy neighbours, big blast radius
[Bridge]
Tenants A, B -> shared app -> DB per tenant
+ mix per layer or per tier
+ premium tenants can be siloed
- two operating models to run
~~~

**Silo**: each tenant gets dedicated resources: its own database, its own stack, or even its own cloud account or virtual network.

- Good: meets strict compliance, **no noisy neighbours**, easy cost per tenant, small blast radius (one tenant's outage stays with them), per-tenant encryption keys and regions are natural.
- Bad: cost (idle resources per tenant), heavy onboarding (provisioning infrastructure for each new tenant), slower to roll out changes, monitoring spread across many places, and limits on how many accounts or databases you can operate. Twenty siloed tenants may be manageable; a thousand hurts.

**Pool**: all tenants share the same infrastructure, and data is separated by a tenant ID column or key prefix.

- Good: cost efficiency (capacity follows total load, not the sum of peaks), one place to deploy and operate, faster to ship, easy onboarding.
- Bad: **noisy neighbours**, harder cost attribution, **a bigger blast radius** (an outage hits everyone), and customers in regulated industries may refuse to share.

**Bridge**: a mix, decided per layer or per service. For example, a shared web and application tier with a separate database (or schema) per tenant.

**Tiering** packages these as plans: free and basic tenants in the pool, premium or regulated tenants in silos, perhaps with their own region. The same code runs in both; a routing layer looks up each tenant's placement.

### A middle option: schema per tenant

Between shared tables and a database per tenant sits **schema per tenant**: one database server, with one schema (namespace) per tenant holding its own copy of every table. It gives cleaner separation than a `tenant_id` column and easy per-tenant export, without a server per tenant. But a PostgreSQL server with 10,000 schemas × 200 tables has 2 million tables, which strains the catalogue, backups and migration tooling. It works up to hundreds or low thousands of tenants per server.

## Shared tables versus database per tenant

~~~dg-matrix Shared tables with tenant_id versus a database per tenant, property by property.
Property | Shared tables (pool) | Database per tenant (silo)
Cost for many small tenants | yes, lowest | no, overhead per tenant
Onboarding speed | yes, insert a row | partial, create and migrate
One-step schema migration | yes | no, runs per tenant
Isolation if code forgets a filter | no, unless row-level security | yes
Noisy neighbour contained | no | yes
Per-tenant backup and restore | no, hard to extract | yes
Data residency, own keys | partial, needs partitioning | yes
Cross-tenant analytics | yes | partial, needs a pipeline
~~~

Two rows deserve more explanation.

**Migrations.** With shared tables, one `ALTER TABLE` covers everyone (done safely, as in lesson 14.05). With 5,000 tenant databases, a migration is 5,000 runs. Some will fail halfway (a lock timeout on one large tenant, a network blip), leaving the fleet on mixed schema versions. The application must cope with both versions until every database catches up, and you need tooling that tracks each tenant's schema version and retries.

**Per-tenant restore.** A tenant deletes half its data by mistake and wants yesterday back. With a database per tenant, restore that database. With shared tables, you restore the whole shared backup to a scratch server, extract that tenant's rows, and merge them back without touching anyone else's newer data: hours of careful work.

Many systems use the pool by default and move big or regulated tenants to silos.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of siloing small tenants**

You have 4,000 tenants. 3,900 are small (under 1 GB, a few requests per second); 100 are large.

Silo for everyone: the smallest managed database instance costs about \$50 a month, so 4,000 × \$50 = \$200,000 a month, most of it idle.

Pool for the 3,900 small tenants on a sharded cluster of 6 databases at \$1,500 each (\$9,000), plus silos for the 100 large ones at \$400 each (\$40,000): \$49,000 a month. Roughly a quarter of the cost, and the 100 tenants who most need isolation still get it.

</div>

## Making the pool safe

Enforce the tenant boundary **below** application code, so a forgotten filter cannot leak data:

- **Tenant context from the token.** The JWT (or session) carries the tenant ID, signed by the identity service; services take it from there, never from a request parameter or body field the user can change.
- **Database-enforced filters.** PostgreSQL **row-level security**: set `app.current_tenant` per connection or transaction, and a policy `USING (tenant_id = current_setting('app.current_tenant'))` means even a query that forgets `WHERE tenant_id = …` sees only its tenant's rows.
- **Scoped credentials.** For DynamoDB, an IAM condition on `dynamodb:LeadingKeys` restricts a request to items whose partition key starts with its tenant ID. Generate these policies on the fly from templates, filling in the tenant ID from the token, instead of creating thousands of static policies.
- **Tenant ID in every key.** Cache keys, object storage paths, search index filters and queue messages all carry the tenant ID. A cache key of `user:42` instead of `t:acme:user:42` is a classic cross-tenant leak.

~~~dg-seq Each request carries its tenant, and the database enforces it even when the code forgets the filter.
Client > Gateway : request + JWT (tenant=acme)
Gateway > Service : forward, tenant context = acme
Service > Token manager : credentials scoped to acme
Token manager -> Service : temporary credentials (acme rows only)
Service > Database : SET app.current_tenant = acme
Service > Database : SELECT * FROM orders (no filter)
Database -> Service : only acme rows (row-level security)
~~~

A watch-out with connection pools: if the tenant setting is set per connection and a pooled connection is reused for another tenant without being reset, the next request runs as the wrong tenant. Set it per transaction (`SET LOCAL`) so it is cleared automatically on commit.

## Noisy neighbours

In the pool, one tenant running a giant export, a badly written integration or a sales event can consume the shared database, cache or workers. Defences, from cheapest:

- **Per-tenant rate limits and quotas** (lesson 14.11) at the gateway, by plan.
- **Per-tenant concurrency limits** on expensive work: at most 4 export jobs per tenant at once.
- **Fair queuing** for background work: one queue per tenant served round-robin, so a tenant with 1 million queued jobs does not starve a tenant with 10.
- **Per-tenant metrics** (requests, CPU seconds, rows scanned) so you can see who is causing load, and bill for it.
- **Move the tenant**: relocate a consistently heavy tenant to its own shard or silo.

## Shuffle sharding: limiting the blast radius

Sharding the pool limits how many tenants one problem can hurt. Shuffle sharding does much better than plain shards with the same hardware.

Start with 8 workers and no sharding: one "poison" request that crashes workers, retried by the client, can crash them one after another and take down **everyone**. **Regular sharding** into 4 shards of 2 workers limits a bad tenant to its shard: 25% of tenants affected.

**Shuffle sharding** gives each tenant its own random pair of workers out of the 8. There are C(8,2) = **28** possible pairs. Two tenants may share one worker, but rarely both. If a bad tenant kills its two workers, any other tenant still has at least one healthy worker, as long as its client retries on the other one. Only tenants with that exact pair, about 1/28 or 3.6%, lose both. That is **7 times better** than regular sharding, using the same 8 workers.

~~~dg-bars Share of tenants that lose all their workers when one tenant's poison request kills its workers: 8 workers in the first three rows, Route 53's 2,048 name servers in the last.
No sharding | 100 | 100%
Regular sharding, 4 x 2 | 25 | 25%
*Shuffle sharding, pairs of 8 | 3.6 | 3.6%
Shuffle sharding, 4 of 2,048 | 0.0000000001 | about 0%
~~~

~~~dg-tree Shuffle sharding with 8 workers: tenants overlap on single workers but almost never on their whole set.
Workers 1 to 8
  Rainbow | workers 1 and 4
  Rose | workers 1 and 8
  Sky | workers 4 and 6
  Moss | workers 2 and 7
  *Poison from Rainbow kills 1 and 4 | Rose keeps 8, Sky keeps 6, Moss untouched
~~~

At scale it gets dramatic. Route 53 has 2,048 virtual name servers and gives each domain 4 of them: C(2048,4) ≈ **730 billion** combinations, enough for every domain to have a unique set, and no two domains share more than 2. A DDoS against one domain saturates its 4 servers, while every other domain keeps at least 2 healthy ones. The attacked domain can then be moved to dedicated capacity. It costs nothing extra: it is just a smarter way to assign existing capacity.

<div class="ex">
<div class="lab">Worked example</div>

**checking the shuffle-shard arithmetic**

With 8 workers and pairs, the number of distinct pairs is 8 × 7 ÷ 2 = 28. If tenants are spread evenly across pairs, the fraction sharing Rainbow's exact pair is 1/28 ≈ 3.6%.

The fraction sharing exactly one worker with Rainbow: pairs that contain worker 1 or worker 4 but not both. Each has 6 other partners besides each other, so 6 + 6 = 12 pairs, 12/28 ≈ 43%. Those tenants lose one worker and see some failed first attempts, but retries on the surviving worker succeed.

The other 15 pairs (54%) share nothing and are unaffected. Compare regular sharding, where 25% are fully down and 75% unaffected: shuffle sharding trades a larger partial impact for a far smaller total outage.

</div>

## Common mistakes

- **Tenant ID from the request body** rather than from the signed token.
- **Relying on every developer to remember `WHERE tenant_id`** with no database-level backstop.
- **Cache, search or storage keys without the tenant ID**, leaking data across tenants.
- **Siloing everyone too early**, then drowning in thousands of databases to migrate.
- **No per-tenant metrics**, so a noisy neighbour cannot be identified or billed.
- **Pooled connections that keep the previous tenant's setting.**

## In an interview

Ask about tenant count and size distribution, and about compliance needs. Then propose pool by default with a tenant ID from the token, row-level security as a backstop and tenant IDs in every key; silos for large or regulated tenants behind a routing layer; per-tenant limits and fair queues for noisy neighbours; and shuffle sharding to shrink blast radius.

<div class="take">
<div class="lab">Key takeaway</div>

pool tenants for cost and speed, silo them for isolation, compliance and blast radius, and mix the two by tier or layer. In the pool, enforce the tenant boundary below the code (tenant from the token, row-level security, scoped credentials, tenant IDs in every key), contain noisy neighbours with per-tenant limits, and use shuffle sharding so one bad tenant can only hurt a tiny fraction of the others.

</div>

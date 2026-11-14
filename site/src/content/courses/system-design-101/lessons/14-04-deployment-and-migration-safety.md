---
id: "14.04"
course: "system-design-101"
module: "14"
slug: "deployment-and-migration-safety"
title: "Deployment and migration safety"
keyIdeas: ["Most outages come from changes, so make deploys safe.", "Roll out gradually: canary to one host or a small percentage, then waves across zones and regions, with bake time and automatic rollback on metric alarms.", "Use feature flags to separate deploy from release.", "Rollback safety: if version N+1 writes data that version N can't read, rolling back breaks things. Fix with two-phase changes: first deploy readers that understand the new format, then deploy writers that produce it.", "Keep every change reversible, and make the rollback path tested."]
question: "Why can a new version that writes a new data format make rollback unsafe?"
answer: "During a rolling or canary deployment, version N and N+1 run at the same time, and after a rollback version N must read whatever N+1 already wrote. If N+1 writes a format that N cannot read (compressed data, JSON instead of XML, a new required field), N hosts fail on that data mid-rollout, and rolling back leaves data the running code cannot read: the rollback does not undo the data. The fix is two phases: first deploy a version that reads both formats but still writes the old one, confirm it is on every host and let it bake, then deploy the version that writes the new format. Rolling back the second phase is safe because the first-phase version can read both."
sources: [{"title": "AWS Builders' Library: Ensuring rollback safety during deployments", "url": "https://builder.aws.com/content/3F04j2yRAAMBuPSPs50xwXZqg01/ensuring-rollback-safety-during-deployments", "note": "Two-phase deploys."}, {"title": "AWS Builders' Library: Automating safe, hands-off deployments", "url": "https://builder.aws.com/content/3ErTKQOTKc5NIw031UePBPxTQ6I/automating-safe-hands-off-deployments", "note": "Waves, bake time, auto rollback."}]
---

## Most outages come from change

A system that nobody touches tends to keep working. Most outages in mature systems are triggered by a change: a code deploy, a configuration push, a schema migration, a certificate rotation, an OS patch. That makes the deployment process the single biggest lever on reliability. The aim is not to stop changing things (that would stop the business). The aim is to make every change **small, gradual, observable and reversible**.

Amazon puts the principle as "avoid walking through one-way doors". A two-way door is a change you can walk back through: a code deploy you can roll back in minutes. A one-way door is a change you cannot easily undo: dropping a column, deleting data, sending a million emails, writing data in a format old code cannot read. Prefer two-way doors, make walking back through them fast and automatic, and when you must go through a one-way door, do it slowly and deliberately.

## Deployment strategies

Before looking at a full pipeline, it helps to know the three basic ways to replace version N with version N+1 on a fleet.

~~~dg-compare Three ways to replace the running version: they trade cost, speed of rollback and how long two versions coexist.
[Rolling]
Replace hosts in batches, e.g. 33% at a time
+ no extra capacity needed
+ simple, the default in most orchestrators
- old and new versions run together for the whole rollout
- rollback is another rolling deploy, minutes
[Blue/green]
Run a full second fleet, switch traffic at the load balancer
+ instant switch and instant switch back
+ new fleet tested before it gets traffic
- double capacity during the switch
- all users move at once, so a bug hits everyone
[*Canary]
Send a small share of traffic to N+1, compare with N
+ a bug hits 1% of users, not 100%
+ automated metrics comparison decides
- needs good per-version metrics
- mixed versions run together
~~~

- **Rolling deployment**: take a batch of hosts out of the load balancer, upgrade them, health-check them, put them back, repeat. Kubernetes Deployments do this by default (`maxUnavailable`, `maxSurge`).
- **Blue/green**: two complete environments. "Blue" serves traffic while "green" gets the new version and is tested; the load balancer or DNS then flips all traffic to green. Blue stays warm for a quick flip back. Its catch: if the new version changes the database, flipping back does not undo that.
- **Canary**: named after the canaries miners carried to detect gas. A small slice of production traffic (1%, then 5%, then 25%) goes to the new version, and an automated comparison of its error rate and latency against the old version decides whether to continue. Most large companies combine canaries with rolling batches.

A fourth tool sits beside all three: **feature flags**. Deploying the code and turning the feature on become separate steps. The new code ships "dark" (present but switched off), and the flag is turned on for 1% of users, then more. Turning a feature off is then a config change taking seconds, not a deploy taking an hour.

## A production pipeline in detail

Amazon sends every change through the same pipeline, whether code, infrastructure, configuration or OS patches:

1. **Source and build.** Code review is the last manual approval. Builds compile, run unit tests, and package and sign the artifact.
2. **Pre-production.** *Alpha* and *beta* stages run functional and integration tests. *Gamma* is the most production-like stage: same deployment configuration, same alarms, continuous synthetic "canary" traffic, deployed in several regions.
3. **Gamma one-box.** The new version goes to one host (or container, or a slice of Lambda traffic) while the rest stays on the old version for 30 to 60 minutes. This catches problems that only appear when old and new versions run side by side.
4. **Production in waves.** The goal is never to hurt two regions, or two Availability Zones (AZs, separate datacentres within a region) in one region, at once.
   - **Wave 1**: a low-traffic region, one AZ or cell at a time.
   - **Wave 2**: a high-traffic region, where rare code paths actually get exercised.
   - **Later waves** grow: 3 regions, then up to 12, then the rest. Each still deploys to only one AZ per region at a time.
5. Each wave starts with a **one-box** stage taking at most about 10% of the region's or AZ's traffic, then a **rolling deployment** that replaces at most 33% of hosts at a time.

~~~dg-timeline One change moving through production waves: the blast radius only widens after the previous step has baked.
Day 1 09:00 | Gamma one-box, bake 1 h
Day 1 11:00 | Wave 1: one-box in a small region, bake 1 h
Day 1 13:00 | Wave 1 rolling, one AZ at a time
Day 2 01:00 | *12 h bake complete, no alarms
Day 2 10:00 | Wave 2: one big region, bake 2 to 4 h
Day 3 | Waves 3 and 4: 3 regions, then 12
Day 4 to 5 | Remaining regions; change fully deployed
~~~

**Bake time** gives slow problems time to show up: memory leaks that take hours to exhaust RAM, daily batch jobs, rare code paths, timezone bugs at midnight. Typical minimums are 1 hour after each one-box, 12 hours after the first regional wave, and 2 to 4 hours after each later wave. A bake can also require a minimum amount of traffic, such as at least 100 calls to the Create API, to prove the new code actually ran. A normal change reaches all regions in about **4 to 5 business days**. Emergency fixes take a faster, approved path.

**Automatic rollback.** Every service has a high-severity alarm in each region covering errors, p50/p90/p99 latency, CPU, memory, disk, error log lines and health checks. These roll up into one aggregate alarm per team, together with the synthetic canary alarms. If it fires, every in-flight deployment for that team in that region rolls back to the previous artifact without a human. One-box stages get alarms scoped to just that box, because 10% of traffic may not move the service-wide numbers enough to notice.

**Blockers and windows.** The pipeline will not start a deployment while the team's alarm is firing. It deploys only when on-call engineers are likely to be awake and at work: not at night, at weekends or on holidays, and for many teams not on Friday afternoons. But windows that are too narrow make changes pile up into large, risky bundles, which is worse.

<div class="ex">
<div class="lab">Worked example</div>

**what a canary catches that a test misses**

A new version adds a cache lookup on the product page. Tests pass. In the one-box stage it serves 10% of one AZ's traffic, about 400 requests per second.

After 25 minutes the one-box alarm fires: its p99 is 900 ms against 250 ms on the old hosts, because the cache client opens a new TCP connection per request under real concurrency, something the single-threaded integration test never did.

The pipeline rolls the box back automatically. Users affected: 10% of one AZ in one small region for 25 minutes, roughly 0.1% of global traffic. Without the one-box stage, a rolling deploy would have reached a third of the fleet before anyone looked.

</div>

## When rollback itself is unsafe

Here is the trap. Rollback is only safe if the old version can still work with everything the new version did. In a distributed system, a rolling or canary deployment means **old and new versions run at the same time**, reading each other's data and talking to each other over the network. After a rollback, the old version must also read anything the new version wrote while it ran.

The most common problem is a **protocol or format change**. Two examples:

- **Compression.** V2 compresses data it writes to disk or cache. V1 cannot read compressed data. During the rollout, V1 hosts reading data that V2 hosts wrote fail with parse errors. Roll back to V1 and every record V2 wrote is now unreadable by every running host.
- **Heartbeats.** Servers send heartbeats every 5 s and drop peers that are silent for 5 s. "Just changing a number" to 10 s means new servers go quiet for 10 s, so old servers drop connections to new ones across the fleet, and the cluster partitions itself.

~~~dg-seq Why rollback breaks: V2 writes a format V1 cannot read, and the data outlives the rollback.
V2 host > Store : write order 42 as JSON
V1 host > Store : read order 42
Store -> V1 host : JSON bytes
V1 host > V1 host : parse as XML fails, 500 error
Operator > V2 host : roll back to V1
V1 host > Store : read order 42 again
Store -> V1 host : still JSON, still fails
~~~

## The two-phase deployment

The fix is to split a format change into two deployments, each of which is safe to roll back on its own. For example, moving stored data from XML to JSON:

| Phase | Version | Reads | Writes | Why rollback is safe |
| --- | --- | --- | --- | --- |
| Prepare | V2 | XML and JSON | XML | No JSON exists yet, so V1 is fine |
| Activate | V3 | XML and JSON | JSON | V2 can read JSON, so rolling V3 back to V2 is fine |

~~~dg-state The two-phase change as fleet states: each arrow forward can be undone by the arrow back.
[*] -> All V1
All V1 -> Mixed V1/V2 : deploy Prepare
Mixed V1/V2 -> All V1 : roll back, no JSON exists
Mixed V1/V2 -> *All V2 : 100% of hosts, bake days
All V2 -> Mixed V2/V3 : deploy Activate
Mixed V2/V3 -> All V2 : roll back, V2 reads JSON
Mixed V2/V3 -> All V3 : bake, backfill old XML
~~~

The heartbeat version works the same way. In Prepare, servers accept up to 10 s of silence but still send every 5 s. In Activate, they start sending every 10 s.

Precautions:

- **Prepare must reach 100% of hosts** before Activate starts. A deploy tool that calls a deployment "successful" once most hosts are healthy is not enough; one host that missed Prepare will fail to read JSON. Check the version on every host, including ones that were out of rotation.
- **Bake between phases** for several days. Once V3 has written JSON, rolling all the way back to V1 is unsafe again, so be sure V2 is good first.
- **Do not drop XML support afterwards** until a backfill has rewritten all old records to JSON. Then a third deploy can remove the XML reader.

### Serialisation habits that avoid the problem

- Use standard, evolvable formats (JSON, Protocol Buffers, Avro) instead of custom binary layouts.
- Store an explicit **format version** with the data, so readers know what they are parsing.
- Do not serialise classes you do not own; a library upgrade can change their layout.
- **Tolerate and keep unknown fields**: if an old version reads a record with a new field and writes it back, it should preserve that field, not erase it. Protocol Buffers do this by default. With this rule, many changes do not need two phases at all.
- New fields must be **optional** with a sensible default, so old writers that omit them do not break new readers.

### Test rollback, do not reason about it

Testing must use several hosts and the same deployment configuration as production. A one-host test environment makes every deploy atomic and hides mixed-version bugs completely. A good test is: deploy to half the fleet (old and new side by side), finish the deployment, then roll back fully, with continuous traffic hitting every API throughout and all errors counted.

<div class="ex">
<div class="lab">Worked example</div>

**adding a "currency" field to orders**

V2 adds `currency` to the stored order record and requires it when reading. Halfway through the rollout, a V1 host updates an order written by V2 and saves it without `currency` (it does not know the field), so V2 hosts reading that order fail. Rolling back also does not fix the records already written.

Safe version: Prepare (V2) reads `currency` if present and defaults to USD otherwise, preserves unknown fields on write, and does not write `currency` yet. Once V2 is on 100% of hosts, Activate (V3) starts writing it. Rolling back V3 to V2 is safe because V2 reads records with or without the field. A backfill then sets `currency` on old orders, and only after that may code treat it as required.

</div>

## Common mistakes

- **"It's just a config change."** Configuration pushes cause as many outages as code; send them through the same staged pipeline.
- **Global changes at once.** A bad flag flipped for 100% of users in every region is a global outage.
- **Rollback never tested.** The first rollback attempt happens during an incident, and it fails.
- **One-way doors inside two-way doors.** A deploy that also drops a column or rewrites data cannot be rolled back, whatever the deploy tool says.
- **Huge bundles.** Twenty changes in one deploy: when it breaks, nobody knows which one.

## In an interview

When a design involves changing a data format, an API contract or a cluster protocol, say out loud: "during the rollout both versions run, so I will deploy readers first and writers second". Mention canaries with automatic rollback, one AZ or region at a time, and feature flags to separate deploy from release. It shows you understand that a design must survive its own upgrades.

<div class="take">
<div class="lab">Key takeaway</div>

roll out in small, widening waves, one AZ or region at a time, with bake time and alarms that roll back automatically. During a rollout old and new versions run together, so any change to data formats or protocols needs two phases: first teach every reader the new format, then switch the writers.

</div>
